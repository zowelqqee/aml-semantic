"""The semantic reasoning engine.

Structure is unchanged from the v0.2 Runtime — facts, evidence, conflicts,
policies, decision — and the ``Evidence``/``Conflict``/``PolicyOutcome`` types
are the frozen ones.  What changed is the vocabulary: a fact is now a predicate
over a semantic object, so every decision is expressible in terms of what the
transaction *means* rather than which numeric threshold it crossed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..models import Conflict, Decision, DecisionRecord, Evidence, PolicyOutcome, Serializable, Transaction
from ..runtime import ConflictEngine, ConflictSpecification, stable_id
from .context import CONTEXT_VERSION, SemanticContextLayer
from .objects import SemanticContextResult, SemanticObject
from .ontology import ONTOLOGY_HASH, ONTOLOGY_VERSION, SemanticType

SEMANTIC_RUNTIME_VERSION = "aml-semantic-runtime/1.0"

#: Semantic states that make the reading *undetermined* rather than clean.
UNDETERMINED_TYPES = frozenset({SemanticType.NO_ESTABLISHED_BASELINE, SemanticType.UNSCALED_VALUE})

#: Structural readings that fully explain an event without any behaviour model.
STRUCTURAL_EXPLANATIONS = frozenset({SemanticType.INTERNAL_BOOK_ENTRY, SemanticType.INTRA_CUSTOMER_TRANSFER})

#: A single unqualified topic is a finding only from these sources.
HIGH_CONCERN_TOPICS = frozenset({"network_motif", "jurisdiction"})

#: The strength a lone high-concern topic must reach to stand on its own.
SINGLE_TOPIC_REVIEW_CONFIDENCE = 0.90

#: A model probability is a scalar over the same observations the semantic
#: layer already read, not an independent source of meaning.  It is excluded
#: from the corroboration count so that it can never manufacture the second
#: "independent" topic that SEM-P10 requires; it escalates only through the
#: explicit ML policy in ``with_ml_evidence``.
NON_SEMANTIC_TOPICS = frozenset({"ml_probability"})


@dataclass(frozen=True)
class SemanticFact(Serializable):
    """A predicate over one semantic object.  Facts no longer read CSV columns."""

    id: str
    type: SemanticType
    subject_id: str
    semantic_object_id: str
    confidence: float
    explanation: str
    timestamp: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.type.value, "subject_id": self.subject_id,
            "semantic_object_id": self.semantic_object_id, "confidence": self.confidence,
            "explanation": self.explanation, "timestamp": self.timestamp, "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SemanticRule:
    """One semantic object type, mapped to one typed evidence item."""

    id: str
    consumes: SemanticType
    direction: str
    topic: str
    source_reliability: float
    description: str


SEMANTIC_RULES: tuple[SemanticRule, ...] = (
    SemanticRule("SEM-R01-VALUE-REGIME-BREAK", SemanticType.UNEXPECTED_LARGE_TRANSFER, "risk", "value_regime", 0.90,
                 "Value left the party's own established regime."),
    SemanticRule("SEM-R02-TEMPO-REGIME-SHIFT", SemanticType.BEHAVIOUR_REGIME_SHIFT, "risk", "tempo_regime", 0.85,
                 "Tempo left the party's own established regime."),
    SemanticRule("SEM-R03-INFORMATIVE-NOVELTY", SemanticType.FIRST_CONTACT, "risk", "counterparty_regime", 0.70,
                 "A counterparty outside the party's known counterparty set."),
    SemanticRule("SEM-R04-JURISDICTION", SemanticType.HIGH_RISK_JURISDICTION_EXPOSURE, "risk", "jurisdiction", 0.95,
                 "A leg is booked in an enhanced-scrutiny jurisdiction."),
    SemanticRule("SEM-R05-VIRTUAL-ASSET", SemanticType.VIRTUAL_ASSET_EXPOSURE, "risk", "virtual_asset", 0.80,
                 "A leg is booked at a virtual-asset venue."),
    SemanticRule("SEM-R06-LAYERING", SemanticType.LAYERING_CHAIN_SEGMENT, "risk", "network_motif", 0.95,
                 "Value was received and forwarded with preservation inside a short window."),
    SemanticRule("SEM-R07-CASH-INSTRUMENT", SemanticType.CASH_INSTRUMENT_SETTLEMENT, "risk", "settlement_instrument", 0.60,
                 "Settlement carries no counterparty audit trail."),
    SemanticRule("SEM-R08-TRANSIT-ROLE", SemanticType.PASS_THROUGH_ACCOUNT, "risk", "account_role", 0.80,
                 "The originating account is a transit point rather than a store of value."),
    SemanticRule("SEM-R09-CONCENTRATION-ROLE", SemanticType.COLLECTION_NODE, "risk", "account_role", 0.75,
                 "The originating account concentrates value from many counterparties."),
    SemanticRule("SEM-R10-CURRENCY-CONVERSION", SemanticType.CURRENCY_CONVERSION_TRANSFER, "risk", "currency_conversion", 0.55,
                 "The two legs settle in different currencies."),
    SemanticRule("SEM-M01-BOOK-ENTRY", SemanticType.INTERNAL_BOOK_ENTRY, "mitigation", "structural_explanation", 0.99,
                 "The event is a self-posting; there is no counterparty."),
    SemanticRule("SEM-M02-INTRA-CUSTOMER", SemanticType.INTRA_CUSTOMER_TRANSFER, "mitigation", "structural_explanation", 0.96,
                 "The event moves value inside one legal party."),
    SemanticRule("SEM-M03-OPERATIONAL-BURST", SemanticType.NORMAL_OPERATIONAL_BURST, "mitigation", "tempo_regime", 0.90,
                 "The burst matches the party's own distribution shape."),
    SemanticRule("SEM-M04-EXPECTED-HIGH-VALUE", SemanticType.EXPECTED_HIGH_VALUE_TRANSFER, "mitigation", "value_regime", 0.88,
                 "The value is high in absolute terms but ordinary for this party."),
    SemanticRule("SEM-M05-ROUTINE-VALUE", SemanticType.ROUTINE_VALUE_TRANSFER, "mitigation", "value_regime", 0.85,
                 "The value sits inside the party's own established regime."),
    SemanticRule("SEM-M06-UNINFORMATIVE-NOVELTY", SemanticType.NON_INFORMATIVE_NOVELTY, "mitigation", "counterparty_regime", 0.90,
                 "Novelty observed against no counterparty baseline carries no information."),
    SemanticRule("SEM-M07-ESTABLISHED-RELATIONSHIP", SemanticType.ESTABLISHED_RELATIONSHIP, "mitigation", "counterparty_regime", 0.85,
                 "The counterparty relationship is one the party uses repeatedly."),
    SemanticRule("SEM-M08-BOOKKEEPING-ROLE", SemanticType.BOOKKEEPING_ACCOUNT, "mitigation", "account_role", 0.90,
                 "The account is an internal accounting location."),
)

#: Declared risk/control disagreements in the semantic vocabulary.  The v0.2
#: conflict engine measured a conflict frequency of exactly 0.0 on this source
#: because the schema carries no controls; semantic objects supply the missing
#: negative pole.
SEMANTIC_CONFLICT_PAIRS: tuple[ConflictSpecification, ...] = (
    ConflictSpecification("*", "SEM-M01-BOOK-ENTRY", "structural_explanation",
                          "Risk evidence is qualified: the event is a self-posting with no counterparty."),
    ConflictSpecification("*", "SEM-M02-INTRA-CUSTOMER", "structural_explanation",
                          "Risk evidence is qualified: value did not leave the legal party."),
    ConflictSpecification("SEM-R02-TEMPO-REGIME-SHIFT", "SEM-M03-OPERATIONAL-BURST", "tempo_regime",
                          "A tempo shift conflicts with the party's own distribution shape."),
    ConflictSpecification("SEM-R01-VALUE-REGIME-BREAK", "SEM-M07-ESTABLISHED-RELATIONSHIP", "counterparty_context",
                          "A value-regime break conflicts with a repeatedly used counterparty relationship."),
    ConflictSpecification("SEM-R03-INFORMATIVE-NOVELTY", "SEM-M05-ROUTINE-VALUE", "value_context",
                          "Counterparty novelty conflicts with a value inside the party's own regime."),
    ConflictSpecification("SEM-R07-CASH-INSTRUMENT", "SEM-M05-ROUTINE-VALUE", "value_context",
                          "Instrument opacity conflicts with a value inside the party's own regime."),
    ConflictSpecification("SEM-R10-CURRENCY-CONVERSION", "SEM-M05-ROUTINE-VALUE", "value_context",
                          "A currency conversion conflicts with a value inside the party's own regime."),
    ConflictSpecification("SEM-R05-VIRTUAL-ASSET", "SEM-M05-ROUTINE-VALUE", "value_context",
                          "Virtual-asset exposure conflicts with a value inside the party's own regime."),
    ConflictSpecification("SEM-R08-TRANSIT-ROLE", "SEM-M08-BOOKKEEPING-ROLE", "account_role",
                          "A transit role conflicts with an internal accounting role."),
    ConflictSpecification("SEM-R09-CONCENTRATION-ROLE", "SEM-M08-BOOKKEEPING-ROLE", "account_role",
                          "A concentration role conflicts with an internal accounting role."),
)


class SemanticFactExtractor:
    """Turns semantic objects into facts.  It reads no transaction fields."""

    CONSUMED = frozenset(rule.consumes for rule in SEMANTIC_RULES)

    def extract(self, transaction: Transaction, context: SemanticContextResult) -> tuple[SemanticFact, ...]:
        facts = [
            SemanticFact(
                id=stable_id("SF", transaction.id, item.type.value, item.id),
                type=item.type,
                subject_id=item.subject_id,
                semantic_object_id=item.id,
                confidence=item.confidence,
                explanation=f"{item.meaning} Supporting: {'; '.join(item.supporting_facts)}.",
                timestamp=transaction.timestamp,
                provenance=f"semantic-object/{item.origin}",
            )
            for item in context.objects
            if item.type in self.CONSUMED
        ]
        return tuple(sorted(facts, key=lambda item: (item.type.value, item.id)))


class SemanticRuleEngine:
    """Maps semantic facts to immutable evidence.  It never decides."""

    def __init__(self, rules: tuple[SemanticRule, ...] = SEMANTIC_RULES) -> None:
        self.rules = rules
        self._by_type = {rule.consumes: rule for rule in rules}

    def evaluate(self, transaction: Transaction, facts: tuple[SemanticFact, ...]) -> tuple[Evidence, ...]:
        evidence: list[Evidence] = []
        for fact in facts:
            rule = self._by_type.get(fact.type)
            if rule is None:
                continue
            evidence.append(Evidence(
                id=stable_id("E", transaction.id, rule.id, fact.id),
                source=f"semantic-context/{fact.type.value}",
                supporting_facts=(fact.id,),
                confidence=round(fact.confidence, 6),
                explanation=f"{rule.description} ({fact.explanation})",
                timestamp=transaction.timestamp,
                rule_id=rule.id,
                direction=rule.direction,
                topic=rule.topic,
                source_reliability=rule.source_reliability,
                recency_days=0,
                metadata={"semantic_object_id": fact.semantic_object_id, "semantic_type": fact.type.value},
            ))
        return tuple(sorted(evidence, key=lambda item: (item.rule_id, item.id)))


class SemanticPolicyEngine:
    """The only component permitted to select a decision.

    Every threshold below is a statement about corroboration, declared before
    the benchmark was executed and never fitted: one signal is not a finding;
    a network motif with independent corroboration is; and an undetermined
    semantic state is abstention, not clearance.
    """

    VERSION = "aml-semantic-policy/1.0"

    @staticmethod
    def _noisy_or(items: tuple[Evidence, ...]) -> float:
        complement = 1.0
        for item in items:
            complement *= 1.0 - item.confidence
        return round(1.0 - complement, 6)

    def evaluate(
        self,
        evidence: tuple[Evidence, ...],
        conflicts: tuple[Conflict, ...],
        object_types: frozenset[SemanticType],
    ) -> tuple[PolicyOutcome, ...]:
        qualified = {item.risk_evidence_id for item in conflicts}
        unqualified = tuple(item for item in evidence if item.direction == "risk" and item.id not in qualified)
        topics = {item.topic for item in unqualified} - NON_SEMANTIC_TOPICS
        strongest = max((item.confidence for item in unqualified if item.topic not in NON_SEMANTIC_TOPICS), default=0.0)
        structural = bool(object_types & STRUCTURAL_EXPLANATIONS)
        undetermined = bool(object_types & UNDETERMINED_TYPES) and not structural

        block = "network_motif" in topics and len(topics) >= 2
        single_high_concern = (
            len(topics) == 1
            and topics.issubset(HIGH_CONCERN_TOPICS)
            and strongest >= SINGLE_TOPIC_REVIEW_CONFIDENCE
        )
        review = not block and (len(topics) >= 2 or single_high_concern)
        abstain = not evidence or (not block and not review and undetermined)
        allow = bool(evidence) and not block and not review and not abstain

        metrics: dict[str, float | int | str] = {
            "policy_version": self.VERSION,
            # Reported for ranking parity with the frozen runtime; the decision
            # itself is selected by corroboration, not by this scalar.
            "effective_risk": self._noisy_or(unqualified),
            "independent_unqualified_topics": len(topics),
            "qualified_risk_count": len(qualified),
            "unqualified_risk_count": len(unqualified),
            "strongest_unqualified_confidence": round(strongest, 6),
            "semantic_state": "undetermined" if undetermined else ("structural" if structural else "determined"),
        }
        risk_ids = tuple(item.id for item in unqualified)
        return (
            PolicyOutcome("SEM-P01", Decision.BLOCK, block,
                          "Block a network layering motif corroborated by an independent semantic topic.",
                          risk_ids, metrics),
            PolicyOutcome("SEM-P10", Decision.REVIEW, review,
                          "Review when at least two independent semantic topics remain unqualified, or a single high-concern topic is strong on its own.",
                          risk_ids, metrics),
            PolicyOutcome("SEM-P20", Decision.ALLOW, allow,
                          "Allow when the semantic reading is determined and no unqualified corroboration remains.",
                          tuple(item.id for item in evidence), metrics),
            PolicyOutcome("SEM-P00", Decision.ABSTAIN, abstain,
                          "Abstain when the semantic state is undetermined: no baseline exists for this party, or the value has no reference frame.",
                          (), metrics),
        )


def select_decision(policies: tuple[PolicyOutcome, ...]) -> DecisionRecord:
    triggered = tuple(item for item in policies if item.triggered)
    selected = next(
        (decision for decision in (Decision.BLOCK, Decision.REVIEW, Decision.ALLOW, Decision.ABSTAIN)
         if any(item.outcome == decision for item in triggered)),
        Decision.ABSTAIN,
    )
    policy_ids = tuple(item.policy_id for item in triggered if item.outcome == selected)
    rationale = next((item.explanation for item in triggered if item.outcome == selected), "No policy was triggered.")
    return DecisionRecord(selected, rationale, policy_ids)


@dataclass(frozen=True)
class SemanticRuntimeResult:
    transaction: Transaction
    context: SemanticContextResult
    facts: tuple[SemanticFact, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    policies: tuple[PolicyOutcome, ...]
    decision: DecisionRecord
    ml_evidence: Evidence | None = None
    routed_to_ml: bool = False

    @property
    def semantic_rationale(self) -> str:
        """The decision, said in semantic objects rather than rule identifiers."""
        types = [item.type.value for item in self.context.objects if item.object_class.value == "event"]
        return f"{self.decision.decision.value}: {self.decision.rationale} Reading: {', '.join(types) or 'no event-level reading'}."

    def audit_record(self) -> dict[str, Any]:
        return {
            "semantic_runtime_version": SEMANTIC_RUNTIME_VERSION,
            "transaction": self.transaction.to_dict(),
            "semantic": {
                "ontology_version": ONTOLOGY_VERSION,
                "ontology_hash": ONTOLOGY_HASH,
                "context_version": CONTEXT_VERSION,
                "context_state_hash": self.context.context_state_hash,
                "objects": [item.to_dict() for item in self.context.objects],
                "withheld": [item.to_dict() for item in self.context.withheld],
                "coverage_gaps": list(self.context.coverage_gaps),
            },
            "facts": [item.to_dict() for item in self.facts],
            "evidence": [item.to_dict() for item in self.evidence],
            "ml_evidence": self.ml_evidence.to_dict() if self.ml_evidence else None,
            "routed_to_ml": self.routed_to_ml,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "policies": [item.to_dict() for item in self.policies],
            "decision": self.decision.to_dict(),
            "semantic_rationale": self.semantic_rationale,
        }


class SemanticDecisionRuntime:
    """Semantic Context Layer -> facts -> evidence -> conflicts -> policy -> decision."""

    def __init__(self, context: SemanticContextLayer) -> None:
        self.context = context
        self.facts = SemanticFactExtractor()
        self.rules = SemanticRuleEngine()
        self.conflicts = ConflictEngine(SEMANTIC_CONFLICT_PAIRS)
        self.policies = SemanticPolicyEngine()

    def evaluate(self, transaction: Transaction, event_index: int) -> SemanticRuntimeResult:
        reading = self.context.observe(transaction, event_index)
        facts = self.facts.extract(transaction, reading)
        evidence = self.rules.evaluate(transaction, facts)
        conflicts = self.conflicts.detect(evidence)
        policies = self.policies.evaluate(evidence, conflicts, reading.types())
        return SemanticRuntimeResult(transaction, reading, facts, evidence, conflicts, policies, select_decision(policies))

    @staticmethod
    def routes_to_ml(result: SemanticRuntimeResult) -> bool:
        """ML is asked only about events the layer has admitted it cannot characterise."""
        return (
            result.decision.decision is Decision.ABSTAIN
            and not (result.context.types() & STRUCTURAL_EXPLANATIONS)
        )

    def with_ml_evidence(self, result: SemanticRuntimeResult, ml_evidence: Evidence, high_band: float) -> SemanticRuntimeResult:
        """Re-run conflicts and policy over semantic evidence plus one ML item.

        The model cannot select a decision.  Its probability becomes evidence,
        is exposed to the same conflict engine as any other evidence, and can
        only lift an abstention to REVIEW through this declared policy.
        """
        combined = result.evidence + (ml_evidence,)
        conflicts = self.conflicts.detect(combined)
        policies = self.policies.evaluate(combined, conflicts, result.context.types())
        base = select_decision(policies)
        qualified = any(item.risk_evidence_id == ml_evidence.id for item in conflicts)
        probability = float(ml_evidence.metadata.get("probability", ml_evidence.confidence))
        metrics: dict[str, float | int | str] = {
            "ml_probability": probability, "ml_high_band": high_band,
            "ml_evidence_qualified_by_conflict": int(qualified),
        }
        if base.decision in (Decision.BLOCK, Decision.REVIEW):
            outcome = PolicyOutcome("SEM-ML-01", base.decision, True,
                                    "The semantic reading already selected this decision; ML evidence adds prioritisation only.",
                                    (ml_evidence.id,), metrics)
        elif not qualified and probability >= high_band:
            outcome = PolicyOutcome("SEM-ML-02", Decision.REVIEW, True,
                                    "An undetermined semantic state plus unqualified high-band ML evidence is escalated to REVIEW.",
                                    (ml_evidence.id,), metrics)
        else:
            outcome = PolicyOutcome("SEM-ML-03", base.decision, True,
                                    "ML evidence is below the declared band or qualified by conflicting semantic evidence; the semantic decision stands.",
                                    (ml_evidence.id,), metrics)
        final = DecisionRecord(outcome.outcome, outcome.explanation, (outcome.policy_id,))
        return SemanticRuntimeResult(
            result.transaction, result.context, result.facts, combined, conflicts,
            policies + (outcome,), final, ml_evidence, True,
        )


def semantic_replay_pins(resolver_hash: str, result: SemanticRuntimeResult) -> dict[str, str]:
    """Everything needed to reproduce this decision byte-for-byte."""
    rules_payload = "|".join(f"{rule.id}:{rule.consumes.value}:{rule.direction}:{rule.topic}:{rule.source_reliability}" for rule in SEMANTIC_RULES)
    objects_payload = "|".join(item.id for item in result.context.objects)
    return {
        "ontology_hash": ONTOLOGY_HASH,
        "inference_rules_hash": hashlib.sha256(CONTEXT_VERSION.encode("utf-8")).hexdigest(),
        "semantic_rules_hash": hashlib.sha256(rules_payload.encode("utf-8")).hexdigest(),
        "policy_hash": hashlib.sha256(SemanticPolicyEngine.VERSION.encode("utf-8")).hexdigest(),
        "entity_snapshot_hash": resolver_hash,
        "context_state_hash": result.context.context_state_hash,
        "semantic_object_set_hash": hashlib.sha256(objects_payload.encode("utf-8")).hexdigest(),
        "input_snapshot_hash": hashlib.sha256(repr(result.transaction.to_dict()).encode("utf-8")).hexdigest(),
    }
