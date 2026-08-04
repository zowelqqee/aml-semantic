"""The fraud semantic reasoning engine.

Structurally identical to ``aml_runtime.semantic.runtime``: facts, evidence,
conflicts, policy, decision — with ``Evidence``, ``Conflict``,
``ConflictSpecification``, ``PolicyOutcome``, ``DecisionRecord``, ``Decision``
and ``ConflictEngine`` retained in the local generic core. Only the
vocabulary — which topics exist, which pairs qualify which, what counts as
high-concern on its own — is fraud-native.

One structural difference from AML is stated rather than hidden: AML had
``InternalBookEntry`` / ``IntraCustomerTransfer`` — event types that fully
explain a transaction on their own (no counterparty, or value staying inside
one legal party) and therefore short-circuit both risk and ML routing. A card
purchase has no one-directional analogue to "this wasn't really a transfer" —
every purchase has a real counterpart charge. So ``STRUCTURAL_EXPLANATIONS``
is declared empty here, not populated with a forced analogue.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ...core import Conflict, Decision, DecisionRecord, Evidence, PolicyOutcome, ConflictEngine, ConflictSpecification, stable_id

from .context import CONTEXT_VERSION, SemanticContextLayer
from .objects import SemanticContextResult
from .ontology import ONTOLOGY_HASH, ONTOLOGY_VERSION, SemanticType

SEMANTIC_RUNTIME_VERSION = "fraud-semantic-runtime/1.0"

#: Semantic states that make the reading *undetermined* rather than clean.
UNDETERMINED_TYPES = frozenset({
    SemanticType.NO_ESTABLISHED_CARD_HISTORY, SemanticType.UNSCALED_SPENDING_AMOUNT,
    SemanticType.UNVERIFIED_DEVICE_CONTEXT,
})

#: See module docstring: no fraud analogue to AML's self-posting exists.
STRUCTURAL_EXPLANATIONS: frozenset[SemanticType] = frozenset()

#: No single fraud-domain client signature or individual semantic signal is a
#: finding on its own.  Corroboration is required.
HIGH_CONCERN_TOPICS = frozenset()

SINGLE_TOPIC_REVIEW_CONFIDENCE = 0.90

#: A model probability is a scalar over observations the layer already read,
#: not an independent semantic topic — excluded from the corroboration count
#: for the same reason aml_runtime excludes it.
NON_SEMANTIC_TOPICS = frozenset({"ml_probability"})


@dataclass(frozen=True)
class SemanticFact:
    id: str
    type: SemanticType
    subject_id: str
    semantic_object_id: str
    confidence: float
    explanation: str
    timestamp: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type.value, "subject_id": self.subject_id,
                "semantic_object_id": self.semantic_object_id, "confidence": self.confidence,
                "explanation": self.explanation, "timestamp": self.timestamp, "provenance": self.provenance}


@dataclass(frozen=True)
class SemanticRule:
    id: str
    consumes: SemanticType
    direction: str
    topic: str
    source_reliability: float
    description: str


SEMANTIC_RULES: tuple[SemanticRule, ...] = (
    SemanticRule("SEM-R01-UNEXPECTED-AMOUNT", SemanticType.UNEXPECTED_SPENDING_AMOUNT, "risk", "amount_regime", 0.90,
                 "Spend amount left the card's own established regime."),
    SemanticRule("SEM-R02-TEMPO-SHIFT", SemanticType.TEMPO_REGIME_SHIFT, "risk", "tempo_regime", 0.85,
                 "Purchase tempo left the card's own established regime."),
    SemanticRule("SEM-R03-FIRST-DEVICE", SemanticType.FIRST_DEVICE_CONTACT, "risk", "device_regime", 0.70,
                 "A device outside the card's known device set."),
    SemanticRule("SEM-R04-UNEXPECTED-REGION", SemanticType.UNEXPECTED_BILLING_REGION, "risk", "region_regime", 0.75,
                 "The billing region differs from every region previously seen on this card."),
    SemanticRule("SEM-R05-TEST-AMOUNT", SemanticType.MINIMAL_TEST_AMOUNT, "risk", "test_amount", 0.75,
                 "A charge small enough to be a live-card test rather than a purchase."),
    SemanticRule("SEM-M01-ROUTINE-AMOUNT", SemanticType.ROUTINE_SPENDING_AMOUNT, "mitigation", "amount_regime", 0.85,
                 "The amount sits inside the card's own established regime."),
    SemanticRule("SEM-M02-EXPECTED-HIGH-VALUE", SemanticType.EXPECTED_HIGH_VALUE_SPEND, "mitigation", "amount_regime", 0.88,
                 "The amount is high in absolute terms but ordinary for this card."),
    SemanticRule("SEM-M03-ESTABLISHED-DEVICE", SemanticType.ESTABLISHED_DEVICE_RELATIONSHIP, "mitigation", "device_regime", 0.85,
                 "The card-device pairing is one the card uses repeatedly."),
    SemanticRule("SEM-M04-NONINFORMATIVE-DEVICE", SemanticType.NON_INFORMATIVE_DEVICE_NOVELTY, "mitigation", "device_regime", 0.90,
                 "Device novelty observed against no device baseline carries no information."),
)

#: Declared risk/control disagreements — reachable pairs only: each pair
#: below can genuinely co-occur on one transaction's evidence set, because
#: they come from different layers (amount, tempo, device, region).
SEMANTIC_CONFLICT_PAIRS: tuple[ConflictSpecification, ...] = (
    ConflictSpecification("*", "SEM-M01-ROUTINE-AMOUNT", "amount_context",
                          "Risk evidence is qualified: the amount sits inside the card's own regime."),
    ConflictSpecification("*", "SEM-M02-EXPECTED-HIGH-VALUE", "amount_context",
                          "Risk evidence is qualified: the amount is high but ordinary for this card."),
    ConflictSpecification("SEM-R01-UNEXPECTED-AMOUNT", "SEM-M03-ESTABLISHED-DEVICE", "device_context",
                          "An amount-regime break conflicts with a well-established device relationship."),
    ConflictSpecification("SEM-R03-FIRST-DEVICE", "SEM-M01-ROUTINE-AMOUNT", "amount_context",
                          "Device novelty conflicts with an amount inside the card's own regime."),
    ConflictSpecification("SEM-R04-UNEXPECTED-REGION", "SEM-M03-ESTABLISHED-DEVICE", "device_context",
                          "An unexpected billing region conflicts with a well-established device relationship."),
    ConflictSpecification("SEM-R05-TEST-AMOUNT", "SEM-M03-ESTABLISHED-DEVICE", "device_context",
                          "A minimal test-sized amount conflicts with a well-established device relationship."),
)


class SemanticFactExtractor:
    CONSUMED = frozenset(rule.consumes for rule in SEMANTIC_RULES)

    def extract(self, transaction, context: SemanticContextResult) -> tuple[SemanticFact, ...]:
        facts = [
            SemanticFact(
                id=stable_id("SF", transaction.id, item.type.value, item.id),
                type=item.type, subject_id=item.subject_id, semantic_object_id=item.id,
                confidence=item.confidence, explanation=f"{item.meaning} Supporting: {'; '.join(item.supporting_facts)}.",
                timestamp=transaction.timestamp, provenance=f"semantic-object/{item.origin}",
            )
            for item in context.objects if item.type in self.CONSUMED
        ]
        return tuple(sorted(facts, key=lambda item: (item.type.value, item.id)))


class SemanticRuleEngine:
    def __init__(self, rules: tuple[SemanticRule, ...] = SEMANTIC_RULES) -> None:
        self.rules = rules
        self._by_type = {rule.consumes: rule for rule in rules}

    def evaluate(self, transaction, facts: tuple[SemanticFact, ...]) -> tuple[Evidence, ...]:
        evidence: list[Evidence] = []
        for fact in facts:
            rule = self._by_type.get(fact.type)
            if rule is None:
                continue
            evidence.append(Evidence(
                id=stable_id("E", transaction.id, rule.id, fact.id), source=f"semantic-context/{fact.type.value}",
                supporting_facts=(fact.id,), confidence=round(fact.confidence, 6),
                explanation=f"{rule.description} ({fact.explanation})", timestamp=transaction.timestamp,
                rule_id=rule.id, direction=rule.direction, topic=rule.topic, source_reliability=rule.source_reliability,
                recency_days=0, metadata={"semantic_object_id": fact.semantic_object_id, "semantic_type": fact.type.value},
            ))
        return tuple(sorted(evidence, key=lambda item: (item.rule_id, item.id)))


class SemanticPolicyEngine:
    """The only component permitted to select a decision. Frozen thresholds."""

    VERSION = "fraud-semantic-policy/1.0"

    @staticmethod
    def _noisy_or(items: tuple[Evidence, ...]) -> float:
        complement = 1.0
        for item in items:
            complement *= 1.0 - item.confidence
        return round(1.0 - complement, 6)

    def evaluate(self, evidence: tuple[Evidence, ...], conflicts: tuple[Conflict, ...],
                 object_types: frozenset[SemanticType]) -> tuple[PolicyOutcome, ...]:
        qualified = {item.risk_evidence_id for item in conflicts}
        unqualified = tuple(item for item in evidence if item.direction == "risk" and item.id not in qualified)
        topics = {item.topic for item in unqualified} - NON_SEMANTIC_TOPICS
        strongest = max((item.confidence for item in unqualified if item.topic not in NON_SEMANTIC_TOPICS), default=0.0)
        structural = bool(object_types & STRUCTURAL_EXPLANATIONS)
        undetermined = bool(object_types & UNDETERMINED_TYPES) and not structural

        block = "device_motif" in topics and len(topics) >= 2
        single_high_concern = len(topics) == 1 and topics.issubset(HIGH_CONCERN_TOPICS) and strongest >= SINGLE_TOPIC_REVIEW_CONFIDENCE
        review = not block and (len(topics) >= 2 or single_high_concern)
        abstain = not evidence or (not block and not review and undetermined)
        allow = bool(evidence) and not block and not review and not abstain

        metrics: dict[str, float | int | str] = {
            "policy_version": self.VERSION, "independent_unqualified_topics": len(topics),
            "qualified_risk_count": len(qualified), "unqualified_risk_count": len(unqualified),
            "strongest_unqualified_confidence": round(strongest, 6),
            "semantic_state": "undetermined" if undetermined else ("structural" if structural else "determined"),
            "effective_risk": self._noisy_or(unqualified),
        }
        risk_ids = tuple(item.id for item in unqualified)
        return (
            PolicyOutcome("SEM-P01", Decision.BLOCK, block,
                          "Block a shared-device motif corroborated by an independent semantic topic.", risk_ids, metrics),
            PolicyOutcome("SEM-P10", Decision.REVIEW, review,
                          "Review when two independent semantic topics remain unqualified, or one high-concern topic is strong on its own.", risk_ids, metrics),
            PolicyOutcome("SEM-P20", Decision.ALLOW, allow,
                          "Allow when the semantic reading is determined and no unqualified corroboration remains.",
                          tuple(item.id for item in evidence), metrics),
            PolicyOutcome("SEM-P00", Decision.ABSTAIN, abstain,
                          "Abstain when the semantic state is undetermined: no baseline exists for this card, or the amount has no reference frame.", (), metrics),
        )


def select_decision(policies: tuple[PolicyOutcome, ...]) -> DecisionRecord:
    triggered = tuple(item for item in policies if item.triggered)
    selected = next(
        (decision for decision in (Decision.BLOCK, Decision.REVIEW, Decision.ALLOW, Decision.ABSTAIN)
         if any(item.outcome == decision for item in triggered)), Decision.ABSTAIN)
    policy_ids = tuple(item.policy_id for item in triggered if item.outcome == selected)
    rationale = next((item.explanation for item in triggered if item.outcome == selected), "No policy was triggered.")
    return DecisionRecord(selected, rationale, policy_ids)


@dataclass(frozen=True)
class SemanticRuntimeResult:
    transaction: Any
    context: SemanticContextResult
    facts: tuple[SemanticFact, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    policies: tuple[PolicyOutcome, ...]
    decision: DecisionRecord

    def audit_record(self) -> dict[str, Any]:
        return {
            "semantic_runtime_version": SEMANTIC_RUNTIME_VERSION,
            "transaction": self.transaction.to_dict(),
            "semantic": {
                "ontology_version": ONTOLOGY_VERSION, "ontology_hash": ONTOLOGY_HASH,
                "context_version": CONTEXT_VERSION, "context_state_hash": self.context.context_state_hash,
                "objects": [item.to_dict() for item in self.context.objects],
                "withheld": [item.to_dict() for item in self.context.withheld],
                "coverage_gaps": list(self.context.coverage_gaps),
            },
            "facts": [item.to_dict() for item in self.facts],
            "evidence": [item.to_dict() for item in self.evidence],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "policies": [item.to_dict() for item in self.policies],
            "decision": self.decision.to_dict(),
        }


class SemanticDecisionRuntime:
    def __init__(self, context: SemanticContextLayer) -> None:
        self.context = context
        self.facts = SemanticFactExtractor()
        self.rules = SemanticRuleEngine()
        self.conflicts = ConflictEngine(SEMANTIC_CONFLICT_PAIRS)
        self.policies = SemanticPolicyEngine()

    def evaluate(self, transaction, event_index: int) -> SemanticRuntimeResult:
        reading = self.context.observe(transaction, event_index)
        facts = self.facts.extract(transaction, reading)
        evidence = self.rules.evaluate(transaction, facts)
        conflicts = self.conflicts.detect(evidence)
        policies = self.policies.evaluate(evidence, conflicts, reading.types())
        return SemanticRuntimeResult(transaction, reading, facts, evidence, conflicts, policies, select_decision(policies))


def semantic_replay_pins(result: SemanticRuntimeResult) -> dict[str, str]:
    rules_payload = "|".join(f"{rule.id}:{rule.consumes.value}:{rule.direction}:{rule.topic}:{rule.source_reliability}" for rule in SEMANTIC_RULES)
    objects_payload = "|".join(item.id for item in result.context.objects)
    return {
        "ontology_hash": ONTOLOGY_HASH,
        "inference_rules_hash": hashlib.sha256(CONTEXT_VERSION.encode("utf-8")).hexdigest(),
        "semantic_rules_hash": hashlib.sha256(rules_payload.encode("utf-8")).hexdigest(),
        "policy_hash": hashlib.sha256(SemanticPolicyEngine.VERSION.encode("utf-8")).hexdigest(),
        "context_state_hash": result.context.context_state_hash,
        "semantic_object_set_hash": hashlib.sha256(objects_payload.encode("utf-8")).hexdigest(),
        "input_snapshot_hash": hashlib.sha256(repr(result.transaction.to_dict()).encode("utf-8")).hexdigest(),
    }
