"""The Semantic Behaviour Runtime.

This module *composes* the Semantic Context Layer and its policy engine; it does
not modify either.  Concretely:

* the semantic runtime is used unchanged to produce semantic evidence;
* behaviour objects are **projected** onto the same ``Evidence`` type — this is a
  projection of the behaviour catalog, not an additional rule layer, so no new
  rule vocabulary is introduced;
* the frozen ``SemanticPolicyEngine`` selects the decision, with its
  corroboration and single-high-concern thresholds untouched.

Motif-class behaviours deliberately carry the existing ``network_motif`` topic
so that the frozen policy treats them as high-concern without that policy
changing.  Every other behaviour topic simply joins the existing count of
independent topics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..models import Conflict, Decision, DecisionRecord, Evidence, PolicyOutcome, Transaction
from ..runtime import ConflictEngine, ConflictSpecification, stable_id
from ..semantic.entities import ResolvedAccount
from ..semantic.objects import SemanticContextResult
from ..semantic.runtime import (
    SEMANTIC_CONFLICT_PAIRS,
    STRUCTURAL_EXPLANATIONS,
    SemanticDecisionRuntime,
    SemanticFact,
    SemanticPolicyEngine,
    SemanticRuntimeResult,
    select_decision,
)
from .layer import BEHAVIOUR_LAYER_VERSION, BehaviourLayer
from .objects import BehaviourFact, BehaviourObject, BehaviourReading
from .ontology import BEHAVIOUR_ONTOLOGY_HASH, BEHAVIOUR_ONTOLOGY_VERSION, BEHAVIOUR_SPECIFICATIONS, BehaviourType

BEHAVIOUR_RUNTIME_VERSION = "aml-behaviour-runtime/1.0"


def behaviour_rule_id(type_: BehaviourType) -> str:
    return f"BEH-{type_.value}"


#: Declared behavioural disagreements.  Two kinds appear here: behaviour against
#: behaviour, and behaviour against the *semantic* risk it explains.  The second
#: kind is the point of the layer — a signal that looks alarming in one event
#: often has an ordinary explanation once the account's behaviour is known.
BEHAVIOUR_CONFLICT_PAIRS: tuple[ConflictSpecification, ...] = (
    ConflictSpecification("*", behaviour_rule_id(BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR), "behaviour_structural",
                          "Risk evidence is qualified: the account's flow is internal treasury movement."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.DISTRIBUTION_BEHAVIOUR), behaviour_rule_id(BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR), "behaviour_regime",
                          "A distribution shape conflicts with a sustained payment-run shape."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.FAN_OUT_DISTRIBUTION), behaviour_rule_id(BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR), "behaviour_regime",
                          "A burst fan-out conflicts with homogeneous payment-run amounts."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.BURST_ACTIVITY_BEHAVIOUR), behaviour_rule_id(BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR), "behaviour_regime",
                          "A tempo burst conflicts with homogeneous payment-run amounts."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.RELATIONSHIP_GROWTH_BEHAVIOUR), behaviour_rule_id(BehaviourType.SUPPLIER_SETTLEMENT_BEHAVIOUR), "behaviour_relationship",
                          "Counterparty expansion conflicts with settlement into an established set."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.TRANSIT_BEHAVIOUR), behaviour_rule_id(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR), "behaviour_shape",
                          "A transit reading conflicts with an infrastructural settlement shape."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.PASS_THROUGH_BEHAVIOUR), behaviour_rule_id(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR), "behaviour_shape",
                          "Prompt forwarding conflicts with an infrastructural settlement shape."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.CIRCULAR_MONEY_MOVEMENT), behaviour_rule_id(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR), "behaviour_shape",
                          "Reciprocal flow is ordinary between infrastructural settlement points."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.MONEY_ACCUMULATION_BEHAVIOUR), behaviour_rule_id(BehaviourType.EXPECTED_BUSINESS_CYCLE), "behaviour_stability",
                          "Accumulation conflicts with an unbroken value and tempo regime."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.COLLECTION_BEHAVIOUR), behaviour_rule_id(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR), "behaviour_shape",
                          "A collection shape conflicts with an infrastructural settlement shape."),
    # behaviour explains semantic risk
    ConflictSpecification("SEM-R01-VALUE-REGIME-BREAK", behaviour_rule_id(BehaviourType.EXPECTED_BUSINESS_CYCLE), "behaviour_stability",
                          "A value-regime break conflicts with an otherwise unbroken business cycle."),
    ConflictSpecification("SEM-R02-TEMPO-REGIME-SHIFT", behaviour_rule_id(BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR), "behaviour_regime",
                          "A tempo shift conflicts with a homogeneous payment run."),
    ConflictSpecification("SEM-R03-INFORMATIVE-NOVELTY", behaviour_rule_id(BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR), "behaviour_regime",
                          "Counterparty novelty is expected of a payment operator."),
    ConflictSpecification("SEM-R03-INFORMATIVE-NOVELTY", behaviour_rule_id(BehaviourType.SUPPLIER_SETTLEMENT_BEHAVIOUR), "behaviour_regime",
                          "Counterparty novelty conflicts with settlement into an established set."),
    ConflictSpecification("SEM-R07-CASH-INSTRUMENT", behaviour_rule_id(BehaviourType.EXPECTED_BUSINESS_CYCLE), "behaviour_stability",
                          "Instrument opacity conflicts with an unbroken business cycle."),
    ConflictSpecification("SEM-R05-VIRTUAL-ASSET", behaviour_rule_id(BehaviourType.EXPECTED_BUSINESS_CYCLE), "behaviour_stability",
                          "Virtual-asset exposure conflicts with an unbroken business cycle."),
    ConflictSpecification("SEM-R08-TRANSIT-ROLE", behaviour_rule_id(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR), "behaviour_shape",
                          "A transit role conflicts with an infrastructural settlement shape."),
    ConflictSpecification("SEM-R09-CONCENTRATION-ROLE", behaviour_rule_id(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR), "behaviour_shape",
                          "A concentration role conflicts with an infrastructural settlement shape."),
)

COMBINED_CONFLICT_PAIRS = SEMANTIC_CONFLICT_PAIRS + BEHAVIOUR_CONFLICT_PAIRS


class BehaviourEvidenceProjection:
    """Projects behaviour objects onto the frozen ``Evidence`` type.

    This is deliberately mechanical: direction, topic and source reliability all
    come from the behaviour's own ontology entry, so the catalog is the single
    place a behaviour's meaning is declared.
    """

    VERSION = "aml-behaviour-projection/1.0"

    def project(self, transaction: Transaction, behaviours: tuple[BehaviourObject, ...]) -> tuple[tuple[BehaviourFact, ...], tuple[Evidence, ...]]:
        facts: list[BehaviourFact] = []
        evidence: list[Evidence] = []
        for item in behaviours:
            specification = BEHAVIOUR_SPECIFICATIONS[item.type]
            if specification.direction == "none":
                continue
            fact = BehaviourFact(
                id=stable_id("BF", transaction.id, item.type.value, item.id),
                type=item.type,
                subject_id=item.subject_id,
                behaviour_object_id=item.id,
                confidence=item.confidence,
                explanation=f"{item.causal_explanation} Observed: {'; '.join(item.supporting_observations)}."
                            + (f" Against: {'; '.join(item.counter_evidence)}." if item.counter_evidence else ""),
                timestamp=transaction.timestamp,
                provenance=f"behaviour-object/{item.origin}",
            )
            facts.append(fact)
            evidence.append(Evidence(
                id=stable_id("E", transaction.id, behaviour_rule_id(item.type), item.id),
                source=f"behaviour-layer/{item.type.value}",
                supporting_facts=(fact.id,),
                confidence=round(item.confidence, 6),
                explanation=f"{specification.meaning} ({item.causal_explanation})",
                timestamp=transaction.timestamp,
                rule_id=behaviour_rule_id(item.type),
                direction=specification.direction,
                topic=specification.topic,
                source_reliability=specification.prior,
                recency_days=0,
                metadata={"behaviour_object_id": item.id, "behaviour_type": item.type.value,
                          "horizon": specification.horizon.value,
                          "interval_span_minutes": str(item.interval.span_minutes)},
            ))
        return tuple(facts), tuple(sorted(evidence, key=lambda item: (item.rule_id, item.id)))


@dataclass(frozen=True)
class BehaviourRuntimeResult:
    transaction: Transaction
    semantic: SemanticRuntimeResult
    behaviour: BehaviourReading
    facts: tuple[SemanticFact | BehaviourFact, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    policies: tuple[PolicyOutcome, ...]
    decision: DecisionRecord
    ml_evidence: Evidence | None = None
    routed_to_ml: bool = False

    @property
    def behaviour_rationale(self) -> str:
        """The decision, said in behaviour and scenario objects."""
        behaviours = [item.type.value for item in self.behaviour.behaviours]
        scenarios = [item.type.value for item in self.behaviour.scenarios]
        return (
            f"{self.decision.decision.value}: {self.decision.rationale} "
            f"Role: {self.behaviour.role.role.value}. "
            f"Behaviour: {', '.join(behaviours) or 'none claimed'}. "
            f"Scenario: {', '.join(scenarios) or 'none matched'}."
        )

    def audit_record(self) -> dict[str, Any]:
        record = self.semantic.audit_record()
        record["behaviour_runtime_version"] = BEHAVIOUR_RUNTIME_VERSION
        record["behaviour"] = {
            "ontology_version": BEHAVIOUR_ONTOLOGY_VERSION,
            "ontology_hash": BEHAVIOUR_ONTOLOGY_HASH,
            "layer_version": BEHAVIOUR_LAYER_VERSION,
            **self.behaviour.to_dict(),
        }
        record["facts"] = [item.to_dict() for item in self.facts]
        record["evidence"] = [item.to_dict() for item in self.evidence]
        record["conflicts"] = [item.to_dict() for item in self.conflicts]
        record["policies"] = [item.to_dict() for item in self.policies]
        record["decision"] = self.decision.to_dict()
        record["ml_evidence"] = self.ml_evidence.to_dict() if self.ml_evidence else None
        record["routed_to_ml"] = self.routed_to_ml
        record["behaviour_rationale"] = self.behaviour_rationale
        return record


class BehaviourDecisionRuntime:
    """Semantic Context -> Behaviour Layer -> evidence -> frozen policy -> decision."""

    VERSION = BEHAVIOUR_RUNTIME_VERSION

    def __init__(self, semantic: SemanticDecisionRuntime, layer: BehaviourLayer | None = None) -> None:
        self.semantic = semantic
        self.layer = layer or BehaviourLayer()
        self.projection = BehaviourEvidenceProjection()
        # Composition, not modification: the same ConflictEngine class, given the
        # union of the semantic and behavioural declared pairs.
        self.conflicts = ConflictEngine(COMBINED_CONFLICT_PAIRS)
        self.policies = SemanticPolicyEngine()

    def evaluate(self, transaction: Transaction, originator: ResolvedAccount, minute: int, event_index: int) -> BehaviourRuntimeResult:
        semantic = self.semantic.evaluate(transaction, event_index)
        behaviour = self.layer.observe(transaction, semantic.context, originator, minute, event_index)
        behaviour_facts, behaviour_evidence = self.projection.project(transaction, behaviour.behaviours)
        facts = semantic.facts + behaviour_facts
        evidence = semantic.evidence + behaviour_evidence
        conflicts = self.conflicts.detect(evidence)
        policies = self.policies.evaluate(evidence, conflicts, semantic.context.types())
        return BehaviourRuntimeResult(transaction, semantic, behaviour, facts, evidence, conflicts, policies,
                                      select_decision(policies))

    def commit(self, transaction: Transaction, result: BehaviourRuntimeResult, minute: int) -> None:
        self.semantic.context.commit(transaction)
        self.layer.commit(transaction, result.semantic.context, result.behaviour, minute)

    @staticmethod
    def routes_to_ml(result: BehaviourRuntimeResult) -> bool:
        """Unchanged from v1, so the two runtimes remain comparable."""
        return (
            result.decision.decision is Decision.ABSTAIN
            and not (result.semantic.context.types() & STRUCTURAL_EXPLANATIONS)
        )

    def with_ml_evidence(self, result: BehaviourRuntimeResult, ml_evidence: Evidence, high_band: float) -> BehaviourRuntimeResult:
        combined = result.evidence + (ml_evidence,)
        conflicts = self.conflicts.detect(combined)
        policies = self.policies.evaluate(combined, conflicts, result.semantic.context.types())
        base = select_decision(policies)
        qualified = any(item.risk_evidence_id == ml_evidence.id for item in conflicts)
        probability = float(ml_evidence.metadata.get("probability", ml_evidence.confidence))
        metrics: dict[str, float | int | str] = {
            "ml_probability": probability, "ml_high_band": high_band,
            "ml_evidence_qualified_by_conflict": int(qualified),
            "role": result.behaviour.role.role.value,
        }
        if base.decision in (Decision.BLOCK, Decision.REVIEW):
            outcome = PolicyOutcome("BEH-ML-01", base.decision, True,
                                    "The behavioural reading already selected this decision; ML evidence adds prioritisation only.",
                                    (ml_evidence.id,), metrics)
        elif not qualified and probability >= high_band:
            outcome = PolicyOutcome("BEH-ML-02", Decision.REVIEW, True,
                                    "An undetermined behavioural state plus unqualified high-band ML evidence is escalated to REVIEW.",
                                    (ml_evidence.id,), metrics)
        else:
            outcome = PolicyOutcome("BEH-ML-03", base.decision, True,
                                    "ML evidence is below the declared band or qualified by conflicting behavioural evidence; the behavioural decision stands.",
                                    (ml_evidence.id,), metrics)
        return BehaviourRuntimeResult(
            result.transaction, result.semantic, result.behaviour, result.facts, combined,
            conflicts, policies + (outcome,),
            DecisionRecord(outcome.outcome, outcome.explanation, (outcome.policy_id,)),
            ml_evidence, True,
        )


def behaviour_replay_pins(entity_snapshot_hash: str, result: BehaviourRuntimeResult) -> dict[str, str]:
    from ..semantic.runtime import semantic_replay_pins

    pins = semantic_replay_pins(entity_snapshot_hash, result.semantic)
    behaviour_payload = "|".join(item.id for item in result.behaviour.behaviours)
    pins.update({
        "behaviour_ontology_hash": BEHAVIOUR_ONTOLOGY_HASH,
        "behaviour_layer_hash": hashlib.sha256(BEHAVIOUR_LAYER_VERSION.encode("utf-8")).hexdigest(),
        "behaviour_projection_hash": hashlib.sha256(BehaviourEvidenceProjection.VERSION.encode("utf-8")).hexdigest(),
        "behaviour_object_set_hash": hashlib.sha256(behaviour_payload.encode("utf-8")).hexdigest(),
        "role_state_hash": hashlib.sha256(
            f"{result.behaviour.role.role.value}|{result.behaviour.role.since_minute}|{result.behaviour.role.transition_count}".encode("utf-8")
        ).hexdigest(),
    })
    return pins
