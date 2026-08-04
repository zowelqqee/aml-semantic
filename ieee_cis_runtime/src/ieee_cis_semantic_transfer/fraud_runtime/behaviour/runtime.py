"""Composition of the fraud Semantic Context and Behaviour layers.

The evidence, conflict and policy machinery is reused unchanged from the AML
runtime.  This module only projects fraud-domain behaviour objects onto that
generic evidence contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ...core import ConflictEngine, ConflictSpecification, stable_id
from ..models import Conflict, Decision, DecisionRecord, Evidence, PolicyOutcome, Transaction
from ..semantic.runtime import SEMANTIC_CONFLICT_PAIRS, STRUCTURAL_EXPLANATIONS, SemanticDecisionRuntime, SemanticFact, SemanticPolicyEngine, SemanticRuntimeResult, select_decision, semantic_replay_pins
from .layer import BEHAVIOUR_LAYER_VERSION, BehaviourLayer
from .objects import BehaviourFact, BehaviourObject, BehaviourReading
from .ontology import BEHAVIOUR_ONTOLOGY_HASH, BEHAVIOUR_ONTOLOGY_VERSION, BEHAVIOUR_SPECIFICATIONS, BehaviourType

BEHAVIOUR_RUNTIME_VERSION = "fraud-behaviour-runtime/1.0"


def behaviour_rule_id(type_: BehaviourType) -> str:
    return f"BEH-{type_.value}"


BEHAVIOUR_CONFLICT_PAIRS: tuple[ConflictSpecification, ...] = (
    ConflictSpecification(behaviour_rule_id(BehaviourType.VELOCITY_BURST_BEHAVIOUR), behaviour_rule_id(BehaviourType.EXPECTED_VELOCITY_BEHAVIOUR), "behaviour_tempo", "A velocity spike conflicts with velocity established as normal for this card."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.DEVICE_ROTATION_BEHAVIOUR), behaviour_rule_id(BehaviourType.TRUSTED_DEVICE_BEHAVIOUR), "behaviour_device", "Rapid client-signature rotation conflicts with a sustained stable card-signature relationship."),
    ConflictSpecification(behaviour_rule_id(BehaviourType.UNEXPECTED_SPENDING_BEHAVIOUR), behaviour_rule_id(BehaviourType.NORMAL_SPENDING_BEHAVIOUR), "behaviour_stability", "A sustained regime break conflicts with an unbroken spending regime."),
    ConflictSpecification("SEM-R01-UNEXPECTED-AMOUNT", behaviour_rule_id(BehaviourType.NORMAL_SPENDING_BEHAVIOUR), "behaviour_stability", "An amount break is qualified by sustained normal spending history."),
    ConflictSpecification("SEM-R02-TEMPO-SHIFT", behaviour_rule_id(BehaviourType.EXPECTED_VELOCITY_BEHAVIOUR), "behaviour_tempo", "A tempo shift is qualified when it fits the card's established velocity."),
    ConflictSpecification("SEM-R03-FIRST-DEVICE", behaviour_rule_id(BehaviourType.TRUSTED_DEVICE_BEHAVIOUR), "behaviour_device", "A first device contact conflicts with the current card's trusted-device behaviour."),
)
COMBINED_CONFLICT_PAIRS = SEMANTIC_CONFLICT_PAIRS + BEHAVIOUR_CONFLICT_PAIRS


class BehaviourEvidenceProjection:
    VERSION = "fraud-behaviour-projection/1.0"

    def project(self, transaction: Transaction, behaviours: tuple[BehaviourObject, ...]) -> tuple[tuple[BehaviourFact, ...], tuple[Evidence, ...]]:
        facts: list[BehaviourFact] = []
        evidence: list[Evidence] = []
        for item in behaviours:
            specification = BEHAVIOUR_SPECIFICATIONS[item.type]
            if specification.direction == "none":
                continue
            fact = BehaviourFact(stable_id("BF", transaction.id, item.type.value, item.id), item.type, item.subject_id, item.id,
                                 item.confidence, f"{item.causal_explanation} Observed: {'; '.join(item.supporting_observations)}.",
                                 transaction.timestamp, f"behaviour-object/{item.origin}")
            facts.append(fact)
            evidence.append(Evidence(
                id=stable_id("E", transaction.id, behaviour_rule_id(item.type), item.id), source=f"behaviour-layer/{item.type.value}",
                supporting_facts=(fact.id,), confidence=round(item.confidence, 6),
                explanation=f"{specification.meaning} ({item.causal_explanation})", timestamp=transaction.timestamp,
                rule_id=behaviour_rule_id(item.type), direction=specification.direction, topic=specification.topic,
                source_reliability=specification.prior, recency_days=0,
                metadata={"behaviour_object_id": item.id, "behaviour_type": item.type.value, "horizon": specification.horizon.value,
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

    @property
    def behaviour_rationale(self) -> str:
        return (f"{self.decision.decision.value}: {self.decision.rationale} Role: {self.behaviour.role.role.value}. "
                f"Behaviour: {', '.join(item.type.value for item in self.behaviour.behaviours) or 'none claimed'}. "
                f"Scenario: {', '.join(item.type.value for item in self.behaviour.scenarios) or 'none matched'}.")

    def audit_record(self) -> dict[str, Any]:
        record = self.semantic.audit_record()
        record["behaviour_runtime_version"] = BEHAVIOUR_RUNTIME_VERSION
        record["behaviour"] = {"ontology_version": BEHAVIOUR_ONTOLOGY_VERSION, "ontology_hash": BEHAVIOUR_ONTOLOGY_HASH,
                               "layer_version": BEHAVIOUR_LAYER_VERSION, **self.behaviour.to_dict()}
        record["facts"] = [item.to_dict() for item in self.facts]
        record["evidence"] = [item.to_dict() for item in self.evidence]
        record["conflicts"] = [item.to_dict() for item in self.conflicts]
        record["policies"] = [item.to_dict() for item in self.policies]
        record["decision"] = self.decision.to_dict()
        record["behaviour_rationale"] = self.behaviour_rationale
        return record


class BehaviourDecisionRuntime:
    """Semantic Context -> Behaviour -> generic Evidence/Conflict/Policy."""

    VERSION = BEHAVIOUR_RUNTIME_VERSION

    def __init__(self, semantic: SemanticDecisionRuntime | None = None, layer: BehaviourLayer | None = None) -> None:
        from ..semantic.context import SemanticContextLayer
        self.semantic = semantic or SemanticDecisionRuntime(SemanticContextLayer())
        self.layer = layer or BehaviourLayer()
        self.projection = BehaviourEvidenceProjection()
        self.conflicts = ConflictEngine(COMBINED_CONFLICT_PAIRS)
        self.policies = SemanticPolicyEngine()

    def evaluate(self, transaction: Transaction, minute: int, event_index: int) -> BehaviourRuntimeResult:
        semantic = self.semantic.evaluate(transaction, event_index)
        behaviour = self.layer.observe(transaction, semantic.context, minute, event_index)
        behaviour_facts, behaviour_evidence = self.projection.project(transaction, behaviour.behaviours)
        facts = semantic.facts + behaviour_facts
        evidence = semantic.evidence + behaviour_evidence
        conflicts = self.conflicts.detect(evidence)
        policies = self.policies.evaluate(evidence, conflicts, semantic.context.types())
        return BehaviourRuntimeResult(transaction, semantic, behaviour, facts, evidence, conflicts, policies, select_decision(policies))

    def commit(self, transaction: Transaction, result: BehaviourRuntimeResult, minute: int) -> None:
        self.semantic.context.commit(transaction)
        self.layer.commit(transaction, result.semantic.context, result.behaviour, minute)

    @staticmethod
    def routes_to_ml(result: BehaviourRuntimeResult) -> bool:
        return result.decision.decision is Decision.ABSTAIN and not (result.semantic.context.types() & STRUCTURAL_EXPLANATIONS)


def behaviour_replay_pins(result: BehaviourRuntimeResult) -> dict[str, str]:
    pins = semantic_replay_pins(result.semantic)
    objects = "|".join(item.id for item in result.behaviour.behaviours)
    pins.update({
        "behaviour_ontology_hash": BEHAVIOUR_ONTOLOGY_HASH,
        "behaviour_layer_hash": hashlib.sha256(BEHAVIOUR_LAYER_VERSION.encode("utf-8")).hexdigest(),
        "behaviour_projection_hash": hashlib.sha256(BehaviourEvidenceProjection.VERSION.encode("utf-8")).hexdigest(),
        "behaviour_object_set_hash": hashlib.sha256(objects.encode("utf-8")).hexdigest(),
        "role_state_hash": hashlib.sha256(f"{result.behaviour.role.role.value}|{result.behaviour.role.since_minute}|{result.behaviour.role.transition_count}".encode("utf-8")).hexdigest(),
    })
    return pins
