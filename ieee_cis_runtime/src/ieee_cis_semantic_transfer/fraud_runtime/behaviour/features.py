"""Named semantic feature vector for the fraud transfer experiment.

The vector contains only emitted semantic/behaviour/lifecycle/evidence objects.
Raw transaction columns remain exclusively in the Raw and Raw + Semantic arms.
"""

from __future__ import annotations

import numpy as np

from ..models import Evidence
from ..semantic.ontology import SemanticType
from .objects import BehaviourReading
from .ontology import BehaviourType, RoleType, ScenarioType

SEMANTIC_OBJECT_FEATURES = tuple(f"sem_{item.value}" for item in SemanticType)
BEHAVIOUR_FEATURES = tuple(f"beh_{item.value}" for item in BehaviourType)
SCENARIO_FEATURES = tuple(f"scn_{item.value}" for item in ScenarioType)
ROLE_FEATURES = ("role_code", "role_confidence", "role_tenure_minutes", "role_transition_count")
LIFECYCLE_FEATURES = ("lifecycle_age_minutes", "lifecycle_idle_minutes", "lifecycle_observed_events", "lifecycle_distinct_devices", "lifecycle_buckets_active", "lifecycle_horizons_filled")
EVIDENCE_FEATURES = ("evidence_risk_count", "evidence_mitigation_count", "evidence_conflict_count", "evidence_unqualified_topics", "evidence_effective_risk", "evidence_strongest_risk")
SEMANTIC_FEATURE_NAMES = SEMANTIC_OBJECT_FEATURES + BEHAVIOUR_FEATURES + SCENARIO_FEATURES + ROLE_FEATURES + LIFECYCLE_FEATURES + EVIDENCE_FEATURES
FEATURE_COUNT = len(SEMANTIC_FEATURE_NAMES)

_SEMANTIC_INDEX = {item: index for index, item in enumerate(SemanticType)}
_BEHAVIOUR_INDEX = {item: index for index, item in enumerate(BehaviourType)}
_SCENARIO_INDEX = {item: index for index, item in enumerate(ScenarioType)}
_ROLE_CODE = {item: index for index, item in enumerate(RoleType)}
_BEHAVIOUR_BASE = len(SEMANTIC_OBJECT_FEATURES)
_SCENARIO_BASE = _BEHAVIOUR_BASE + len(BEHAVIOUR_FEATURES)
_ROLE_BASE = _SCENARIO_BASE + len(SCENARIO_FEATURES)
_LIFECYCLE_BASE = _ROLE_BASE + len(ROLE_FEATURES)
_EVIDENCE_BASE = _LIFECYCLE_BASE + len(LIFECYCLE_FEATURES)


def semantic_feature_vector(semantic_types_confidence: dict[SemanticType, float], behaviour: BehaviourReading,
                            evidence: tuple[Evidence, ...], conflicts_count: int, unqualified_topics: int,
                            effective_risk: float, out: np.ndarray | None = None) -> np.ndarray:
    vector = np.zeros(FEATURE_COUNT, dtype=np.float32) if out is None else out
    vector.fill(0.0)
    for type_, confidence in semantic_types_confidence.items():
        vector[_SEMANTIC_INDEX[type_]] = confidence
    for item in behaviour.behaviours:
        vector[_BEHAVIOUR_BASE + _BEHAVIOUR_INDEX[item.type]] = item.confidence
    for item in behaviour.scenarios:
        vector[_SCENARIO_BASE + _SCENARIO_INDEX[item.type]] = item.confidence
    role = behaviour.role
    vector[_ROLE_BASE:_ROLE_BASE + 4] = (_ROLE_CODE[role.role], role.confidence, role.tenure_minutes, role.transition_count)
    lifecycle = behaviour.lifecycle
    vector[_LIFECYCLE_BASE:_LIFECYCLE_BASE + 6] = (lifecycle.age_minutes, lifecycle.idle_minutes, lifecycle.observed_events,
                                                    lifecycle.distinct_devices, lifecycle.buckets_active, len(lifecycle.horizons_filled))
    risk = [item for item in evidence if item.direction == "risk"]
    vector[_EVIDENCE_BASE:_EVIDENCE_BASE + 6] = (len(risk), len(evidence) - len(risk), conflicts_count, unqualified_topics,
                                                  effective_risk, max((item.confidence for item in risk), default=0.0))
    return vector
