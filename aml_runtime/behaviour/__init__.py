"""Semantic Behaviour Layer for the AML decision runtime.

Design: ``behaviour_layer_architecture.md`` and the three catalogs at the
repository root.  This package composes the Semantic Context Layer; it does not
modify it.
"""

from .features import FEATURE_COUNT, SEMANTIC_FEATURE_NAMES, semantic_feature_vector
from .layer import BEHAVIOUR_LAYER_VERSION, BehaviourLayer, SemanticFlags
from .objects import (
    BehaviourFact,
    BehaviourObject,
    BehaviourReading,
    LifecycleObject,
    RoleObject,
    RoleTransition,
    ScenarioObject,
    TimeInterval,
)
from .ontology import (
    BEHAVIOUR_ONTOLOGY_HASH,
    BEHAVIOUR_ONTOLOGY_VERSION,
    BEHAVIOUR_SPECIFICATIONS,
    BehaviourType,
    Horizon,
    RoleType,
    ScenarioType,
    Stage,
    behaviour_confidence,
)
from .runtime import (
    BEHAVIOUR_CONFLICT_PAIRS,
    BEHAVIOUR_RUNTIME_VERSION,
    BehaviourDecisionRuntime,
    BehaviourEvidenceProjection,
    BehaviourRuntimeResult,
    behaviour_replay_pins,
    behaviour_rule_id,
)
from .temporal import AccountBehaviourState, TemporalEngine

__all__ = [
    "AccountBehaviourState", "BEHAVIOUR_CONFLICT_PAIRS", "BEHAVIOUR_LAYER_VERSION",
    "BEHAVIOUR_ONTOLOGY_HASH", "BEHAVIOUR_ONTOLOGY_VERSION", "BEHAVIOUR_RUNTIME_VERSION",
    "BEHAVIOUR_SPECIFICATIONS", "BehaviourDecisionRuntime", "BehaviourEvidenceProjection",
    "BehaviourFact", "BehaviourLayer", "BehaviourObject", "BehaviourReading",
    "BehaviourRuntimeResult", "BehaviourType", "FEATURE_COUNT", "Horizon", "LifecycleObject",
    "RoleObject", "RoleTransition", "RoleType", "SEMANTIC_FEATURE_NAMES", "ScenarioObject",
    "ScenarioType", "SemanticFlags", "Stage", "TemporalEngine", "TimeInterval",
    "behaviour_confidence", "behaviour_replay_pins", "behaviour_rule_id",
    "semantic_feature_vector",
]
