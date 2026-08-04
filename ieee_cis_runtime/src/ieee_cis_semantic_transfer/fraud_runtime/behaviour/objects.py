"""Behaviour, role, scenario and lifecycle objects for the fraud runtime.

Identical shape to ``aml_runtime/behaviour/objects.py``. A behaviour object is
a hypothesis, not a label: it carries what supports it, what argues against
it, and the interval it covers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ...core import Serializable

from .ontology import BEHAVIOUR_ONTOLOGY_VERSION, BEHAVIOUR_SPECIFICATIONS, BehaviourType, Horizon, RoleType, ScenarioType, Stage


def behaviour_id(prefix: str, type_: str, subject_id: str, *parts: str) -> str:
    payload = "|".join((BEHAVIOUR_ONTOLOGY_VERSION, type_, subject_id, *parts)).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class TimeInterval(Serializable):
    start_minute: int
    end_minute: int
    horizon: Horizon

    @property
    def span_minutes(self) -> int:
        return max(0, self.end_minute - self.start_minute)

    def to_dict(self) -> dict[str, Any]:
        return {"start_minute": self.start_minute, "end_minute": self.end_minute,
                "span_minutes": self.span_minutes, "horizon": self.horizon.value}


@dataclass(frozen=True)
class BehaviourObject(Serializable):
    id: str
    type: BehaviourType
    subject_id: str
    confidence: float
    confidence_explanation: str
    interval: TimeInterval
    supporting_semantic_objects: tuple[str, ...]
    supporting_entities: tuple[str, ...]
    supporting_relationships: tuple[str, ...]
    supporting_observations: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    causal_explanation: str
    origin: str
    version: str
    created_at: str

    @property
    def direction(self) -> str:
        return BEHAVIOUR_SPECIFICATIONS[self.type].direction

    @property
    def topic(self) -> str:
        return BEHAVIOUR_SPECIFICATIONS[self.type].topic

    @property
    def meaning(self) -> str:
        return BEHAVIOUR_SPECIFICATIONS[self.type].meaning

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.type.value, "subject_id": self.subject_id,
            "confidence": self.confidence, "confidence_explanation": self.confidence_explanation,
            "interval": self.interval.to_dict(),
            "supporting_semantic_objects": list(self.supporting_semantic_objects),
            "supporting_entities": list(self.supporting_entities),
            "supporting_relationships": list(self.supporting_relationships),
            "supporting_observations": list(self.supporting_observations),
            "counter_evidence": list(self.counter_evidence), "causal_explanation": self.causal_explanation,
            "direction": self.direction, "topic": self.topic, "meaning": self.meaning,
            "origin": self.origin, "version": self.version, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class BehaviourFact(Serializable):
    id: str
    type: BehaviourType
    subject_id: str
    behaviour_object_id: str
    confidence: float
    explanation: str
    timestamp: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type.value, "subject_id": self.subject_id,
                "behaviour_object_id": self.behaviour_object_id, "confidence": self.confidence,
                "explanation": self.explanation, "timestamp": self.timestamp, "provenance": self.provenance}


@dataclass(frozen=True)
class RoleTransition(Serializable):
    id: str
    subject_id: str
    from_role: RoleType
    to_role: RoleType
    at_minute: int
    at_event: str
    caused_by: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "subject_id": self.subject_id, "from_role": self.from_role.value,
                "to_role": self.to_role.value, "at_minute": self.at_minute, "at_event": self.at_event,
                "caused_by": list(self.caused_by), "explanation": self.explanation}


@dataclass(frozen=True)
class RoleObject(Serializable):
    id: str
    subject_id: str
    role: RoleType
    confidence: float
    since_minute: int
    tenure_minutes: int
    transition_count: int
    supporting_behaviours: tuple[str, ...]
    previous_role: RoleType
    explanation: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "subject_id": self.subject_id, "role": self.role.value,
                "confidence": self.confidence, "since_minute": self.since_minute,
                "tenure_minutes": self.tenure_minutes, "transition_count": self.transition_count,
                "supporting_behaviours": list(self.supporting_behaviours),
                "previous_role": self.previous_role.value, "explanation": self.explanation, "version": self.version}


@dataclass(frozen=True)
class ScenarioObject(Serializable):
    id: str
    type: ScenarioType
    subject_id: str
    confidence: float
    matched_stages: tuple[Stage, ...]
    observed_stages: tuple[Stage, ...]
    interval: TimeInterval
    supporting_behaviours: tuple[str, ...]
    counter_evidence: tuple[str, ...]
    causal_explanation: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type.value, "subject_id": self.subject_id, "confidence": self.confidence,
                "matched_stages": [item.value for item in self.matched_stages],
                "observed_stages": [item.value for item in self.observed_stages],
                "interval": self.interval.to_dict(), "supporting_behaviours": list(self.supporting_behaviours),
                "counter_evidence": list(self.counter_evidence), "causal_explanation": self.causal_explanation,
                "version": self.version}


@dataclass(frozen=True)
class LifecycleObject(Serializable):
    """Where the card is in its own observed life, independent of risk.

    ``distinct_devices`` is this port's analogue of AML's
    ``distinct_counterparties`` — the natural "how many others has this
    entity dealt with" count for a card is the devices it has purchased from,
    not other cards (a card never transacts with another card).
    """

    id: str
    subject_id: str
    age_minutes: int
    idle_minutes: int
    observed_events: int
    distinct_devices: int
    buckets_active: int
    first_seen_minute: int
    horizons_filled: tuple[str, ...]
    horizons_unfillable: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "subject_id": self.subject_id, "age_minutes": self.age_minutes,
                "idle_minutes": self.idle_minutes, "observed_events": self.observed_events,
                "distinct_devices": self.distinct_devices, "buckets_active": self.buckets_active,
                "first_seen_minute": self.first_seen_minute, "horizons_filled": list(self.horizons_filled),
                "horizons_unfillable": list(self.horizons_unfillable)}


@dataclass(frozen=True)
class BehaviourReading(Serializable):
    transaction_id: str
    subject_id: str
    behaviours: tuple[BehaviourObject, ...]
    role: RoleObject
    transition: RoleTransition | None
    scenarios: tuple[ScenarioObject, ...]
    lifecycle: LifecycleObject
    stage: Stage
    withheld: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def types(self) -> frozenset[BehaviourType]:
        return frozenset(item.type for item in self.behaviours)

    def by_type(self, type_: BehaviourType) -> BehaviourObject | None:
        return next((item for item in self.behaviours if item.type is type_), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id, "subject_id": self.subject_id,
            "behaviours": [item.to_dict() for item in self.behaviours], "role": self.role.to_dict(),
            "transition": self.transition.to_dict() if self.transition else None,
            "scenarios": [item.to_dict() for item in self.scenarios], "lifecycle": self.lifecycle.to_dict(),
            "stage": self.stage.value, "withheld": [dict(item) for item in self.withheld],
        }
