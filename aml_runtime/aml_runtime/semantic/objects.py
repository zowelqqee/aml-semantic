"""Immutable semantic objects: the things the Runtime is allowed to reason about.

A semantic object is an assertion about meaning.  It is auditable by
construction: it names the inference rule that produced it, the causal evidence
it read, the entities and relationships it depends on, and the arithmetic behind
its confidence.  Objects are never mutated; a revision is a new object whose
``supersedes`` points at the one it replaces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..models import Serializable
from .ontology import ONTOLOGY_VERSION, ObjectClass, SPECIFICATIONS, SemanticType


def object_id(type_: SemanticType, subject_id: str, *causal_parts: str) -> str:
    payload = "|".join((ONTOLOGY_VERSION, type_.value, subject_id, *causal_parts)).encode("utf-8")
    return f"SO-{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class SemanticEntity(Serializable):
    """A resolved party or place.  Entities are looked up, never inferred."""

    id: str
    kind: str
    label: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticObject(Serializable):
    """One assertion, with everything needed to audit and replay it."""

    id: str
    type: SemanticType
    object_class: ObjectClass
    subject_id: str
    confidence: float
    confidence_explanation: str
    supporting_facts: tuple[str, ...]
    supporting_entities: tuple[str, ...]
    supporting_relationships: tuple[str, ...]
    causal_evidence: dict[str, Any]
    origin: str
    version: str
    created_at: str
    supersedes: str = ""

    @property
    def meaning(self) -> str:
        return SPECIFICATIONS[self.type].meaning

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.type.value, "object_class": self.object_class.value,
            "subject_id": self.subject_id, "confidence": self.confidence,
            "confidence_explanation": self.confidence_explanation,
            "supporting_facts": list(self.supporting_facts),
            "supporting_entities": list(self.supporting_entities),
            "supporting_relationships": list(self.supporting_relationships),
            "causal_evidence": dict(sorted(self.causal_evidence.items())),
            "origin": self.origin, "version": self.version, "created_at": self.created_at,
            "supersedes": self.supersedes, "meaning": self.meaning,
        }


@dataclass(frozen=True)
class WithheldObject(Serializable):
    """A claim the layer declined to make, and the inputs it lacked.

    Recording this is what separates honest ignorance from a negative finding.
    """

    type: str
    missing_inputs: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SemanticContextResult(Serializable):
    """The complete semantic reading of one transaction."""

    transaction_id: str
    objects: tuple[SemanticObject, ...]
    withheld: tuple[WithheldObject, ...]
    coverage_gaps: tuple[str, ...]
    entities: tuple[SemanticEntity, ...]
    context_state_hash: str

    def types(self) -> frozenset[SemanticType]:
        return frozenset(item.type for item in self.objects)

    def by_type(self, type_: SemanticType) -> SemanticObject | None:
        return next((item for item in self.objects if item.type is type_), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "objects": [item.to_dict() for item in self.objects],
            "withheld": [item.to_dict() for item in self.withheld],
            "coverage_gaps": list(self.coverage_gaps),
            "entities": [item.to_dict() for item in self.entities],
            "context_state_hash": self.context_state_hash,
        }
