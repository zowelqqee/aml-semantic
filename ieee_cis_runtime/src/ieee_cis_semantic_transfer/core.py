"""Domain-independent evidence, conflict, decision, and audit primitives.

Copied unchanged in behaviour from the generic Runtime core used by the AML
experiment, so this package has no runtime dependency on ``aml_runtime``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def as_primitive(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, datetime): return value.isoformat()
    if isinstance(value, tuple): return [as_primitive(item) for item in value]
    if isinstance(value, list): return [as_primitive(item) for item in value]
    if isinstance(value, dict): return {key: as_primitive(item) for key, item in sorted(value.items())}
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]: return as_primitive(asdict(self))


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Evidence(Serializable):
    id: str; source: str; supporting_facts: tuple[str, ...]; confidence: float; explanation: str; timestamp: str; rule_id: str
    direction: str = "risk"; topic: str = ""; source_reliability: float = 0.0; recency_days: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Conflict(Serializable):
    id: str; risk_evidence_id: str; mitigating_evidence_id: str; kind: str; explanation: str
    dimensions: tuple[str, ...] = (); risk_confidence: float = 0.0; mitigating_confidence: float = 0.0
    risk_source_reliability: float = 0.0; mitigating_source_reliability: float = 0.0


@dataclass(frozen=True)
class PolicyOutcome(Serializable):
    policy_id: str; outcome: Decision; triggered: bool; explanation: str; evidence_ids: tuple[str, ...]
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord(Serializable):
    decision: Decision; rationale: str; policy_ids: tuple[str, ...]


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:12]}"


@dataclass(frozen=True)
class ConflictSpecification:
    risk_rule_id: str; mitigating_rule_id: str; kind: str; explanation: str; temporal_supersession: bool = False


class ConflictEngine:
    """Generic explicit risk/control conflict detector."""
    def __init__(self, pairs: tuple[ConflictSpecification, ...] = ()) -> None: self.pairs = pairs

    def detect(self, evidence: tuple[Evidence, ...]) -> tuple[Conflict, ...]:
        by_rule = {item.rule_id: item for item in evidence}; conflicts: list[Conflict] = []
        for specification in self.pairs:
            mitigation = by_rule.get(specification.mitigating_rule_id)
            if mitigation is None: continue
            risks = [item for item in evidence if item.direction == "risk"] if specification.risk_rule_id == "*" else ([by_rule[specification.risk_rule_id]] if specification.risk_rule_id in by_rule else [])
            for risk in risks:
                dimensions = ["positive_negative"]
                if abs(risk.confidence - mitigation.confidence) >= .08: dimensions.append("confidence_asymmetry")
                if abs(risk.source_reliability - mitigation.source_reliability) >= .08: dimensions.append("source_strength_asymmetry")
                if specification.temporal_supersession or (risk.recency_days is not None and mitigation.recency_days in (None, 0)): dimensions.append("old_vs_recent")
                conflicts.append(Conflict(stable_id("C", risk.id, mitigation.id, specification.kind), risk.id, mitigation.id, specification.kind,
                                          specification.explanation, tuple(dimensions), risk.confidence, mitigation.confidence,
                                          risk.source_reliability, mitigation.source_reliability))
        return tuple(sorted(conflicts, key=lambda item: item.id))
