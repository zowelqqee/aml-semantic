"""Immutable domain records used by the AML decision runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def as_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [as_primitive(item) for item in value]
    if isinstance(value, list):
        return [as_primitive(item) for item in value]
    if isinstance(value, dict):
        return {key: as_primitive(item) for key, item in sorted(value.items())}
    return value


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return as_primitive(asdict(self))


class FactType(str, Enum):
    LARGE_TRANSFER = "LargeTransfer"
    NEW_BENEFICIARY = "NewBeneficiary"
    HIGH_RISK_COUNTRY = "HighRiskCountry"
    VELOCITY_INCREASE = "VelocityIncrease"
    OLD_KYC = "OldKYC"
    PREVIOUS_SAR = "PreviousSAR"
    CONNECTED_TO_SAR = "ConnectedToSAR"
    REPEATED_DESTINATION = "RepeatedDestination"
    VERIFIED_SOURCE_OF_FUNDS = "VerifiedSourceOfFunds"
    RECENT_MANUAL_VERIFICATION = "RecentlyManuallyVerified"
    EXPECTED_PAYROLL_DAY = "ExpectedPayrollDay"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Country(Serializable):
    id: str
    code: str
    high_risk: bool = False


@dataclass(frozen=True)
class Customer(Serializable):
    id: str
    name: str = ""
    country_id: str = ""
    kyc_updated_at: str = ""


@dataclass(frozen=True)
class Account(Serializable):
    id: str
    customer_id: str = ""
    bank_id: str = ""
    country_id: str = ""
    kyc_updated_at: str = ""
    is_sar: bool = False
    verified_source_of_funds: bool = False
    recently_manually_verified: bool = False
    expected_payroll_day: bool = False


@dataclass(frozen=True)
class Transaction(Serializable):
    id: str
    timestamp: str
    originator_account_id: str
    beneficiary_account_id: str
    amount: float
    currency: str = "USD"
    country_id: str = ""
    payment_type: str = ""
    alert_id: str = ""
    is_sar: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship(Serializable):
    id: str
    from_entity_id: str
    to_entity_id: str
    kind: str


@dataclass(frozen=True)
class Alert(Serializable):
    id: str
    transaction_id: str
    alert_type: str = ""


@dataclass(frozen=True)
class HistoricalBehavior(Serializable):
    account_id: str
    transaction_count_30d: int
    average_amount: float
    destination_count_30d: int


@dataclass(frozen=True)
class SARFlag(Serializable):
    id: str
    entity_id: str
    source: str


@dataclass(frozen=True)
class Fact(Serializable):
    id: str
    type: FactType
    subject_id: str
    value: str
    confidence: float
    explanation: str
    timestamp: str
    provenance: str


@dataclass(frozen=True)
class Evidence(Serializable):
    id: str
    source: str
    supporting_facts: tuple[str, ...]
    confidence: float
    explanation: str
    timestamp: str
    rule_id: str
    direction: str = "risk"
    topic: str = ""
    source_reliability: float = 0.0
    recency_days: int | None = None
    # Provider-specific, serializable provenance.  Runtime rule evidence keeps
    # this empty; external providers use it to pin the exact artifact that
    # produced a piece of evidence without altering policy semantics.
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Conflict(Serializable):
    id: str
    risk_evidence_id: str
    mitigating_evidence_id: str
    kind: str
    explanation: str
    dimensions: tuple[str, ...] = ()
    risk_confidence: float = 0.0
    mitigating_confidence: float = 0.0
    risk_source_reliability: float = 0.0
    mitigating_source_reliability: float = 0.0


@dataclass(frozen=True)
class PolicyOutcome(Serializable):
    policy_id: str
    outcome: Decision
    triggered: bool
    explanation: str
    evidence_ids: tuple[str, ...]
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord(Serializable):
    decision: Decision
    rationale: str
    policy_ids: tuple[str, ...]
