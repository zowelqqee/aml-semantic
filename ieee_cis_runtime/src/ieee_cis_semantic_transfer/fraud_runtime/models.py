"""The one fraud-domain record type. Everything else is imported unchanged.

``Evidence``, ``Conflict``, ``PolicyOutcome``, ``DecisionRecord`` and
``Decision`` carry zero AML-specific fields. They are re-exported from the
local generic core rather than redefined, which is the point: the decision
machinery does not know or care which domain it is deciding about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import Conflict, Decision, DecisionRecord, Evidence, PolicyOutcome, Serializable, as_primitive

__all__ = ["Conflict", "Decision", "DecisionRecord", "Evidence", "PolicyOutcome", "Serializable", "as_primitive", "Transaction"]


@dataclass(frozen=True)
class Transaction(Serializable):
    """One IEEE-CIS transaction row, resolved to fraud-domain fields.

    A card purchase is one-directional — a card spends, it never receives —
    so this record has no beneficiary/originator pair the way an AML payment
    does. The counterpart entity a purchase relates to is the *device* it was
    made from, not another account.

    ``timestamp`` is deliberately not a calendar date: ``TransactionDT`` is
    seconds from an undisclosed reference point (Vesta's own documentation
    does not specify it), so rendering it as an ISO datetime would imply a
    precision that does not exist. It is kept as an explicit relative marker.
    """

    id: str
    transaction_dt: int
    timestamp: str
    card_id: str
    amount: float
    product_channel: str
    billing_region: str
    billing_country: str
    device_id: str
    has_identity: bool
    purchaser_email_domain: str
    recipient_email_domain: str
    distance1: float | None
    distance2: float | None
    is_fraud: bool
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return as_primitive({
            "id": self.id, "transaction_dt": self.transaction_dt, "timestamp": self.timestamp,
            "card_id": self.card_id, "amount": self.amount, "product_channel": self.product_channel,
            "billing_region": self.billing_region, "billing_country": self.billing_country,
            "device_id": self.device_id, "has_identity": self.has_identity,
            "purchaser_email_domain": self.purchaser_email_domain,
            "recipient_email_domain": self.recipient_email_domain,
            "distance1": self.distance1, "distance2": self.distance2,
            "is_fraud": self.is_fraud, "metadata": self.metadata,
        })
