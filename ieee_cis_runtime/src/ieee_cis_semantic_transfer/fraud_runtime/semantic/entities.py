"""Entity resolution for the fraud runtime.

Unlike ``aml_runtime.semantic.entities`` — which joined a separate account
reference file — IEEE-CIS carries every entity-defining field directly on the
transaction row (or on its identity-table join). There is no external
reference file to load; resolution is a pure function of the row.

Card ≠ Customer here in the same sense Account ≠ Customer mattered for AML:
``card1`` alone collides across genuinely different cards (it is one of six
fields Vesta uses to fingerprint a card), so the resolvable entity is the
composite key, not any single column.
"""

from __future__ import annotations

from dataclasses import dataclass

from .objects import SemanticEntity

#: card1 is the primary fingerprint; card2/3/5 refine it; card4 is the card
#: network (visa/mastercard/...); card6 is the card type (debit/credit/...).
#: The composite key is what actually identifies "the same card" in this
#: pseudonymised release.
CARD_KEY_COLUMNS = ("card1", "card2", "card3", "card4", "card5", "card6")

#: DeviceType (mobile/desktop) plus DeviceInfo (a device-model or browser
#: string, 1,787 distinct values in train) is the finest device fingerprint
#: the public release carries.
DEVICE_KEY_COLUMNS = ("DeviceType", "DeviceInfo")


def card_key(values: dict[str, str]) -> str:
    return "|".join(values.get(column, "") for column in CARD_KEY_COLUMNS)


def device_key(values: dict[str, str]) -> str:
    device_type = values.get("DeviceType", "")
    device_info = values.get("DeviceInfo", "")
    if not device_type and not device_info:
        return ""
    return f"{device_type}|{device_info}"


@dataclass(frozen=True)
class ResolvedCard:
    """A card's stable, non-behavioural identity."""

    card_id: str
    network: str  # card4: visa / mastercard / american express / discover
    card_type: str  # card6: debit / credit / charge card / debit or credit
    billing_region: str  # addr1
    billing_country: str  # addr2

    def entities(self) -> tuple[SemanticEntity, ...]:
        return (
            SemanticEntity(f"card:{self.card_id}", "Card", self.card_id,
                            {"network": self.network, "type": self.card_type}),
            SemanticEntity(f"issuer:{self.network}:{self.card_type}", "Issuer", self.network,
                            {"card_type": self.card_type}),
            SemanticEntity(f"region:{self.billing_region}", "BillingRegion", self.billing_region, {}),
        )


@dataclass(frozen=True)
class ResolvedDevice:
    device_id: str
    device_type: str

    def entity(self) -> SemanticEntity:
        return SemanticEntity(f"device:{self.device_id}", "Device", self.device_id, {"device_type": self.device_type})


UNRESOLVED_CARD = ResolvedCard("", "", "", "", "")
