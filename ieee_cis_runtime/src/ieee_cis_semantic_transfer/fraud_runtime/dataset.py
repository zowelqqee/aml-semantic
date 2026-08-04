"""Loader for the IEEE-CIS Fraud Detection release.

``train_transaction.csv`` (590,540 rows) is already chronologically ordered by
``TransactionDT`` — verified before this module was written, not assumed.
``train_identity.csv`` (144,233 rows, a 24.4% join rate) is loaded once into
memory and joined by ``TransactionID``.

``test_transaction.csv`` is deliberately never read here: its labels do not
exist (this is a withheld Kaggle leaderboard file), and its ``TransactionDT``
range (18,403,224–34,214,345) sits entirely after ``train``'s
(86,400–15,811,131) — it is later, unlabelled data, useful for neither
training nor evaluation under this protocol.

Only the C1-C14, D1-D15, M1-M9 and V1-V339 transaction columns, and the
numeric ``id_01``-``id_11`` identity columns, are skipped by this loader. Their
meaning is undisclosed by Vesta (masked "engineered" signals, not raw
observations), so they are out of scope for both the raw and the semantic
feature space — see ``docs/fraud/dataset-selection.md`` for the reasoning.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .models import Transaction

TRANSACTION_COLUMNS = (
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
)
IDENTITY_COLUMNS = (
    "TransactionID", "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_23", "id_28", "id_29", "id_30", "id_31",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
)


def _float_or_none(value: str) -> float | None:
    return float(value) if value else None


class IEEECISLoader:
    """Streams the labelled train partition in chronological order."""

    def __init__(self, transaction_path: str | Path, identity_path: str | Path) -> None:
        self.transaction_path = Path(transaction_path)
        self.identity_path = Path(identity_path)

    def load_identity(self) -> dict[str, dict[str, str]]:
        identity: dict[str, dict[str, str]] = {}
        with self.identity_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                identity[row["TransactionID"]] = {key: row.get(key, "") for key in IDENTITY_COLUMNS}
        return identity

    def rows(self, identity: dict[str, dict[str, str]], limit: int | None = None) -> Iterator[tuple[int, Transaction]]:
        from .semantic.entities import card_key, device_key

        with self.transaction_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if limit is not None and index >= limit:
                    return
                transaction_id = row["TransactionID"]
                identity_row = identity.get(transaction_id)
                has_identity = identity_row is not None
                device_id = device_key(identity_row) if identity_row else ""
                transaction_dt = int(row["TransactionDT"])
                card_values = {column: row.get(column, "") for column in ("card1", "card2", "card3", "card4", "card5", "card6")}
                yield index, Transaction(
                    id=f"TXN-{transaction_id}",
                    transaction_dt=transaction_dt,
                    timestamp=f"REL+{transaction_dt:08d}s",
                    card_id=card_key(card_values),
                    amount=float(row["TransactionAmt"]),
                    product_channel=row["ProductCD"],
                    billing_region=row.get("addr1", ""),
                    billing_country=row.get("addr2", ""),
                    device_id=device_id,
                    has_identity=has_identity,
                    purchaser_email_domain=row.get("P_emaildomain", ""),
                    recipient_email_domain=row.get("R_emaildomain", ""),
                    distance1=_float_or_none(row.get("dist1", "")),
                    distance2=_float_or_none(row.get("dist2", "")),
                    is_fraud=row["isFraud"] == "1",
                    metadata={"network": card_values["card4"], "card_type": card_values["card6"],
                              "device_type": (identity_row or {}).get("DeviceType", "")},
                )
