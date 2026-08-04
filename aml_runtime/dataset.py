"""Parser and loader for the CSV layouts emitted by IBM AMLSim."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Account, Country, Transaction


def _normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _value(row: dict[str, str], *names: str, default: str = "") -> str:
    normalised = {_normalise(key): (value or "").strip() for key, value in row.items()}
    for name in names:
        if _normalise(name) in normalised:
            return normalised[_normalise(name)]
    return default


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "sar", "laundering"}


def _number(value: str) -> float:
    try:
        return float(value.replace(",", "").replace("$", ""))
    except (AttributeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class AMLSimDataset:
    transactions: tuple[Transaction, ...]
    accounts: dict[str, Account]
    countries: dict[str, Country]
    source_path: str
    fingerprint: str


class AMLSimLoader:
    """Loads common AMLSim transaction, account, and alert CSV variants.

    IBM AMLSim's generated transaction log normally uses `From Bank`, `Account`,
    `To Bank`, and `Account.1`; compact exports using explicit account column
    names are accepted too.
    """

    TRANSACTION_HINTS = ("transaction", "tx_log", "transactions")
    ACCOUNT_HINTS = ("account", "accounts")

    def load(self, path: str | Path) -> AMLSimDataset:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"AMLSim dataset path does not exist: {source}")
        csv_paths = [source] if source.is_file() else sorted(source.rglob("*.csv"))
        if not csv_paths:
            raise ValueError(f"No CSV files found under {source}")
        transaction_path = self._select(csv_paths, self.TRANSACTION_HINTS, required=True)
        account_path = self._select(csv_paths, self.ACCOUNT_HINTS, required=False, exclude=transaction_path)
        transactions = tuple(sorted(self._load_transactions(transaction_path), key=lambda item: (item.timestamp, item.id)))
        if not transactions:
            raise ValueError(f"No transactions found in {transaction_path}")
        accounts = self._load_accounts(account_path) if account_path else {}
        for transaction in transactions:
            accounts.setdefault(transaction.originator_account_id, Account(id=transaction.originator_account_id, country_id=transaction.country_id))
            accounts.setdefault(transaction.beneficiary_account_id, Account(id=transaction.beneficiary_account_id, country_id=transaction.country_id))
            if transaction.is_sar:
                account = accounts[transaction.originator_account_id]
                accounts[account.id] = Account(**{**account.to_dict(), "is_sar": True})
        country_codes = sorted({code for code in (tx.country_id for tx in transactions) if code})
        high_risk = {"AF", "BY", "IR", "KP", "MM", "RU", "SY", "VE", "YE"}
        countries = {code: Country(id=f"country:{code}", code=code, high_risk=code in high_risk) for code in country_codes}
        fingerprint = self._fingerprint(transactions, accounts)
        return AMLSimDataset(transactions, dict(sorted(accounts.items())), countries, str(source), fingerprint)

    def _select(self, paths: list[Path], hints: tuple[str, ...], required: bool, exclude: Path | None = None) -> Path | None:
        choices = [path for path in paths if path != exclude and any(hint in path.name.lower() for hint in hints)]
        if choices:
            return sorted(choices)[0]
        if required:
            # A single arbitrary CSV is a practical supported transaction input.
            remaining = [path for path in paths if path != exclude]
            if len(remaining) == 1:
                return remaining[0]
            raise ValueError("Could not identify an AMLSim transaction CSV; use a name containing 'transaction' or 'tx_log'.")
        return None

    def _rows(self, path: Path) -> Iterable[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)

    def _load_transactions(self, path: Path) -> list[Transaction]:
        result: list[Transaction] = []
        for index, row in enumerate(self._rows(path), start=1):
            transaction_id = _value(row, "transaction_id", "tx_id", "txid", "id", default=f"TX-{index:06d}")
            origin = _value(row, "from_account_id", "from_account", "originator_account_id", "source_account", "account")
            destination = _value(row, "to_account_id", "to_account", "beneficiary_account_id", "destination_account", "account.1")
            if not origin or not destination:
                raise ValueError(f"Transaction {transaction_id} must provide originator and beneficiary account IDs")
            country = _value(row, "to_country", "beneficiary_country", "destination_country", "country", "country_code").upper()
            result.append(Transaction(
                id=transaction_id,
                timestamp=_value(row, "timestamp", "datetime", "date", "tx_datetime", default="1970-01-01T00:00:00"),
                originator_account_id=origin,
                beneficiary_account_id=destination,
                amount=_number(_value(row, "amount", "amount_received", "amount_paid")),
                currency=_value(row, "currency", "receiving_currency", "payment_currency", default="USD").upper(),
                country_id=country,
                payment_type=_value(row, "payment_type", "payment format", "type"),
                alert_id=_value(row, "alert_id", "alertid"),
                is_sar=_truth(_value(row, "is_sar", "is_laundering", "laundering", "sar")),
                metadata={key: value for key, value in sorted(row.items()) if value},
            ))
        return result

    def _load_accounts(self, path: Path) -> dict[str, Account]:
        accounts: dict[str, Account] = {}
        for row in self._rows(path):
            account_id = _value(row, "account_id", "account", "id")
            if not account_id:
                continue
            accounts[account_id] = Account(
                id=account_id,
                customer_id=_value(row, "customer_id", "customer", "party_id"),
                bank_id=_value(row, "bank_id", "bank"),
                country_id=_value(row, "country", "country_code").upper(),
                kyc_updated_at=_value(row, "kyc_updated_at", "kyc_date", "last_kyc"),
                is_sar=_truth(_value(row, "is_sar", "sar", "known_suspicious")),
                verified_source_of_funds=_truth(_value(row, "verified_source_of_funds", "source_of_funds_verified")),
                recently_manually_verified=_truth(_value(row, "recently_manually_verified", "manual_kyc_verified")),
                expected_payroll_day=_truth(_value(row, "expected_payroll_day", "payroll_account")),
            )
        return accounts

    def _fingerprint(self, transactions: tuple[Transaction, ...], accounts: dict[str, Account]) -> str:
        payload = "\n".join(repr(item.to_dict()) for item in transactions) + "\n" + "\n".join(repr(item.to_dict()) for _, item in sorted(accounts.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
