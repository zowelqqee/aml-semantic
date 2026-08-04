"""Entity resolution: accounts are booking locations, customers are parties.

The v0.2 Runtime had no customer concept, so it could not tell a payment to a
stranger from a movement between two accounts of the same legal party.  This
module supplies that distinction from the account reference data shipped with
IBM AML `HI-Small`, plus the booking jurisdiction encoded in each bank name.

Nothing here is inferred from behaviour and nothing is derived from the
laundering label; this is reference data, resolved once and held immutable.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .objects import SemanticEntity
from .ontology import (
    DOMESTIC_JURISDICTION,
    ENHANCED_SCRUTINY_JURISDICTIONS,
    ENTITY_FORM_BY_SOURCE_LABEL,
    EntityForm,
    JURISDICTION_TOKENS,
    VIRTUAL_ASSET_BANK_TOKEN,
    VIRTUAL_ASSET_JURISDICTION,
)

_BANK_STEM = re.compile(r"\s*#\d+\s*$")
_BANK_PREFIX = re.compile(r"^(.+) Bank$")


def account_key(bank: str, number: str) -> str:
    """The join key shared with the ML feature pipeline: ``<int bank>:<ACCOUNT>``."""
    return f"{int(bank)}:{number.strip().upper()}"


def jurisdiction_of(bank_name: str) -> str:
    """Booking jurisdiction encoded in the bank's name.

    ``"Germany Bank #123"`` books in Germany; ``"Savings Bank of Topeka"`` is a
    domestic institution; ``"Crytpo Bank #7"`` (the source's spelling) is a
    virtual-asset venue and deliberately not given a country.
    """
    stem = _BANK_STEM.sub("", bank_name).strip()
    match = _BANK_PREFIX.match(stem)
    if not match:
        return DOMESTIC_JURISDICTION
    prefix = match.group(1).strip()
    if prefix == VIRTUAL_ASSET_BANK_TOKEN:
        return VIRTUAL_ASSET_JURISDICTION
    return prefix if prefix in JURISDICTION_TOKENS else DOMESTIC_JURISDICTION


@dataclass(frozen=True)
class ResolvedAccount:
    """An account's stable, non-behavioural identity."""

    account_id: str
    customer_id: str
    form: EntityForm
    bank_id: str
    jurisdiction: str

    @property
    def virtual_asset_venue(self) -> bool:
        return self.jurisdiction == VIRTUAL_ASSET_JURISDICTION

    @property
    def enhanced_scrutiny(self) -> bool:
        return self.jurisdiction in ENHANCED_SCRUTINY_JURISDICTIONS

    def entities(self) -> tuple[SemanticEntity, ...]:
        return (
            SemanticEntity(f"cust:{self.customer_id}", self.form.value, self.customer_id, {}),
            SemanticEntity(f"acct:{self.account_id}", "Account", self.account_id, {"bank": self.bank_id}),
            SemanticEntity(f"juris:{self.jurisdiction}", "Jurisdiction", self.jurisdiction, {}),
        )


UNRESOLVED = ResolvedAccount("", "", EntityForm.UNRESOLVED_FORM, "", DOMESTIC_JURISDICTION)


class EntityResolver:
    """Loads account reference data and answers identity questions about it."""

    def __init__(self) -> None:
        self._accounts: dict[str, ResolvedAccount] = {}
        self.source_path = ""
        self.snapshot_hash = ""

    @property
    def resolved_count(self) -> int:
        return len(self._accounts)

    def load(self, path: str | Path, restrict_to: set[str] | None = None) -> "EntityResolver":
        """Resolve accounts from the IBM `*_accounts.csv` reference file.

        ``restrict_to`` keeps only the accounts a run will actually touch; the
        full file carries 518,581 rows and the working set is a fraction of it.
        """
        source = Path(path)
        digest = hashlib.sha256()
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = account_key(row["Bank ID"], row["Account Number"])
                if restrict_to is not None and key not in restrict_to:
                    continue
                label = _BANK_STEM.sub("", row["Entity Name"]).strip()
                form = ENTITY_FORM_BY_SOURCE_LABEL.get(label, EntityForm.UNRESOLVED_FORM)
                resolved = ResolvedAccount(
                    account_id=key,
                    customer_id=row["Entity ID"].strip(),
                    form=form,
                    bank_id=str(int(row["Bank ID"])),
                    jurisdiction=jurisdiction_of(row["Bank Name"]),
                )
                self._accounts[key] = resolved
                digest.update(f"{key}|{resolved.customer_id}|{form.value}|{resolved.jurisdiction}\n".encode("utf-8"))
        self.source_path = str(source)
        self.snapshot_hash = digest.hexdigest()
        return self

    def resolve(self, account_id: str) -> ResolvedAccount:
        found = self._accounts.get(account_id)
        if found is not None:
            return found
        # An unresolvable account is a coverage gap, not a default-clean party.
        return ResolvedAccount(account_id, "", EntityForm.UNRESOLVED_FORM, "", DOMESTIC_JURISDICTION)

    def resolved(self, account_id: str) -> bool:
        return account_id in self._accounts
