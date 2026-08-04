"""Deterministic transaction graph and historical-context builder."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

from .dataset import AMLSimDataset
from .models import HistoricalBehavior, Relationship, Transaction


def parse_timestamp(value: str) -> datetime:
    cleaned = value.strip().replace("Z", "+00:00")
    for candidate in (cleaned, cleaned.replace(" ", "T")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            pass
    for pattern in ("%Y/%m/%d %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            pass
    return datetime(1970, 1, 1)


class TransactionGraph:
    def __init__(self, dataset: AMLSimDataset) -> None:
        self.dataset = dataset
        self.transactions = tuple(sorted(dataset.transactions, key=lambda tx: (parse_timestamp(tx.timestamp), tx.id)))
        self.by_id = {transaction.id: transaction for transaction in self.transactions}
        self.outbound: dict[str, tuple[Transaction, ...]] = {}
        self.pair_history: dict[tuple[str, str], tuple[Transaction, ...]] = {}
        out: dict[str, list[Transaction]] = defaultdict(list)
        pairs: dict[tuple[str, str], list[Transaction]] = defaultdict(list)
        adjacency: dict[str, set[str]] = defaultdict(set)
        for transaction in self.transactions:
            out[transaction.originator_account_id].append(transaction)
            pairs[(transaction.originator_account_id, transaction.beneficiary_account_id)].append(transaction)
            adjacency[transaction.originator_account_id].add(transaction.beneficiary_account_id)
            adjacency[transaction.beneficiary_account_id].add(transaction.originator_account_id)
        self.outbound = {key: tuple(value) for key, value in out.items()}
        self.pair_history = {key: tuple(value) for key, value in pairs.items()}
        self.adjacency = {key: tuple(sorted(value)) for key, value in adjacency.items()}
        self.sar_accounts = frozenset(account_id for account_id, account in dataset.accounts.items() if account.is_sar)

    def prior_outbound(self, transaction: Transaction, hours: int | None = None, days: int | None = None) -> tuple[Transaction, ...]:
        current = parse_timestamp(transaction.timestamp)
        cutoff_seconds = (hours * 3600 if hours is not None else days * 86400 if days is not None else None)
        return tuple(item for item in self.outbound.get(transaction.originator_account_id, ())
                     if (parse_timestamp(item.timestamp), item.id) < (current, transaction.id)
                     and (cutoff_seconds is None or (current - parse_timestamp(item.timestamp)).total_seconds() <= cutoff_seconds))

    def prior_pair(self, transaction: Transaction, days: int = 365) -> tuple[Transaction, ...]:
        current = parse_timestamp(transaction.timestamp)
        return tuple(item for item in self.pair_history.get((transaction.originator_account_id, transaction.beneficiary_account_id), ())
                     if (parse_timestamp(item.timestamp), item.id) < (current, transaction.id)
                     and (current - parse_timestamp(item.timestamp)).days <= days)

    def connected_to_sar(self, account_id: str, max_hops: int = 2) -> tuple[str, ...] | None:
        if account_id in self.sar_accounts:
            return (account_id,)
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(account_id, (account_id,))])
        seen = {account_id}
        while queue:
            node, path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for neighbour in self.adjacency.get(node, ()):
                if neighbour in seen:
                    continue
                candidate = path + (neighbour,)
                if neighbour in self.sar_accounts:
                    return candidate
                seen.add(neighbour)
                queue.append((neighbour, candidate))
        return None

    def historical_behavior(self, transaction: Transaction) -> HistoricalBehavior:
        prior = self.prior_outbound(transaction, days=30)
        average = sum(item.amount for item in prior) / len(prior) if prior else 0.0
        destinations = {item.beneficiary_account_id for item in prior}
        return HistoricalBehavior(transaction.originator_account_id, len(prior), average, len(destinations))

    def relationships_for(self, transaction: Transaction) -> tuple[Relationship, ...]:
        return (
            Relationship(f"rel:{transaction.id}:from", transaction.id, transaction.originator_account_id, "originates_from"),
            Relationship(f"rel:{transaction.id}:to", transaction.id, transaction.beneficiary_account_id, "sent_to"),
        )
