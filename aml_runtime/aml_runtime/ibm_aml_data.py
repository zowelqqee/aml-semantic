"""Native adapter and chronological benchmark runner for IBM AML-Data CSV files.

The IBM AML-Data Is Laundering column is deliberately retained only in
IBMAMLRecord.laundering_label. It is never copied into Transaction, Account,
graph state, facts, evidence, policies, or audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .dataset import AMLSimDataset
from .models import Account, Decision, Transaction
from .research import ResearchMetrics, write_metrics_chart, write_publication_report


IBM_TRANSACTION_COLUMNS = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)
IBM_ACCOUNT_COLUMNS = ("Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name")
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
ACCOUNT_TOKEN = re.compile(r"^[A-Za-z0-9]+$")


class IBMAMLDataValidationError(ValueError):
    """A source row cannot be represented safely by the deterministic runtime."""


@dataclass(frozen=True)
class IBMAMLRecord:
    """Validated source event; ground truth is retained only for evaluation."""

    transaction: Transaction
    occurred_at: datetime
    laundering_label: bool
    source_line: int


@dataclass(frozen=True)
class IBMAMLScan:
    schema: tuple[str, ...]
    file_size_bytes: int
    source_rows: int
    accepted_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int]
    first_five: tuple[IBMAMLRecord, ...]
    chronological_sample: tuple[IBMAMLRecord, ...]


def _canonical_bank(value: str) -> str:
    cleaned = value.strip()
    if not cleaned.isdigit():
        raise IBMAMLDataValidationError("malformed bank identifier")
    return str(int(cleaned))


def _canonical_account(bank: str, account: str) -> str:
    cleaned = account.strip()
    if not cleaned or not ACCOUNT_TOKEN.fullmatch(cleaned):
        raise IBMAMLDataValidationError("malformed account identifier")
    return f"{_canonical_bank(bank)}:{cleaned.upper()}"


def _parse_label(value: str) -> bool:
    cleaned = value.strip()
    if cleaned == "0":
        return False
    if cleaned == "1":
        return True
    raise IBMAMLDataValidationError("invalid laundering label")


def _parse_amount(value: str) -> float:
    try:
        amount = float(value)
    except ValueError as exc:
        raise IBMAMLDataValidationError("invalid amount") from exc
    if not math.isfinite(amount) or amount <= 0:
        raise IBMAMLDataValidationError("invalid amount")
    return amount


@lru_cache(maxsize=None)
def _parse_timestamp(value: str) -> datetime:
    """Parse each distinct native IBM minute timestamp exactly once."""
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise IBMAMLDataValidationError("malformed timestamp") from exc


class IBMAMLDataLoader:
    """Read IBM AML-Data's native CSV schema without preprocessing or renaming."""

    def __init__(self, transaction_path: str | Path, account_path: str | Path | None = None) -> None:
        self.transaction_path = Path(transaction_path)
        self.account_path = Path(account_path) if account_path else None

    def _read_rows(self) -> Iterable[tuple[int, list[str]]]:
        with self.transaction_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                schema = tuple(next(reader))
            except StopIteration as exc:
                raise IBMAMLDataValidationError("empty transaction file") from exc
            if schema != IBM_TRANSACTION_COLUMNS:
                raise IBMAMLDataValidationError(
                    f"unexpected IBM AML-Data schema: {schema!r}; expected {IBM_TRANSACTION_COLUMNS!r}"
                )
            for source_line, row in enumerate(reader, start=2):
                yield source_line, row

    def schema(self) -> tuple[str, ...]:
        with self.transaction_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                return tuple(next(reader))
            except StopIteration as exc:
                raise IBMAMLDataValidationError("empty transaction file") from exc

    def _validate_row(self, row: list[str]) -> tuple[datetime, str, str, float, str, str, str, str, bool]:
        if len(row) != len(IBM_TRANSACTION_COLUMNS):
            raise IBMAMLDataValidationError("wrong column count")
        (
            timestamp_text,
            from_bank,
            from_account,
            to_bank,
            to_account,
            amount_received,
            receiving_currency,
            amount_paid,
            payment_currency,
            payment_format,
            laundering_label,
        ) = row
        occurred_at = _parse_timestamp(timestamp_text.strip())
        originator = _canonical_account(from_bank, from_account)
        beneficiary = _canonical_account(to_bank, to_account)
        amount = _parse_amount(amount_received)
        _parse_amount(amount_paid)
        if not receiving_currency.strip() or not payment_currency.strip() or not payment_format.strip():
            raise IBMAMLDataValidationError("missing transaction attribute")
        label = _parse_label(laundering_label)
        return (
            occurred_at,
            originator,
            beneficiary,
            amount,
            amount_paid.strip(),
            receiving_currency.strip(),
            payment_currency.strip(),
            payment_format.strip(),
            label,
        )

    def parse_row(self, source_line: int, row: list[str]) -> IBMAMLRecord:
        (
            occurred_at,
            originator,
            beneficiary,
            amount,
            amount_paid,
            receiving_currency,
            payment_currency,
            payment_format,
            label,
        ) = self._validate_row(row)
        # Reuse the native bank values only after strict validation.
        from_bank = _canonical_bank(row[1])
        to_bank = _canonical_bank(row[3])
        transaction = Transaction(
            id=f"IBM-AML-DATA-{source_line:08d}",
            timestamp=occurred_at.isoformat(timespec="seconds"),
            originator_account_id=originator,
            beneficiary_account_id=beneficiary,
            amount=amount,
            currency=receiving_currency.strip(),
            country_id="",
            payment_type=payment_format.strip(),
            metadata={
                "amount_paid": amount_paid,
                "from_bank": from_bank,
                "payment_currency": payment_currency,
                "receiving_currency": receiving_currency,
                "to_bank": to_bank,
            },
        )
        return IBMAMLRecord(transaction, occurred_at, label, source_line)

    def scan_initial_chronological(self, sample_size: int) -> IBMAMLScan:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        schema = self.schema()
        if schema != IBM_TRANSACTION_COLUMNS:
            raise IBMAMLDataValidationError(
                f"unexpected IBM AML-Data schema: {schema!r}; expected {IBM_TRANSACTION_COLUMNS!r}"
            )
        earliest: list[tuple[int, int, list[str]]] = []
        first_five: list[IBMAMLRecord] = []
        reasons: Counter[str] = Counter()
        source_rows = accepted_rows = 0
        for source_line, row in self._read_rows():
            source_rows += 1
            try:
                occurred_at, *_unused, label = self._validate_row(row)
            except IBMAMLDataValidationError as exc:
                reasons[str(exc)] += 1
                continue
            accepted_rows += 1
            if len(first_five) < 5:
                first_five.append(self.parse_row(source_line, row))
            # The native timestamps are timezone-free civil times.  Construct
            # their sortable ordinal directly instead of calling
            # ``datetime.timestamp()``, which consults the host timezone for
            # every source row and is both slower and less portable.
            epoch_seconds = (
                occurred_at.toordinal() * 86_400
                + occurred_at.hour * 3_600
                + occurred_at.minute * 60
                + occurred_at.second
            )
            # Do not allocate a runtime Transaction unless this source row is
            # in the bounded earliest-event candidate set.
            heap_item = (-epoch_seconds, -source_line, row)
            if len(earliest) < sample_size:
                heapq.heappush(earliest, heap_item)
            elif heap_item[:2] > earliest[0][:2]:
                heapq.heapreplace(earliest, heap_item)
        sample = tuple(
            self.parse_row(-item[1], item[2])
            for item in sorted(
                earliest,
                key=lambda item: (-item[0], -item[1]),
            )
        )
        return IBMAMLScan(
            schema=schema,
            file_size_bytes=self.transaction_path.stat().st_size,
            source_rows=source_rows,
            accepted_rows=accepted_rows,
            rejected_rows=sum(reasons.values()),
            rejection_reasons=dict(sorted(reasons.items())),
            first_five=tuple(first_five),
            chronological_sample=sample,
        )

    def source_account_count(self) -> int:
        if self.account_path is None:
            return 0
        with self.account_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                schema = tuple(next(reader))
            except StopIteration as exc:
                raise IBMAMLDataValidationError("empty account file") from exc
            if schema != IBM_ACCOUNT_COLUMNS:
                raise IBMAMLDataValidationError(f"unexpected IBM AML-Data account schema: {schema!r}")
            count = 0
            for source_line, row in enumerate(reader, start=2):
                if len(row) != len(IBM_ACCOUNT_COLUMNS):
                    raise IBMAMLDataValidationError(f"malformed account row at line {source_line}")
                _canonical_account(row[1], row[2])
                count += 1
            return count


class _ChronologicalGraph:
    """Causal graph interface consumed by the unchanged FactExtractor."""

    def __init__(self, dataset: AMLSimDataset, occurred_at: dict[str, datetime]) -> None:
        self.dataset = dataset
        self.by_id: dict[str, Transaction] = {}
        self._occurred_at = occurred_at
        self._outbound: dict[str, list[Transaction]] = defaultdict(list)
        self._pair_history: dict[tuple[str, str], list[Transaction]] = defaultdict(list)
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        # The source account export has no independent SAR field. Ground-truth
        # labels are not converted to graph or SAR state.
        self.sar_accounts: frozenset[str] = frozenset()
        self.temporal_edge_events = 0

    def prior_outbound(self, transaction: Transaction, hours: int | None = None, days: int | None = None) -> tuple[Transaction, ...]:
        current = self._occurred_at[transaction.id]
        limit = timedelta(hours=hours) if hours is not None else timedelta(days=days) if days is not None else None
        return tuple(
            item
            for item in self._outbound[transaction.originator_account_id]
            if limit is None or current - self._occurred_at[item.id] <= limit
        )

    def prior_pair(self, transaction: Transaction, days: int = 365) -> tuple[Transaction, ...]:
        current = self._occurred_at[transaction.id]
        limit = timedelta(days=days)
        return tuple(
            item
            for item in self._pair_history[(transaction.originator_account_id, transaction.beneficiary_account_id)]
            if current - self._occurred_at[item.id] <= limit
        )

    def connected_to_sar(self, account_id: str, max_hops: int = 2) -> tuple[str, ...] | None:
        return None

    def commit(self, transaction: Transaction) -> None:
        self._outbound[transaction.originator_account_id].append(transaction)
        self._pair_history[(transaction.originator_account_id, transaction.beneficiary_account_id)].append(transaction)
        self.adjacency[transaction.originator_account_id].add(transaction.beneficiary_account_id)
        self.adjacency[transaction.beneficiary_account_id].add(transaction.originator_account_id)
        self.temporal_edge_events += 1


def _source_fingerprint(transaction_path: Path, account_path: Path | None) -> str:
    digest = hashlib.sha256()
    for path in (transaction_path, account_path):
        if path is None:
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _peak_memory_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class IBMAMLChronologicalRunner:
    """Run unchanged rules, conflicts, policies, and precedence on causal input."""

    def __init__(self, transaction_path: str | Path, account_path: str | Path | None, audit_directory: str | Path) -> None:
        self.loader = IBMAMLDataLoader(transaction_path, account_path)
        self.audit_directory = Path(audit_directory)

    def run(
        self,
        sample_size: int,
        report_path: str | Path,
        research_report_path: str | Path | None = None,
        chart_path: str | Path | None = None,
    ) -> dict[str, object]:
        scan_started = time.perf_counter()
        scan = self.loader.scan_initial_chronological(sample_size)
        source_account_count = self.loader.source_account_count()
        scan_seconds = time.perf_counter() - scan_started
        if len(scan.chronological_sample) != sample_size:
            raise IBMAMLDataValidationError(
                f"only {len(scan.chronological_sample)} valid transactions available; requested {sample_size}"
            )

        accounts: dict[str, Account] = {}
        occurred_at: dict[str, datetime] = {}
        labels: dict[str, bool] = {}
        for record in scan.chronological_sample:
            tx = record.transaction
            accounts.setdefault(tx.originator_account_id, Account(id=tx.originator_account_id, bank_id=tx.metadata["from_bank"]))
            accounts.setdefault(tx.beneficiary_account_id, Account(id=tx.beneficiary_account_id, bank_id=tx.metadata["to_bank"]))
            occurred_at[tx.id] = record.occurred_at
            labels[tx.id] = record.laundering_label
        dataset = AMLSimDataset(
            transactions=(),
            accounts=accounts,
            countries={},
            source_path=str(self.loader.transaction_path),
            fingerprint=_source_fingerprint(self.loader.transaction_path, self.loader.account_path),
        )

        # Import and invoke the existing runtime unchanged.
        from .runtime import AMLDecisionRuntime

        runtime = AMLDecisionRuntime(dataset, self.audit_directory)
        graph = _ChronologicalGraph(dataset, occurred_at)
        runtime.graph = graph
        research_metrics = ResearchMetrics(tuple(rule.id for rule in runtime.rules.rules))
        decisions: Counter[str] = Counter()
        labels_seen: Counter[str] = Counter()
        decision_by_label: Counter[tuple[str, str]] = Counter()
        latencies_ms: list[float] = []
        evidence_count = conflict_count = audit_count = 0
        representative: dict[str, dict[str, object]] = {}
        representative_candidates: list[dict[str, object]] = []
        started = time.perf_counter()
        for record in scan.chronological_sample:
            tx = record.transaction
            graph.by_id[tx.id] = tx
            transaction_started = time.perf_counter_ns()
            result = runtime.execute(tx.id)
            latency_ms = (time.perf_counter_ns() - transaction_started) / 1_000_000
            latencies_ms.append(latency_ms)
            # Commit only after all decision stages finish.
            graph.commit(tx)
            decision = result.decision.decision.value
            label_name = "laundering" if labels[tx.id] else "non_laundering"
            decisions[decision] += 1
            labels_seen[label_name] += 1
            decision_by_label[(label_name, decision)] += 1
            evidence_count += len(result.evidence)
            conflict_count += len(result.conflicts)
            audit_count += 1
            research_metrics.observe(result)
            case = {
                "transaction_id": tx.id,
                "timestamp": tx.timestamp,
                "amount": tx.amount,
                "decision": decision,
                "laundering_label": label_name,
                "fact_types": [fact.type.value for fact in result.facts],
                "conflict_count": len(result.conflicts),
            }
            desired = (
                ("laundering_allowed", labels[tx.id] and decision == Decision.ALLOW.value),
                ("laundering_reviewed", labels[tx.id] and decision == Decision.REVIEW.value),
                ("laundering_blocked", labels[tx.id] and decision == Decision.BLOCK.value),
                ("non_laundering_reviewed_or_blocked", not labels[tx.id] and decision in {Decision.REVIEW.value, Decision.BLOCK.value}),
                ("abstained", decision == Decision.ABSTAIN.value),
                ("conflict", bool(result.conflicts)),
            )
            for name, matched in desired:
                if matched and name not in representative:
                    representative[name] = case
            if any(matched for _name, matched in desired):
                representative_candidates.append(case)
        runtime_seconds = time.perf_counter() - started
        ordered_latencies = sorted(latencies_ms)
        p95_index = max(0, math.ceil(len(ordered_latencies) * 0.95) - 1)
        unique_undirected_edges = sum(len(destinations) for destinations in graph.adjacency.values()) // 2
        categories = (
            "laundering_allowed",
            "laundering_reviewed",
            "laundering_blocked",
            "non_laundering_reviewed_or_blocked",
            "abstained",
            "conflict",
        )
        # Preserve category coverage first, then fill deterministically in
        # chronological processing order.  No unavailable category is
        # fabricated merely to fill the requested ten cases.
        representative_cases: list[dict[str, object]] = []
        selected_ids: set[str] = set()
        for name in categories:
            case = representative.get(name)
            if case is not None:
                representative_cases.append(case)
                selected_ids.add(str(case["transaction_id"]))
        for case in representative_candidates:
            if len(representative_cases) == 10:
                break
            transaction_id = str(case["transaction_id"])
            if transaction_id not in selected_ids:
                representative_cases.append(case)
                selected_ids.add(transaction_id)
        report: dict[str, object] = {
            "dataset": {
                "transaction_file": str(self.loader.transaction_path),
                "account_file": str(self.loader.account_path) if self.loader.account_path else None,
                "file_size_bytes": scan.file_size_bytes,
                "schema": list(scan.schema),
                "source_rows": scan.source_rows,
                "accepted_rows": scan.accepted_rows,
                "rejected_rows": scan.rejected_rows,
                "rejection_reasons": scan.rejection_reasons,
                "source_account_rows": source_account_count,
                "source_sha256": dataset.fingerprint,
            },
            "smoke_transactions": [
                {
                    "source_line": record.source_line,
                    "occurred_at": record.occurred_at.isoformat(),
                    "transaction": record.transaction.to_dict(),
                    # Evaluation only; never admitted to runtime state.
                    "evaluation_laundering_label": record.laundering_label,
                }
                for record in scan.first_five
            ],
            "chronological_sample": {
                "requested_transactions": sample_size,
                "processed_transactions": len(scan.chronological_sample),
                "first_timestamp": scan.chronological_sample[0].occurred_at.isoformat(),
                "last_timestamp": scan.chronological_sample[-1].occurred_at.isoformat(),
                "unique_accounts": len(accounts),
                "graph_nodes": len(graph.adjacency),
                "temporal_edge_events": graph.temporal_edge_events,
                "unique_undirected_account_edges": unique_undirected_edges,
            },
            "evaluation": {
                "decision_distribution": {decision.value: decisions[decision.value] for decision in Decision},
                "laundering_label_distribution": dict(sorted(labels_seen.items())),
                "decision_distribution_by_laundering_label": {
                    label: {decision.value: decision_by_label[(label, decision.value)] for decision in Decision}
                    for label in ("laundering", "non_laundering")
                },
                "confusion_style_table": [
                    {
                        "laundering_label": label,
                        "ALLOW": decision_by_label[(label, Decision.ALLOW.value)],
                        "REVIEW": decision_by_label[(label, Decision.REVIEW.value)],
                        "BLOCK": decision_by_label[(label, Decision.BLOCK.value)],
                        "ABSTAIN": decision_by_label[(label, Decision.ABSTAIN.value)],
                    }
                    for label in ("laundering", "non_laundering")
                ],
                "rules_triggered": evidence_count,
                "conflicts_produced": conflict_count,
                "audit_records_produced": audit_count,
                "representative_cases": representative_cases,
                "representative_cases_by_category": representative,
                "unavailable_representative_categories": [name for name in categories if name not in representative],
            },
            "research_metrics": research_metrics.to_dict(),
            "performance": {
                "scan_and_selection_seconds": scan_seconds,
                "runtime_seconds": runtime_seconds,
                "end_to_end_seconds": time.perf_counter() - scan_started,
                "average_latency_ms": sum(latencies_ms) / len(latencies_ms),
                "p95_latency_ms": ordered_latencies[p95_index],
                "peak_memory_bytes": _peak_memory_bytes(),
            },
            "causality": {
                "laundering_labels_passed_to_runtime": False,
                "edges_committed_after_decision": True,
                "transactions_processed_in_chronological_order": True,
            },
        }
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if chart_path is not None:
            chart_destination = write_metrics_chart(report["research_metrics"], chart_path)
            report["research_artifacts"] = {"chart_path": str(chart_destination)}
        if research_report_path is not None:
            research_destination = Path(research_report_path)
            chart_reference = str(chart_path) if chart_path is not None else ""
            if chart_reference and research_destination.parent.name == "docs":
                chart_reference = "../" + chart_reference
            write_publication_report(report, research_destination, chart_reference)
            report.setdefault("research_artifacts", {})["publication_report_path"] = str(research_destination)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--accounts", type=Path)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--audit-dir", type=Path, default=Path("artifacts/ibm_aml_data_audit"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/ibm_aml_data_report.json"))
    parser.add_argument("--research-report", type=Path, default=Path("docs/aml_runtime_v0_2_report.md"))
    parser.add_argument("--chart", type=Path, default=Path("artifacts/ibm_aml_data_v02_charts.svg"))
    args = parser.parse_args()
    report = IBMAMLChronologicalRunner(args.transactions, args.accounts, args.audit_dir).run(
        args.sample_size, args.report, args.research_report, args.chart
    )
    print(json.dumps({
        "dataset": report["dataset"],
        "smoke_transactions": report["smoke_transactions"],
        "chronological_sample": report["chronological_sample"],
        "evaluation": report["evaluation"],
        "research_metrics": report["research_metrics"],
        "performance": report["performance"],
        "causality": report["causality"],
        "research_artifacts": report.get("research_artifacts", {}),
        "report_path": str(args.report),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
