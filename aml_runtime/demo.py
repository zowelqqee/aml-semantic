#!/usr/bin/env python3
"""Run the deterministic AML Decision Runtime on AMLSim CSV data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from aml_runtime import AMLDecisionRuntime, AMLSimLoader
from aml_runtime.visualize import write_decision_graph


ROOT = Path(__file__).resolve().parent


def line(title: str) -> None:
    print("=" * 66)
    print(title)
    print("=" * 66)


def print_result(result) -> None:
    transaction = result.transaction
    line("Transaction")
    print(f"ID       {transaction.id}")
    print(f"Amount   ${transaction.amount:,.2f} {transaction.currency}")
    print(f"Country  {transaction.country_id or 'Unknown'}")
    print(f"Route    {transaction.originator_account_id} -> {transaction.beneficiary_account_id}")
    line("Entity Extraction")
    print(f"Account: {transaction.originator_account_id}")
    print(f"Account: {transaction.beneficiary_account_id}")
    print(f"Country: {transaction.country_id or 'Unknown'}")
    line("Facts")
    if result.facts:
        for fact in result.facts:
            print(f"✓ {fact.type.value}: {fact.explanation}")
    else:
        print("None")
    line("Evidence")
    if result.evidence:
        for evidence in result.evidence:
            print(f"{evidence.id} | {evidence.rule_id} | {evidence.direction} | confidence {evidence.confidence:.2f}")
            print(f"  {evidence.explanation}")
    else:
        print("None")
    line("Conflicts")
    if result.conflicts:
        for conflict in result.conflicts:
            print(f"{conflict.kind}: {conflict.explanation}")
    else:
        print("None")
    line("Policies")
    for policy in result.policies:
        state = "triggered" if policy.triggered else "not triggered"
        print(f"{policy.policy_id} {state} -> {policy.outcome.value}")
    line("Decision")
    print(result.decision.decision.value)
    print(result.decision.rationale)
    print(f"Audit saved: {result.audit['path']}")
    print(f"Replay available: runtime.replay('{transaction.id}')")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="AMLSim transaction CSV or directory. Defaults to AMLSIM_DATASET or bundled AMLSim-format data.")
    parser.add_argument("--transaction-id", help="Execute one transaction ID instead of the representative demo selection.")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "artifacts" / "aml_audit")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "artifacts" / "aml_graphs")
    args = parser.parse_args()
    data_path = args.data or Path(os.environ.get("AMLSIM_DATASET", ROOT / "data" / "amlsim_sample"))
    dataset = AMLSimLoader().load(data_path)
    runtime = AMLDecisionRuntime(dataset, args.audit_dir)
    if args.transaction_id:
        chosen = [args.transaction_id]
    else:
        # The fixture is deliberately ordered to demonstrate review, block, and allow outcomes.
        chosen = ["TX-1004", "TX-1006", "TX-1008"]
        chosen = [item for item in chosen if item in runtime.graph.by_id] or [dataset.transactions[0].id]
    print(f"Loaded {len(dataset.transactions)} AMLSim transactions from {dataset.source_path}")
    for transaction_id in chosen:
        result = runtime.execute(transaction_id)
        print_result(result)
        graph_path = write_decision_graph(result, args.graph_dir / f"{transaction_id}.dot")
        print(f"Graphviz trace: {graph_path}")
        replay = runtime.replay(transaction_id)
        if result.to_dict() != replay.to_dict():
            raise RuntimeError(f"Replay mismatch for {transaction_id}")
        print("Replay verified: identical deterministic output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
