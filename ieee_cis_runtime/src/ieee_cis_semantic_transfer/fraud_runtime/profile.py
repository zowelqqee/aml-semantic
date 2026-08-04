"""Executed source and ontology coverage profile for the fraud-runtime report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .behaviour import BehaviourDecisionRuntime
from .dataset import IEEECISLoader


def profile(transactions: Path, identity_path: Path, output: Path) -> dict[str, object]:
    loader = IEEECISLoader(transactions, identity_path)
    identity = loader.load_identity()
    runtime = BehaviourDecisionRuntime()
    semantic, behaviours, labels, decisions = Counter(), Counter(), Counter(), Counter()
    minimum, maximum, previous, sorted_ok = None, None, -1, True
    identity_rows = 0
    for index, transaction in loader.rows(identity):
        sorted_ok &= transaction.transaction_dt >= previous; previous = transaction.transaction_dt
        minimum = transaction.transaction_dt if minimum is None else min(minimum, transaction.transaction_dt)
        maximum = transaction.transaction_dt if maximum is None else max(maximum, transaction.transaction_dt)
        identity_rows += int(transaction.has_identity)
        result = runtime.evaluate(transaction, transaction.transaction_dt // 60, index)
        semantic.update(item.type.value for item in result.semantic.context.objects)
        behaviours.update(item.type.value for item in result.behaviour.behaviours)
        labels["fraud" if transaction.is_fraud else "legitimate"] += 1
        decisions[result.decision.decision.value] += 1
        runtime.commit(transaction, result, transaction.transaction_dt // 60)
    result = {"rows": sum(labels.values()), "labels": dict(labels), "fraud_rate": labels["fraud"] / sum(labels.values()),
              "transaction_dt": {"minimum": minimum, "maximum": maximum, "span_days": (maximum - minimum) / 86400, "non_decreasing": sorted_ok},
              "identity_join": {"rows": identity_rows, "rate": identity_rows / sum(labels.values())},
              "semantic_object_emissions": dict(sorted(semantic.items())), "behaviour_object_emissions": dict(sorted(behaviours.items())),
              "runtime_decisions": dict(sorted(decisions.items())), "semantic_commits": runtime.semantic.context.events_committed,
              "behaviour_commits": runtime.layer.engine.events_committed}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("ieee_cis_data/train_transaction.csv")); parser.add_argument("--identity", type=Path, default=Path("ieee_cis_data/train_identity.csv")); parser.add_argument("--output", type=Path, default=Path("artifacts/fraud_semantic/source_profile.json"))
    args = parser.parse_args(); print(json.dumps(profile(args.transactions, args.identity, args.output), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
