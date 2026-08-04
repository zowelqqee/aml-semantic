"""Five-arm comparison: does semantic reasoning reduce false positives?

Arms, all on the same frozen chronological protocol (first 500,000 IBM AML
`HI-Small` events; 400,000 train; the next 100,000 as the evaluation horizon):

1. Runtime only              -- the frozen v0.2 transaction-level rule runtime
2. ML only                   -- XGBoost / LightGBM / CatBoost at p >= 0.50
3. Runtime + ML              -- ML probability as evidence into the frozen policy engine
4. Semantic Runtime only     -- the Semantic Context Layer
5. Semantic Runtime + ML     -- ML asked only about semantically undetermined events

Nothing here is tuned.  Every threshold in the semantic layer was declared in
`ontology.py` / `SemanticPolicyEngine` before this module was executed, and the
laundering label is loaded into a separate array that no decision path reads.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import os
import resource
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .dataset import AMLSimDataset
from .ml_benchmark import FEATURE_NAMES, CausalFeatureState, StreamingRuntimeGraph, _account, _amount, _canonical_bank
from .models import Account, Decision, Evidence, Transaction
from .runtime import AMLDecisionRuntime, stable_id
from .runtime_first_cascade import CascadeConfig, _EphemeralAccounts, _models, _sha256
from .semantic import (
    ONTOLOGY_HASH,
    ONTOLOGY_VERSION,
    EntityResolver,
    SemanticContextLayer,
    SemanticDecisionRuntime,
    SemanticRuntimeResult,
    account_key,
    semantic_replay_pins,
)
from .semantic.runtime import SEMANTIC_RULES, SEMANTIC_RUNTIME_VERSION

SEMANTIC_BENCHMARK_VERSION = "aml-semantic-benchmark/1.0"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost")

#: Declared before execution and shared with the v1 cascade experiment.
ML_HIGH_BAND = 0.90
ML_STANDALONE_THRESHOLD = 0.50


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _alerts(decisions: list[str]) -> np.ndarray:
    return np.asarray([1 if item in ("REVIEW", "BLOCK") else 0 for item in decisions], dtype=np.uint8)


def _metrics(labels: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "alert_volume": int(predicted.sum()),
        "alert_rate": float(predicted.mean()),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fp_per_tp": float(fp / tp) if tp else math.inf,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.0,
        "pr_auc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.0,
    }


def _ml_evidence(transaction: Transaction, model: str, probability: float) -> Evidence:
    confidence = round(float(min(0.999999, max(0.0, probability))), 6)
    return Evidence(
        id=stable_id("E", transaction.id, "ML", model, f"{confidence:.6f}"),
        source=f"ML/{model}",
        supporting_facts=(),
        confidence=confidence,
        explanation=f"{model} supplied a label-trained probability as external risk evidence.",
        timestamp=transaction.timestamp,
        rule_id=f"ML-{model}",
        direction="risk",
        topic="ml_probability",
        source_reliability=1.0,
        recency_days=0,
        metadata={"probability": f"{probability:.8f}", "model": model},
    )


class SemanticBenchmark:
    def __init__(self, transactions: str | Path, accounts: str | Path, output_dir: str | Path, config: CascadeConfig | None = None) -> None:
        self.transactions = Path(transactions)
        self.accounts = Path(accounts)
        self.output_dir = Path(output_dir)
        self.config = config or CascadeConfig()
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")

    # -- input -----------------------------------------------------------
    def _rows(self):
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index >= self.config.input_row_cap:
                    return
                yield index, row

    @staticmethod
    def _transaction(index: int, row: list[str], timestamp: datetime) -> Transaction:
        return Transaction(
            id=f"IBM-ML-{index + 2:08d}",
            timestamp=timestamp.isoformat(timespec="seconds"),
            originator_account_id=_account(row[1], row[2]),
            beneficiary_account_id=_account(row[3], row[4]),
            amount=_amount(row[5]),
            currency=row[6],
            country_id="",
            payment_type=row[9],
            metadata={"from_bank": _canonical_bank(row[1]), "to_bank": _canonical_bank(row[3]),
                      "amount_paid": row[7], "payment_currency": row[8]},
        )

    def _account_keys(self) -> set[str]:
        keys: set[str] = set()
        for _index, row in self._rows():
            keys.add(account_key(row[1], row[2]))
            keys.add(account_key(row[3], row[4]))
        return keys

    def _category_maps(self) -> tuple[dict[str, int], ...]:
        values = [set() for _ in range(5)]
        columns = (1, 3, 6, 8, 9)
        for index, row in self._rows():
            if index >= self.config.train_rows:
                break
            for position, column in enumerate(columns):
                values[position].add(row[column])
        return tuple({value: order for order, value in enumerate(sorted(group))} for group in values)

    def _features(self, categories: tuple[dict[str, int], ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_rows, test_rows = self.config.train_rows, self.config.evaluation_horizon
        train_x = np.empty((train_rows, len(FEATURE_NAMES)), dtype=np.float32)
        test_x = np.empty((test_rows, len(FEATURE_NAMES)), dtype=np.float32)
        train_y = np.empty(train_rows, dtype=np.uint8)
        test_y = np.empty(test_rows, dtype=np.uint8)
        state = CausalFeatureState()
        first_day: datetime | None = None
        for index, row in self._rows():
            timestamp = datetime.strptime(row[0], TIMESTAMP_FORMAT)
            first_day = timestamp if first_day is None else first_day
            originator, beneficiary = _account(row[1], row[2]), _account(row[3], row[4])
            received, paid = _amount(row[5]), _amount(row[7])
            vector = state.features(timestamp, originator, beneficiary, received, paid, categories, row, first_day)
            if index < train_rows:
                train_x[index] = vector
                train_y[index] = int(row[10] == "1")
            else:
                test_x[index - train_rows] = vector
                test_y[index - train_rows] = int(row[10] == "1")
            state.commit(timestamp, originator, beneficiary, received)
        return train_x, train_y, test_x, test_y

    # -- decision pass ----------------------------------------------------
    def _decide(self, resolver: EntityResolver, probabilities: dict[str, np.ndarray]) -> dict[str, object]:
        """One chronological pass producing every arm's decisions."""
        fingerprint = _sha256(f"{self.transactions.resolve()}:{self.transactions.stat().st_size}")
        dataset = AMLSimDataset((), _EphemeralAccounts(), {}, str(self.transactions), fingerprint)
        frozen = AMLDecisionRuntime(dataset, self.output_dir / "frozen_runtime_audits_unused")
        graph = StreamingRuntimeGraph(dataset)
        frozen.graph = graph
        fusion = AMLDecisionRuntime(AMLSimDataset((), _EphemeralAccounts(), {}, "hybrid-policy", "hybrid-policy"),
                                    self.output_dir / "hybrid_policy_audits_unused")
        context = SemanticContextLayer(resolver)
        semantic = SemanticDecisionRuntime(context)

        train_rows = self.config.train_rows
        horizon = self.config.evaluation_horizon
        runtime_decisions: list[str] = []
        runtime_scores = np.zeros(horizon, dtype=np.float64)
        semantic_decisions: list[str] = []
        semantic_scores = np.zeros(horizon, dtype=np.float64)
        hybrid_decisions = {name: [] for name in MODEL_NAMES}
        semantic_ml_decisions = {name: [] for name in MODEL_NAMES}
        semantic_ml_scores = {name: np.zeros(horizon, dtype=np.float64) for name in MODEL_NAMES}
        routed = {name: 0 for name in MODEL_NAMES}
        object_census: Counter[str] = Counter()
        evidence_census: Counter[str] = Counter()
        conflict_census: Counter[str] = Counter()
        examples: list[dict[str, object]] = []
        audit_path = self.output_dir / "audits" / "semantic_decisions.jsonl.gz"
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        runtime_seconds = semantic_seconds = 0.0
        with gzip.GzipFile(filename="", mode="wb", fileobj=audit_path.open("wb"), mtime=0) as stream:
            for index, row in self._rows():
                timestamp = datetime.strptime(row[0], TIMESTAMP_FORMAT)
                transaction = self._transaction(index, row, timestamp)
                if index < train_rows:
                    graph.commit(transaction)
                    context.commit(transaction)
                    continue
                position = index - train_rows

                started = time.perf_counter()
                graph.by_id[transaction.id] = transaction
                preliminary = frozen.evaluate_preliminary(transaction.id)
                runtime_seconds += time.perf_counter() - started
                runtime_decisions.append(preliminary.decision.decision.value)
                runtime_scores[position] = float(preliminary.policies[0].metrics["effective_risk"])

                started = time.perf_counter()
                reading = semantic.evaluate(transaction, index)
                semantic_seconds += time.perf_counter() - started
                semantic_decisions.append(reading.decision.decision.value)
                semantic_scores[position] = float(reading.policies[0].metrics["effective_risk"])
                object_census.update(item.type.value for item in reading.context.objects)
                evidence_census.update(item.rule_id for item in reading.evidence)
                conflict_census.update(item.kind for item in reading.conflicts)

                route = SemanticDecisionRuntime.routes_to_ml(reading)
                for name in MODEL_NAMES:
                    probability = float(probabilities[name][position])
                    evidence = _ml_evidence(transaction, name, probability)
                    combined = preliminary.evidence + (evidence,)
                    conflicts = fusion.conflicts.detect(combined)
                    outcomes = fusion.policies.evaluate(combined, conflicts)
                    triggered = tuple(item for item in outcomes if item.triggered)
                    hybrid = next((decision for decision in (Decision.BLOCK, Decision.REVIEW, Decision.ALLOW, Decision.ABSTAIN)
                                   if any(item.outcome == decision for item in triggered)), Decision.ABSTAIN)
                    hybrid_decisions[name].append(hybrid.value)

                    if route:
                        routed[name] += 1
                        fused = semantic.with_ml_evidence(reading, evidence, ML_HIGH_BAND)
                        semantic_ml_decisions[name].append(fused.decision.decision.value)
                        semantic_ml_scores[name][position] = max(semantic_scores[position], probability)
                    else:
                        semantic_ml_decisions[name].append(reading.decision.decision.value)
                        semantic_ml_scores[name][position] = semantic_scores[position]

                stream.write((json.dumps({
                    "transaction_id": transaction.id,
                    "semantic_decision": reading.decision.decision.value,
                    "semantic_policy_ids": list(reading.decision.policy_ids),
                    "object_types": sorted(item.type.value for item in reading.context.objects),
                    "evidence_rules": sorted(item.rule_id for item in reading.evidence),
                    "conflict_kinds": sorted(item.kind for item in reading.conflicts),
                    "routed_to_ml": route,
                    "frozen_runtime_decision": preliminary.decision.decision.value,
                    "context_state_hash": reading.context.context_state_hash,
                }, sort_keys=True, separators=(",", ":")) + "\n").encode())

                if len(examples) < 24 and position % 977 == 0:
                    examples.append(self._example(reading, resolver))

                graph.commit(transaction)
                del graph.by_id[transaction.id]
                context.commit(transaction)

        return {
            "runtime_decisions": runtime_decisions, "runtime_scores": runtime_scores,
            "semantic_decisions": semantic_decisions, "semantic_scores": semantic_scores,
            "hybrid_decisions": hybrid_decisions, "semantic_ml_decisions": semantic_ml_decisions,
            "semantic_ml_scores": semantic_ml_scores, "routed": routed,
            "object_census": object_census, "evidence_census": evidence_census,
            "conflict_census": conflict_census, "examples": examples,
            "runtime_seconds": runtime_seconds, "semantic_seconds": semantic_seconds,
            "audit_path": str(audit_path),
        }

    @staticmethod
    def _example(result: SemanticRuntimeResult, resolver: EntityResolver) -> dict[str, object]:
        record = result.audit_record()
        record["replay_pins"] = semantic_replay_pins(resolver.snapshot_hash, result)
        return record

    # -- execution --------------------------------------------------------
    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        keys = self._account_keys()
        resolver = EntityResolver().load(self.accounts, restrict_to=keys)
        resolution_seconds = time.perf_counter() - started

        categories = self._category_maps()
        started = time.perf_counter()
        train_x, train_y, test_x, test_y = self._features(categories)
        feature_seconds = time.perf_counter() - started

        positives = int(train_y.sum())
        scale = (len(train_y) - positives) / positives
        probabilities: dict[str, np.ndarray] = {}
        model_results: dict[str, dict[str, object]] = {}
        models = _models(self.config, scale)
        for name in MODEL_NAMES:
            model = models.pop(name)
            started = time.perf_counter()
            model.fit(train_x, train_y)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            probability = model.predict_proba(test_x)[:, 1]
            predict_seconds = time.perf_counter() - started
            probabilities[name] = probability
            predicted = (probability >= ML_STANDALONE_THRESHOLD).astype(np.uint8)
            model_results[name] = {
                "metrics": _metrics(test_y, probability, predicted),
                "training_seconds": train_seconds,
                "prediction_latency_ms": 1000 * predict_seconds / len(test_y),
                "ml_inferences": int(len(test_y)),
            }
            del model
            gc.collect()
        del train_x, test_x, models
        gc.collect()

        outcome = self._decide(resolver, probabilities)
        results = self._assemble(test_y, probabilities, model_results, outcome, resolver,
                                 feature_seconds, resolution_seconds, positives)
        self._write(results, test_y, probabilities, outcome)
        return results

    def _assemble(self, labels, probabilities, model_results, outcome, resolver, feature_seconds, resolution_seconds, train_positives) -> dict[str, object]:
        runtime_predicted = _alerts(outcome["runtime_decisions"])
        semantic_predicted = _alerts(outcome["semantic_decisions"])
        arms: dict[str, object] = {
            "runtime_only": {
                "metrics": _metrics(labels, outcome["runtime_scores"], runtime_predicted),
                "decision_distribution": dict(sorted(Counter(outcome["runtime_decisions"]).items())),
                "ml_inferences": 0,
                "decision_latency_ms": 1000 * outcome["runtime_seconds"] / len(labels),
            },
            "semantic_runtime_only": {
                "metrics": _metrics(labels, outcome["semantic_scores"], semantic_predicted),
                "decision_distribution": dict(sorted(Counter(outcome["semantic_decisions"]).items())),
                "ml_inferences": 0,
                "decision_latency_ms": 1000 * outcome["semantic_seconds"] / len(labels),
            },
        }
        for name in MODEL_NAMES:
            hybrid = _alerts(outcome["hybrid_decisions"][name])
            semantic_ml = _alerts(outcome["semantic_ml_decisions"][name])
            flagged = int((probabilities[name] >= ML_STANDALONE_THRESHOLD).sum())
            arms[f"ml_only/{name}"] = {
                **model_results[name],
                "decision_distribution": {"ALERT": flagged, "CLEAR": int(len(labels)) - flagged},
            }
            arms[f"runtime_plus_ml/{name}"] = {
                "metrics": _metrics(labels, np.maximum(outcome["runtime_scores"], probabilities[name]), hybrid),
                "decision_distribution": dict(sorted(Counter(outcome["hybrid_decisions"][name]).items())),
                "ml_inferences": int(len(labels)),
            }
            arms[f"semantic_runtime_plus_ml/{name}"] = {
                "metrics": _metrics(labels, outcome["semantic_ml_scores"][name], semantic_ml),
                "decision_distribution": dict(sorted(Counter(outcome["semantic_ml_decisions"][name]).items())),
                "ml_inferences": int(outcome["routed"][name]),
                "ml_inference_fraction": outcome["routed"][name] / len(labels),
            }
        return {
            "benchmark_version": SEMANTIC_BENCHMARK_VERSION,
            "semantic_runtime_version": SEMANTIC_RUNTIME_VERSION,
            "ontology": {"version": ONTOLOGY_VERSION, "hash": ONTOLOGY_HASH,
                         "entity_snapshot_hash": resolver.snapshot_hash,
                         "resolved_accounts": resolver.resolved_count},
            "protocol": {
                "transactions": str(self.transactions), "accounts": str(self.accounts),
                "input_row_cap": self.config.input_row_cap, "train_rows": self.config.train_rows,
                "evaluation_horizon": self.config.evaluation_horizon,
                "ml_standalone_threshold": ML_STANDALONE_THRESHOLD, "ml_high_band": ML_HIGH_BAND,
                "label_boundary": "the laundering column is read into a separate array used for model fitting and post-decision evaluation only; no decision path, semantic object, profile, or audit reads it",
                "tuning_statement": "every semantic constant was declared in ontology.py before this benchmark executed; no threshold was selected against the evaluation labels and none was changed after seeing a result",
            },
            "class_distribution": {"train_positive": train_positives,
                                   "test_positive": int(labels.sum()),
                                   "test_negative": int(len(labels) - labels.sum())},
            "arms": arms,
            "semantic_object_census": dict(outcome["object_census"].most_common()),
            "semantic_evidence_census": dict(outcome["evidence_census"].most_common()),
            "semantic_conflict_census": dict(outcome["conflict_census"].most_common()),
            "timing": {"entity_resolution_seconds": resolution_seconds, "feature_seconds": feature_seconds,
                       "frozen_runtime_seconds": outcome["runtime_seconds"],
                       "semantic_runtime_seconds": outcome["semantic_seconds"]},
            "peak_rss_bytes": _peak_rss_bytes(),
            "audit_stream": outcome["audit_path"],
        }

    # -- reporting --------------------------------------------------------
    def _write(self, results: dict[str, object], labels: np.ndarray, probabilities: dict[str, np.ndarray], outcome: dict[str, object]) -> None:
        (self.output_dir / "comparison_results.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        with (self.output_dir / "semantic_object_census.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("semantic_type", "emissions_over_evaluation_horizon"))
            writer.writerows(results["semantic_object_census"].items())

        samples = self.output_dir / "audits" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        for record in outcome["examples"]:
            (samples / f"{record['transaction']['id']}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        lines = ["# Semantic decision examples", "",
                 "Every rationale below is stated in semantic objects. No transaction-level rule identifier appears in it.", ""]
        for record in outcome["examples"]:
            lines += [f"## {record['transaction']['id']} — {record['decision']['decision']}", "",
                      f"- Reading: {record['semantic_rationale']}",
                      f"- Objects: {', '.join(item['type'] for item in record['semantic']['objects'])}",
                      f"- Evidence: {', '.join(item['rule_id'] for item in record['evidence']) or 'none'}",
                      f"- Conflicts: {', '.join(item['kind'] for item in record['conflicts']) or 'none'}",
                      f"- Routed to ML: {record['routed_to_ml']}", ""]
        (self.output_dir / "decision_examples.md").write_text("\n".join(lines), encoding="utf-8")

        curves = {"Semantic Runtime": outcome["semantic_scores"], "Frozen Runtime": outcome["runtime_scores"]}
        for name in MODEL_NAMES:
            curves[f"{name}"] = probabilities[name]
            curves[f"Semantic + {name}"] = outcome["semantic_ml_scores"][name]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, values in curves.items():
            false_positive, true_positive, _ = roc_curve(labels, values)
            precision, recall, _ = precision_recall_curve(labels, values)
            axes[0].plot(false_positive, true_positive, label=f"{name} (AUC={roc_auc_score(labels, values):.3f})")
            axes[1].plot(recall, precision, label=f"{name} (AP={average_precision_score(labels, values):.4f})")
        axes[0].plot((0, 1), (0, 1), "k--", linewidth=0.8)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.output_dir / "curves.png", dpi=160)
        plt.close(figure)

        arms = results["arms"]
        rows = []
        for key in sorted(arms):
            metric = arms[key].get("metrics")
            if not metric:
                continue
            rows.append((key, metric["tp"], metric["fp"], metric["fn"], f"{metric['recall']:.4f}",
                         f"{metric['precision']:.6f}", metric["alert_volume"], f"{metric['alert_rate']:.4f}",
                         arms[key].get("ml_inferences", "")))
        header = ("Arm", "TP", "FP", "FN", "Recall", "Precision", "Alerts", "Alert rate", "ML inferences")
        table = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        table += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        census = results["semantic_object_census"]
        census_rows = ["| " + " | ".join(("Semantic type", "Emissions")) + " |", "|---|---|"]
        census_rows += [f"| {key} | {value:,} |" for key, value in census.items()]
        report = [
            "# Semantic Context Layer — five-arm benchmark", "",
            f"Protocol: `{results['protocol']['transactions']}`, first {results['protocol']['input_row_cap']:,} chronological events; "
            f"{results['protocol']['train_rows']:,} train; {results['protocol']['evaluation_horizon']:,}-event evaluation horizon "
            f"containing {results['class_distribution']['test_positive']} laundering-labelled events.", "",
            f"Ontology `{results['ontology']['version']}` hash `{results['ontology']['hash'][:16]}…`; "
            f"{results['ontology']['resolved_accounts']:,} accounts resolved from reference data.", "",
            results["protocol"]["tuning_statement"].capitalize() + ".", "",
            "## Results", "", *table, "",
            "## Semantic object census (evaluation horizon)", "", *census_rows, "",
            "## Files", "",
            "- `comparison_results.json` — every measured number",
            "- `audits/semantic_decisions.jsonl.gz` — one compact audit line per evaluated event",
            "- `audits/samples/` — full semantic audit records with replay pins",
            "- `decision_examples.md` — decisions stated in semantic objects",
            "- `curves.png` — ROC and precision-recall", "",
        ]
        (self.output_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("data/ibm_aml_data/HI-Small_Trans.csv"))
    parser.add_argument("--accounts", type=Path, default=Path("data/ibm_aml_data/HI-Small_accounts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aml_semantic_v1"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    results = SemanticBenchmark(args.transactions, args.accounts, args.output_dir, CascadeConfig(threads=args.threads)).run()
    print(json.dumps({"output_dir": str(args.output_dir),
                      "arms": {key: value.get("metrics") for key, value in results["arms"].items()}},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
