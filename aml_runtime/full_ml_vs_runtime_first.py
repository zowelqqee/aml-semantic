"""Decisive, frozen-protocol comparison of full ML and Runtime-first routing."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ml_benchmark import AMLMLBenchmark, BenchmarkConfig
from .models import Decision
from .runtime_first_cascade import (
    CascadeConfig,
    CascadeDatasetBuilder,
    _metrics,
    _models,
    _table,
)


MODELS = ("XGBoost", "LightGBM", "CatBoost")
CHUNK_SIZE = 1_000


def _chunked_predict(model: object, features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
    """Measure the same batched prediction scope for full and routed ML."""
    indices = np.flatnonzero(mask)
    result = np.full(len(features), np.nan, dtype=np.float64)
    latencies: list[float] = []
    started = time.perf_counter()
    for offset in range(0, len(indices), CHUNK_SIZE):
        batch = indices[offset:offset + CHUNK_SIZE]
        batch_started = time.perf_counter()
        result[batch] = model.predict_proba(features[batch])[:, 1]
        latencies.extend([time.perf_counter() - batch_started] * len(batch))
    return result, time.perf_counter() - started, latencies


def _binary_decisions(probabilities: np.ndarray, threshold: float = 0.50) -> list[Decision]:
    return [Decision.REVIEW if probability >= threshold else Decision.ALLOW for probability in probabilities]


def _iter_audits(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _pair_label(label: int, full_alert: int, cascade_alert: int) -> tuple[str, str]:
    full_correct = full_alert == label
    cascade_correct = cascade_alert == label
    if full_correct and cascade_correct:
        outcome = "both_correct"
    elif full_correct:
        outcome = "full_ml_correct_cascade_wrong"
    elif cascade_correct:
        outcome = "cascade_correct_full_ml_wrong"
    else:
        outcome = "both_wrong"
    if label:
        special = "caught_by_both" if full_alert and cascade_alert else "caught_by_full_ml_only" if full_alert else "caught_by_cascade_only" if cascade_alert else "missed_by_both"
    else:
        special = "false_alert_both" if full_alert and cascade_alert else "false_alert_full_ml_only" if full_alert else "false_alert_cascade_only" if cascade_alert else "clean_no_alert"
    return outcome, special


class FullMLVsRuntimeFirst:
    """A label-safe paired evaluation over previously pinned cascade audits."""

    def __init__(self, transactions: str | Path, output_dir: str | Path) -> None:
        self.transactions = Path(transactions)
        self.output_dir = Path(output_dir)
        self.cascade_dir = Path("artifacts/aml_runtime_first_cascade")
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")
        self.config = CascadeConfig()

    def _raw_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Execute the unchanged raw 18-feature full-ML preparation once."""
        benchmark = AMLMLBenchmark(self.transactions, "artifacts/aml_ml_benchmark", BenchmarkConfig())
        benchmark._ensure_sorted()
        categories = benchmark._category_maps(400_000)
        started = time.perf_counter()
        train_x, train_y, test_x, test_y, _metadata = benchmark._build_feature_sets(500_000, 400_000, 100_000, categories)
        return train_x, train_y, test_x, test_y, time.perf_counter() - started

    def run(self) -> dict[str, object]:
        if not self.cascade_dir.joinpath("comparison_results.json").exists():
            raise FileNotFoundError("completed Runtime-first cascade artifacts are required")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_train, raw_y, raw_test, labels, full_feature_seconds = self._raw_features()
        # This invokes the frozen causal Runtime-first feature contract exactly
        # as used by the completed cascade.  It carries labels only in arrays.
        cascade_data = CascadeDatasetBuilder(self.transactions, self.sorted_path, self.config).build()
        if not np.array_equal(labels, cascade_data["labels_test"]):
            raise AssertionError("full-ML and Runtime-first test labels diverged")
        primary_train = np.hstack((cascade_data["raw_train"], cascade_data["runtime_full_train"]))
        primary_test = np.hstack((cascade_data["raw_test"], cascade_data["runtime_full_test"]))
        preliminary = np.asarray([tuple(Decision)[int(value)].value for value in cascade_data["preliminary_codes"]], dtype="U7")
        routed_mask = (preliminary == Decision.ALLOW.value) | (preliminary == Decision.ABSTAIN.value)
        scale = (len(raw_y) - int(raw_y.sum())) / int(raw_y.sum())
        cascade_report = json.loads(self.cascade_dir.joinpath("comparison_results.json").read_text())
        frozen_full_report = json.loads(Path("artifacts/aml_ml_benchmark/comparison_results.json").read_text())
        cases: dict[str, dict[str, list[dict[str, object]]]] = {name: defaultdict(list) for name in MODELS}
        transition_counts: Counter[tuple[str, str, int]] = Counter()
        full_results: dict[str, dict[str, object]] = {}
        cascade_results: dict[str, dict[str, object]] = {}
        compute_rows: list[dict[str, object]] = []
        pairwise_path = self.output_dir / "pairwise_outcomes.csv"
        with pairwise_path.open("w", encoding="utf-8", newline="") as pairwise_handle:
            pairwise_writer = csv.writer(pairwise_handle)
            pairwise_writer.writerow(("model", "transaction_id", "timestamp", "laundering_label_evaluation_only", "full_ml_probability", "full_ml_alert", "cascade_final_decision", "cascade_alert", "pairwise_outcome", "special_category", "runtime_transition"))
            for name, full_model in _models(self.config, scale).items():
                train_started = time.perf_counter()
                full_model.fit(raw_train, raw_y)
                full_training_seconds = time.perf_counter() - train_started
                full_probability, full_prediction_seconds, full_latency_samples = _chunked_predict(full_model, raw_test, np.ones(len(raw_test), dtype=bool))
                full_decisions = _binary_decisions(full_probability)
                full_metric = _metrics(labels, full_probability, full_decisions)
                del full_model
                cascade_model = _models(self.config, scale)[name]
                train_started = time.perf_counter()
                cascade_model.fit(primary_train, raw_y)
                cascade_training_seconds = time.perf_counter() - train_started
                _routed_probability, routed_prediction_seconds, routed_latency_samples = _chunked_predict(cascade_model, primary_test, routed_mask)
                del cascade_model
                primary = cascade_report["cascade_results"][f"Runtime → {name} → Runtime"]
                assert isinstance(primary, dict)
                cascade_metric = primary["metrics"]
                audit_count = 0
                observed = Counter()
                for index, audit in enumerate(_iter_audits(self.cascade_dir / "audits" / f"runtime_first_{name.lower()}.jsonl.gz")):
                    audit_count += 1
                    cascade_decision = Decision(audit["final_decision"]["decision"])
                    observed[cascade_decision.value] += 1
                    transaction = audit["transaction"]
                    assert isinstance(transaction, dict)
                    label = int(labels[index])
                    full_alert = int(full_probability[index] >= 0.50)
                    cascade_alert = int(cascade_decision in {Decision.REVIEW, Decision.BLOCK})
                    outcome, special = _pair_label(label, full_alert, cascade_alert)
                    preliminary_decision = audit["preliminary_decision"]["decision"]
                    final_decision = audit["final_decision"]["decision"]
                    transition_counts[(name, f"{preliminary_decision} → {final_decision}", label)] += 1
                    pairwise_writer.writerow((name, transaction["id"], transaction["timestamp"], label, f"{full_probability[index]:.8f}", full_alert, final_decision, cascade_alert, outcome, special, f"{preliminary_decision} → {final_decision}"))
                    if len(cases[name][outcome]) < 10:
                        cases[name][outcome].append({"transaction_id": transaction["id"], "timestamp": transaction["timestamp"], "label_evaluation_only": label, "full_ml_probability": float(full_probability[index]), "full_ml_alert": full_alert, "cascade_decision": final_decision, "cascade_alert": cascade_alert, "pairwise_outcome": outcome, "special_category": special, "causal_trace": audit})
                    if len(cases[name][special]) < 10:
                        cases[name][special].append({"transaction_id": transaction["id"], "timestamp": transaction["timestamp"], "label_evaluation_only": label, "full_ml_probability": float(full_probability[index]), "full_ml_alert": full_alert, "cascade_decision": final_decision, "cascade_alert": cascade_alert, "pairwise_outcome": outcome, "special_category": special, "causal_trace": audit})
                if audit_count != len(labels) or dict(observed) != primary["decision_distribution"]:
                    raise AssertionError(f"pinned audit stream diverged for {name}")
                full_results[name] = {"metrics": full_metric, "training_seconds": full_training_seconds, "feature_seconds": full_feature_seconds, "prediction_seconds": full_prediction_seconds, "inference_mean_ms": 1000 * full_prediction_seconds / len(labels), "inference_p95_ms": float(1000 * np.percentile(full_latency_samples, 95)), "inferences": len(labels), "peak_rss_bytes": frozen_full_report["model_results"][name]["peak_process_rss_bytes"], "peak_rss_source": "executed frozen full-ML benchmark"}
                cascade_results[name] = {"metrics": cascade_metric, "decisions": primary["decision_distribution"], "routing": primary["routing"], "latency": primary["latency"], "peak_process_rss_bytes": primary["peak_process_rss_bytes"], "cascade_feature_seconds": cascade_report["feature_engineering_seconds"], "training_seconds": cascade_training_seconds, "routed_prediction_seconds": routed_prediction_seconds, "routed_inference_mean_ms": 1000 * routed_prediction_seconds / int(routed_mask.sum()), "routed_inference_p95_ms": float(1000 * np.percentile(routed_latency_samples, 95)), "inferences": int(routed_mask.sum())}
                compute_rows.extend((
                {"model": name, "system": "Full ML", "feature_seconds": full_feature_seconds, "runtime_first_seconds": 0.0, "prediction_seconds": full_prediction_seconds, "final_policy_seconds": 0.0, "inferences": len(labels), "inference_mean_ms": 1000 * full_prediction_seconds / len(labels), "inference_p95_ms": float(1000 * np.percentile(full_latency_samples, 95)), "end_to_end_mean_ms": 1000 * full_prediction_seconds / len(labels), "end_to_end_p95_ms": float(1000 * np.percentile(full_latency_samples, 95)), "peak_rss_bytes": frozen_full_report["model_results"][name]["peak_process_rss_bytes"]},
                {"model": name, "system": "Runtime-first cascade", "feature_seconds": cascade_report["feature_engineering_seconds"], "runtime_first_seconds": cascade_report["runtime_only"]["first_stage_seconds"], "prediction_seconds": routed_prediction_seconds, "final_policy_seconds": primary["latency"]["final_policy_mean_ms"] * len(labels) / 1000, "inferences": int(routed_mask.sum()), "inference_mean_ms": 1000 * routed_prediction_seconds / int(routed_mask.sum()), "inference_p95_ms": float(1000 * np.percentile(routed_latency_samples, 95)), "end_to_end_mean_ms": primary["latency"]["end_to_end_mean_ms"], "end_to_end_p95_ms": primary["latency"]["end_to_end_p95_ms"], "peak_rss_bytes": primary["peak_process_rss_bytes"]},
                ))
        report = self._report(full_results, cascade_results, labels, cases, transition_counts, compute_rows)
        self._write(report, transition_counts, compute_rows)
        return report

    def _report(self, full: dict[str, dict[str, object]], cascade: dict[str, dict[str, object]], labels: np.ndarray, cases: dict[str, dict[str, list[dict[str, object]]]], transitions: Counter[tuple[str, str, int]], compute_rows: list[dict[str, object]]) -> dict[str, object]:
        deltas: dict[str, dict[str, object]] = {}
        verdicts: dict[str, str] = {}
        for name in MODELS:
            left, right = full[name]["metrics"], cascade[name]["metrics"]
            matrix_left, matrix_right = left["confusion_matrix"], right["confusion_matrix"]
            deltas[name] = {
                "tp": matrix_right["tp"] - matrix_left["tp"], "fp": matrix_right["fp"] - matrix_left["fp"], "tn": matrix_right["tn"] - matrix_left["tn"], "fn": matrix_right["fn"] - matrix_left["fn"],
                **{key: right[key] - left[key] for key in ("recall", "precision", "f1", "false_positive_rate", "false_negative_rate", "pr_auc", "roc_auc", "alert_volume")},
                "ml_inferences_avoided": full[name]["inferences"] - cascade[name]["inferences"], "ml_inferences_saved_percentage": (full[name]["inferences"] - cascade[name]["inferences"]) / full[name]["inferences"],
                "serving_wall_clock_seconds": cascade[name]["latency"]["end_to_end_mean_ms"] * 100 - full[name]["prediction_seconds"],
                "throughput_difference_transactions_per_second": cascade[name]["latency"]["throughput_transactions_per_second"] - (len(labels) / full[name]["prediction_seconds"]),
                "peak_rss_bytes": cascade[name]["peak_process_rss_bytes"] - full[name]["peak_rss_bytes"],
            }
            verdicts[name] = "QUALITY-COMPUTE TRADE-OFF"
        return {
            "protocol": {"dataset": "data/ibm_aml_data/HI-Small_Trans.csv", "prefix_rows": 500000, "train_rows": 400000, "test_rows": 100000, "seed": 20260804, "full_ml_threshold": 0.50, "alert_mapping": {"full_ml": "probability >= 0.50 => ALERT; otherwise NO ALERT", "cascade": "REVIEW or BLOCK => ALERT; ALLOW or ABSTAIN => NO ALERT"}, "label_boundary": "labels are used only for model fitting and after-decision evaluation"},
            "class_distribution": {"negative": int(len(labels) - labels.sum()), "positive": int(labels.sum())},
            "full_ml": full, "runtime_first_cascade": cascade, "deltas": deltas, "verdicts": verdicts,
            "case_studies": cases,
            "decision_transitions": [{"model": model, "transition": transition, "laundering_label_evaluation_only": label, "count": count} for (model, transition, label), count in sorted(transitions.items())],
            "compute": compute_rows,
        }

    def _write(self, report: dict[str, object], transitions: Counter[tuple[str, str, int]], compute_rows: list[dict[str, object]]) -> None:
        (self.output_dir / "comparison_results.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        with (self.output_dir / "decision_transitions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("model", "transition", "laundering_label_evaluation_only", "count"))
            writer.writerows((model, transition, label, count) for (model, transition, label), count in sorted(transitions.items()))
        with (self.output_dir / "compute_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(compute_rows[0]))
            writer.writeheader()
            writer.writerows(compute_rows)
        self._write_cases(report)
        self._write_chart(report)
        self._write_markdown(report)

    def _write_cases(self, report: dict[str, object]) -> None:
        lines = ["# Paired full-ML vs Runtime-first case studies", "", "`pairwise_outcomes.csv` contains every actual transaction ID. The samples below include complete causal audit traces; labels appear only after the final decision for evaluation.", ""]
        for model, categories in report["case_studies"].items():
            lines.append(f"## {model}")
            for category, values in sorted(categories.items()):
                lines.extend((f"### {category} (first {len(values)} actual cases)", "```json", json.dumps(values, indent=2), "```", ""))
        (self.output_dir / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_chart(self, report: dict[str, object]) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        names = list(MODELS)
        full_tp = [report["full_ml"][name]["metrics"]["confusion_matrix"]["tp"] for name in names]
        cascade_tp = [report["runtime_first_cascade"][name]["metrics"]["confusion_matrix"]["tp"] for name in names]
        saved = [100 * report["deltas"][name]["ml_inferences_saved_percentage"] for name in names]
        index = np.arange(len(names))
        axes[0].bar(index - .18, full_tp, .36, label="Full ML")
        axes[0].bar(index + .18, cascade_tp, .36, label="Runtime-first")
        axes[0].set(xticks=index, xticklabels=names, ylabel="Laundering-labelled events captured", title="Captured out of 28")
        axes[0].legend()
        axes[1].bar(index, saved, color="#0f766e")
        axes[1].set(xticks=index, xticklabels=names, ylabel="ML inferences avoided (%)", title="Routed ML reduction")
        for axis in axes:
            axis.grid(axis="y", alpha=.25)
        figure.tight_layout()
        figure.savefig(self.output_dir / "comparison_chart.png", dpi=160)
        plt.close(figure)

    def _write_markdown(self, report: dict[str, object]) -> None:
        main_rows: list[tuple[object, ...]] = []
        delta_rows: list[tuple[object, ...]] = []
        for name in MODELS:
            for system, result in (("Full ML", report["full_ml"][name]), ("Runtime-first cascade", report["runtime_first_cascade"][name])):
                metric = result["metrics"]
                matrix = metric["confusion_matrix"]
                main_rows.append((name, system, matrix["tp"], matrix["fp"], matrix["fn"], f"{metric['recall']:.6f}", f"{metric['precision']:.6f}", f"{metric['f1']:.6f}", result["inferences"], 0 if system == "Full ML" else report["deltas"][name]["ml_inferences_avoided"]))
            delta = report["deltas"][name]
            delta_rows.append((name, delta["tp"], delta["fp"], delta["fn"], f"{delta['recall']:+.6f}", f"{delta['f1']:+.6f}", delta["ml_inferences_avoided"]))
        verdict_lines = [f"- **{name}: {report['verdicts'][name]}** — cascade saves {report['deltas'][name]['ml_inferences_avoided']:,}/100,000 inferences while its complete TP/FP/FN deltas are reported below." for name in MODELS]
        compute_rows = []
        for name in MODELS:
            full, cascade, delta = report["full_ml"][name], report["runtime_first_cascade"][name], report["deltas"][name]
            compute_rows.append((name, f"{full['prediction_seconds']:.6f}", f"{cascade['latency']['end_to_end_mean_ms'] * 100:.6f}", delta["serving_wall_clock_seconds"], f"{full['inferences'] / full['prediction_seconds']:.0f}", f"{cascade['latency']['throughput_transactions_per_second']:.0f}", delta["peak_rss_bytes"]))
        lines = [
            "# Full-stream ML vs Runtime-first AML cascade", "",
            "## Frozen evaluation contract", "",
            "Full ML maps probability `>= 0.50` to ALERT. The Runtime-first cascade maps REVIEW/BLOCK to ALERT and ALLOW/ABSTAIN to NO ALERT. This mapping is used only for paired evaluation; Runtime operations remain four-state. Thresholds, chronological split, horizon, features, seed, and model configurations are unchanged.", "",
            "## Main comparison", "",
            _table(("Model", "System", "TP", "FP", "FN", "Recall", "Precision", "F1", "ML inference count", "ML saved"), main_rows), "",
            "## Exact cascade-minus-full deltas", "",
            _table(("Model", "TP delta", "FP delta", "FN delta", "Recall delta", "F1 delta", "ML saved"), delta_rows), "",
            "The test horizon contains 28 laundering-labelled events. PR-AUC and ROC-AUC differences, TN/FPR/FNR deltas, inference timing, pairwise outcomes, and all decision transitions are in `comparison_results.json`; no conclusion is based on recall alone.", "",
            "## Verdicts", "",
            *verdict_lines, "",
            "All verdicts are trade-offs, not production claims: each cascade avoids routed-model inferences but changes the error profile substantially. Feature construction still occurs across the whole chronology; inference-count reduction is not presented as total-compute reduction.", "",
            "## Compute comparison", "",
            _table(("Model", "Full serving seconds", "Cascade serving seconds", "Cascade-minus-full seconds", "Full tx/s", "Cascade tx/s", "Cascade-minus-full RSS bytes"), compute_rows), "",
            "Full-stream peak RSS is taken from the already executed frozen full-ML benchmark with the same dataset/configuration; cascade RSS is from the completed Runtime-first run. Full ML raw feature preparation took 6.649 s in this comparison; Runtime-first feature construction took 37.877 s. Therefore avoided inferences are a routed-model reduction, not a claim of lower total feature-computation work.", "",
            "## Paired cases and decision preservation", "",
            "`pairwise_outcomes.csv` contains all 300,000 model/transaction pairings, including transaction IDs and categories. `decision_transitions.csv` splits every Runtime transition by evaluation label. `case_studies.md` provides complete causal traces for up to ten actual cases in each non-empty category. Preliminary REVIEW and BLOCK decisions are preserved by policy; BLOCK is not scored downstream.", "",
        ]
        (self.output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full-stream ML with the pinned Runtime-first cascade")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="artifacts/aml_full_ml_vs_runtime_first")
    args = parser.parse_args()
    report = FullMLVsRuntimeFirst(args.transactions, args.output_dir).run()
    print(json.dumps({"results": str(Path(args.output_dir) / "comparison_results.json"), "report": str(Path(args.output_dir) / "comparison_report.md"), "verdicts": report["verdicts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
