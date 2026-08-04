"""Chronological, label-safe ML comparison for the deterministic AML runtime.

The laundering label is loaded into separate training/evaluation arrays only.
Feature state, runtime transactions, graph history, rule evidence, conflicts,
and hybrid evidence fusion never receive it.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import matplotlib

# The benchmark is non-interactive. Selecting Agg before pyplot avoids macOS
# GUI initialisation (and its NSApplication abort) in the headless runner.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve

from .dataset import AMLSimDataset
from .models import Account, Decision, DecisionRecord, Evidence, Transaction
from .runtime import AMLDecisionRuntime, RuntimeResult, stable_id


TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
FEATURE_NAMES = (
    "log_amount_received", "log_amount_paid", "amount_ratio", "hour", "minute", "day_offset",
    "from_bank_code", "to_bank_code", "same_bank", "receiving_currency_code", "payment_currency_code",
    "payment_format_code", "originator_prior_count", "originator_prior_mean_log_amount",
    "beneficiary_prior_count", "pair_prior_count", "new_beneficiary", "originator_24h_count",
)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:
        # The streaming runtime receives the canonical transaction timestamp
        # emitted by this adapter; feature ingestion still uses the untouched
        # native IBM format above.
        return datetime.fromisoformat(value)


def _canonical_bank(value: str) -> str:
    return str(int(value.strip()))


def _account(bank: str, number: str) -> str:
    return f"{_canonical_bank(bank)}:{number.strip().upper()}"


def _amount(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid amount {value!r}")
    return result


@dataclass(frozen=True)
class BenchmarkConfig:
    train_fraction: float = 0.80
    # 500k is the smallest chronological prefix that retains the fixed 80/20
    # split and the required 100k-event evaluation horizon on this machine.
    input_row_cap: int = 500_000
    evaluation_horizon: int = 100_000
    probability_threshold: float = 0.50
    seed: int = 20260804
    estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    threads: int = 4


@dataclass(frozen=True)
class TestMetadata:
    transaction_id: str
    timestamp: str
    amount: float


class CausalFeatureState:
    """Online aggregates updated only after the current feature vector exists."""

    def __init__(self) -> None:
        self.origin_count: Counter[str] = Counter()
        self.origin_sum: Counter[str] = Counter()
        self.beneficiary_count: Counter[str] = Counter()
        self.pair_count: Counter[tuple[str, str]] = Counter()
        self.origin_24h: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)

    def features(self, timestamp: datetime, originator: str, beneficiary: str, received: float, paid: float, categories: tuple[dict[str, int], ...], raw: list[str], first_day: datetime) -> np.ndarray:
        window = self.origin_24h[originator]
        cutoff = timestamp - timedelta(hours=24)
        while window and window[0][0] < cutoff:
            window.popleft()
        prior_count = self.origin_count[originator]
        prior_sum = self.origin_sum[originator]
        pair = (originator, beneficiary)
        received_log = math.log1p(received)
        return np.asarray((
            received_log, math.log1p(paid), received / paid, timestamp.hour, timestamp.minute,
            (timestamp.date() - first_day.date()).days, categories[0].get(raw[1], -1), categories[1].get(raw[3], -1),
            float(_canonical_bank(raw[1]) == _canonical_bank(raw[3])), categories[2].get(raw[6], -1),
            categories[3].get(raw[8], -1), categories[4].get(raw[9], -1), prior_count,
            math.log1p(prior_sum / prior_count) if prior_count else 0.0, self.beneficiary_count[beneficiary],
            self.pair_count[pair], float(self.pair_count[pair] == 0), len(window),
        ), dtype=np.float32)

    def commit(self, timestamp: datetime, originator: str, beneficiary: str, received: float) -> None:
        self.origin_count[originator] += 1
        self.origin_sum[originator] += received
        self.beneficiary_count[beneficiary] += 1
        self.pair_count[(originator, beneficiary)] += 1
        self.origin_24h[originator].append((timestamp, received))


class _CountedHistory:
    """A length-only historical view used by the fact extractor.

    The Runtime asks only ``bool``/``len`` of streaming history.  Returning a
    repeated tuple allocates one object per prior event and can dominate memory
    on high-frequency account pairs; this view preserves those semantics.
    """

    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __bool__(self) -> bool:
        return self.count > 0


class StreamingRuntimeGraph:
    """Causal graph facade for unchanged FactExtractor/rules/policies.

    It stores only the counts and 24-hour timestamps needed by the existing
    fact extractor, not future transactions or labels.
    """

    def __init__(self, dataset: AMLSimDataset) -> None:
        self.dataset = dataset
        self.by_id: dict[str, Transaction] = {}
        self._pair_count: Counter[tuple[str, str]] = Counter()
        self._origin_24h: dict[str, deque[datetime]] = defaultdict(deque)
        self.sar_accounts: frozenset[str] = frozenset()

    def prior_outbound(self, transaction: Transaction, hours: int | None = None, days: int | None = None) -> _CountedHistory | tuple[()]:
        if hours is None:
            return ()
        current = _parse_timestamp(transaction.timestamp)
        window = self._origin_24h[transaction.originator_account_id]
        cutoff = current - timedelta(hours=hours)
        while window and window[0] < cutoff:
            window.popleft()
        return _CountedHistory(len(window))

    def prior_pair(self, transaction: Transaction, days: int = 365) -> _CountedHistory:
        return _CountedHistory(self._pair_count[(transaction.originator_account_id, transaction.beneficiary_account_id)])

    def connected_to_sar(self, account_id: str, max_hops: int = 2) -> None:
        return None

    def commit(self, transaction: Transaction) -> None:
        timestamp = _parse_timestamp(transaction.timestamp)
        self._pair_count[(transaction.originator_account_id, transaction.beneficiary_account_id)] += 1
        self._origin_24h[transaction.originator_account_id].append(timestamp)


class MLEvidenceProvider:
    """Converts a trained-model probability to immutable risk evidence."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def provide(self, transaction: Transaction, probability: float) -> Evidence:
        confidence = round(float(min(0.999999, max(0.0, probability))), 6)
        return Evidence(
            stable_id("E", transaction.id, "ML", self.model_name, f"{confidence:.6f}"),
            f"ML/{self.model_name}", (), confidence,
            f"{self.model_name} supplied a label-trained probability as external risk evidence.",
            transaction.timestamp, f"ML-{self.model_name}", "risk", "ml_probability", 1.0, None,
        )


def _decision_from_policies(policies: tuple) -> DecisionRecord:
    triggered = tuple(item for item in policies if item.triggered)
    selected = next((decision for decision in (Decision.BLOCK, Decision.REVIEW, Decision.ALLOW, Decision.ABSTAIN) if any(item.outcome == decision for item in triggered)), Decision.ABSTAIN)
    policy_ids = tuple(item.policy_id for item in triggered if item.outcome == selected)
    rationale = next((item.explanation for item in triggered if item.outcome == selected), "No policy was triggered.")
    return DecisionRecord(selected, rationale, policy_ids)


def _hybrid_decision(runtime: AMLDecisionRuntime, base: RuntimeResult, evidence: Evidence) -> DecisionRecord:
    """Run the frozen policy engine over base evidence plus ML evidence."""

    all_evidence = base.evidence + (evidence,)
    conflicts = runtime.conflicts.detect(all_evidence)
    policies = runtime.policies.evaluate(all_evidence, conflicts)
    return _decision_from_policies(policies)


def _binary_decision(decision: Decision) -> int:
    return int(decision in {Decision.REVIEW, Decision.BLOCK})


def _metrics(y_true: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=(0, 1)).ravel()
    return {
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


class AMLMLBenchmark:
    def __init__(self, transaction_path: str | Path, output_directory: str | Path, config: BenchmarkConfig | None = None) -> None:
        self.transaction_path = Path(transaction_path)
        self.output_directory = Path(output_directory)
        self.config = config or BenchmarkConfig()
        self.sorted_path = self.output_directory / "HI-Small_Trans.chronological.csv"

    def _ensure_sorted(self) -> int:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        source_stat = self.transaction_path.stat()
        marker = self.sorted_path.with_suffix(".manifest.json")
        signature = {"source": str(self.transaction_path.resolve()), "size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns, "sort_format": "tail-pipe-v4"}
        if not self.sorted_path.exists() or not marker.exists() or json.loads(marker.read_text()) != signature:
            with self.transaction_path.open("rb") as source, self.sorted_path.open("wb") as destination:
                header = source.readline()
                if not header:
                    raise ValueError("empty transaction source")
                destination.write(header)
                destination.flush()
                environment = {**os.environ, "LC_ALL": "C"}
                # `tail` owns a separate descriptor, avoiding buffered Python
                # input being passed to sort at a stale byte position.
                tail = subprocess.Popen(["tail", "-n", "+2", str(self.transaction_path)], stdout=subprocess.PIPE)
                assert tail.stdout is not None
                try:
                    subprocess.run(["sort", "-s", "-t,", "-k1,1"], stdin=tail.stdout, stdout=destination, check=True, env=environment)
                finally:
                    tail.stdout.close()
                    if tail.wait() != 0:
                        raise RuntimeError("tail failed while preparing chronological input")
            marker.write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in handle) - 1

    def _category_maps(self, train_rows: int) -> tuple[dict[str, int], ...]:
        values = [set() for _ in range(5)]
        columns = (1, 3, 6, 8, 9)
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            schema = next(reader)
            if schema != ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]:
                raise ValueError(f"unexpected IBM AML-Data schema: {schema!r}")
            for index, row in enumerate(reader):
                if index == train_rows:
                    break
                for position, column in enumerate(columns):
                    values[position].add(row[column])
        return tuple({value: index for index, value in enumerate(sorted(group))} for group in values)

    def _build_feature_sets(self, total_rows: int, train_rows: int, test_rows: int, categories: tuple[dict[str, int], ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[TestMetadata]]:
        train_x = np.empty((train_rows, len(FEATURE_NAMES)), dtype=np.float32)
        test_x = np.empty((test_rows, len(FEATURE_NAMES)), dtype=np.float32)
        train_y = np.empty(train_rows, dtype=np.uint8)
        test_y = np.empty(test_rows, dtype=np.uint8)
        metadata: list[TestMetadata] = []
        state = CausalFeatureState()
        first_day: datetime | None = None
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index >= train_rows + test_rows:
                    break
                timestamp = _parse_timestamp(row[0])
                first_day = timestamp if first_day is None else first_day
                originator = _account(row[1], row[2])
                beneficiary = _account(row[3], row[4])
                received, paid = _amount(row[5]), _amount(row[7])
                feature = state.features(timestamp, originator, beneficiary, received, paid, categories, row, first_day)
                if index < train_rows:
                    train_x[index] = feature
                    train_y[index] = int(row[10] == "1")
                else:
                    test_index = index - train_rows
                    test_x[test_index] = feature
                    test_y[test_index] = int(row[10] == "1")
                    metadata.append(TestMetadata(f"IBM-ML-{index + 2:08d}", timestamp.isoformat(timespec="seconds"), received))
                state.commit(timestamp, originator, beneficiary, received)
        if len(metadata) != test_rows:
            raise ValueError(f"expected {test_rows} test rows, found {len(metadata)}")
        return train_x, train_y, test_x, test_y, metadata

    def _models(self, scale_pos_weight: float) -> dict[str, object]:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        common = {"n_estimators": self.config.estimators, "max_depth": self.config.max_depth, "learning_rate": self.config.learning_rate, "random_state": self.config.seed}
        return {
            "XGBoost": XGBClassifier(**common, objective="binary:logistic", eval_metric="logloss", tree_method="hist", subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, n_jobs=self.config.threads),
            "LightGBM": LGBMClassifier(**common, objective="binary", subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, n_jobs=self.config.threads, verbosity=-1),
            "CatBoost": CatBoostClassifier(iterations=self.config.estimators, depth=self.config.max_depth, learning_rate=self.config.learning_rate, loss_function="Logloss", eval_metric="AUC", random_seed=self.config.seed, scale_pos_weight=scale_pos_weight, thread_count=self.config.threads, verbose=False, allow_writing_files=False),
        }

    def _runtime_results(
        self,
        train_rows: int,
        test_rows: int,
        metadata: list[TestMetadata],
        model_probabilities: dict[str, np.ndarray],
    ) -> tuple[list[Decision], np.ndarray, float, dict[str, list[Decision]], dict[str, float]]:
        accounts: dict[str, Account] = {}
        source_stat = self.transaction_path.stat()
        fingerprint = hashlib.sha256(f"{self.transaction_path.resolve()}:{source_stat.st_size}:{source_stat.st_mtime_ns}".encode("utf-8")).hexdigest()
        dataset = AMLSimDataset((), accounts, {}, str(self.transaction_path), fingerprint)
        audit_dir = self.output_directory / "runtime_audits"
        runtime = AMLDecisionRuntime(dataset, audit_dir)
        graph = StreamingRuntimeGraph(dataset)
        runtime.graph = graph
        scores = np.empty(test_rows, dtype=np.float64)
        decisions: list[Decision] = []
        hybrid_decisions: dict[str, list[Decision]] = {name: [] for name in model_probabilities}
        hybrid_seconds: dict[str, float] = {name: 0.0 for name in model_probabilities}
        providers = {name: MLEvidenceProvider(name) for name in model_probabilities}
        fusion_runtime = AMLDecisionRuntime(
            AMLSimDataset((), {}, {}, "hybrid-policy", "hybrid-policy"),
            self.output_directory / "hybrid_policy_audits",
        )
        started = time.perf_counter()
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index >= train_rows + test_rows:
                    break
                timestamp = _parse_timestamp(row[0])
                originator, beneficiary = _account(row[1], row[2]), _account(row[3], row[4])
                transaction = Transaction(
                    id=f"IBM-ML-{index + 2:08d}", timestamp=timestamp.isoformat(timespec="seconds"),
                    originator_account_id=originator, beneficiary_account_id=beneficiary, amount=_amount(row[5]),
                    currency=row[6], country_id="", payment_type=row[9],
                    metadata={"from_bank": _canonical_bank(row[1]), "to_bank": _canonical_bank(row[3]), "amount_paid": row[7], "payment_currency": row[8]},
                )
                accounts.setdefault(originator, Account(id=originator, bank_id=_canonical_bank(row[1])))
                accounts.setdefault(beneficiary, Account(id=beneficiary, bank_id=_canonical_bank(row[3])))
                if index < train_rows:
                    graph.commit(transaction)
                    continue
                test_index = index - train_rows
                if transaction.id != metadata[test_index].transaction_id:
                    raise AssertionError("feature and runtime test streams diverged")
                graph.by_id[transaction.id] = transaction
                result = runtime.execute(transaction.id)
                scores[test_index] = float(result.policies[0].metrics["effective_risk"])
                decisions.append(result.decision.decision)
                for name, probability in model_probabilities.items():
                    fusion_started = time.perf_counter()
                    hybrid_decision = _hybrid_decision(
                        fusion_runtime, result, providers[name].provide(transaction, float(probability[test_index]))
                    )
                    hybrid_seconds[name] += time.perf_counter() - fusion_started
                    hybrid_decisions[name].append(hybrid_decision.decision)
                graph.commit(transaction)
                del graph.by_id[transaction.id]
        return decisions, scores, time.perf_counter() - started, hybrid_decisions, hybrid_seconds

    @staticmethod
    def _case(metadata: TestMetadata, label: int, model_probability: float, model_prediction: int, runtime_decision: Decision, hybrid_decision: Decision) -> dict[str, object]:
        return {
            "transaction_id": metadata.transaction_id,
            "timestamp": metadata.timestamp,
            "amount_received": metadata.amount,
            "laundering_label_evaluation_only": int(label),
            "ml_probability": round(float(model_probability), 6),
            "ml_prediction": int(model_prediction),
            "runtime_decision": runtime_decision.value,
            "hybrid_decision": hybrid_decision.value,
        }

    def _comparison_cases(self, metadata: list[TestMetadata], labels: np.ndarray, probabilities: np.ndarray, runtime_predictions: np.ndarray, hybrid_predictions: np.ndarray, runtime_decisions: list[Decision], hybrid_decisions: list[Decision]) -> dict[str, list[dict[str, object]]]:
        ml_predictions = (probabilities >= self.config.probability_threshold).astype(np.uint8)
        result: dict[str, list[dict[str, object]]] = {"ml_wrong_runtime_correct": [], "runtime_wrong_ml_correct": [], "ml_false_positives": [], "runtime_false_positives": [], "ml_false_negatives": [], "runtime_false_negatives": []}
        conditions: dict[str, Callable[[int], bool]] = {
            "ml_wrong_runtime_correct": lambda i: ml_predictions[i] != labels[i] and runtime_predictions[i] == labels[i],
            "runtime_wrong_ml_correct": lambda i: runtime_predictions[i] != labels[i] and ml_predictions[i] == labels[i],
            "ml_false_positives": lambda i: labels[i] == 0 and ml_predictions[i] == 1,
            "runtime_false_positives": lambda i: labels[i] == 0 and runtime_predictions[i] == 1,
            "ml_false_negatives": lambda i: labels[i] == 1 and ml_predictions[i] == 0,
            "runtime_false_negatives": lambda i: labels[i] == 1 and runtime_predictions[i] == 0,
        }
        for index in range(len(labels)):
            for name, condition in conditions.items():
                if len(result[name]) < 10 and condition(index):
                    result[name].append(self._case(metadata[index], int(labels[index]), float(probabilities[index]), int(ml_predictions[index]), runtime_decisions[index], hybrid_decisions[index]))
        return result

    def _write_curves(self, labels: np.ndarray, scores: dict[str, np.ndarray], output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, values in scores.items():
            false_positive, true_positive, _ = roc_curve(labels, values)
            precision, recall, _ = precision_recall_curve(labels, values)
            axes[0].plot(false_positive, true_positive, label=f"{name} (AUC={roc_auc_score(labels, values):.3f})")
            axes[1].plot(recall, precision, label=f"{name} (AP={average_precision_score(labels, values):.3f})")
        axes[0].plot((0, 1), (0, 1), "k--", linewidth=0.8)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plt.close(figure)

    def run(self) -> dict[str, object]:
        source_rows = self._ensure_sorted()
        total_rows = min(source_rows, self.config.input_row_cap)
        train_rows = int(total_rows * self.config.train_fraction)
        test_rows = min(self.config.evaluation_horizon, total_rows - train_rows)
        if test_rows != self.config.evaluation_horizon:
            raise ValueError(
                f"input_row_cap={self.config.input_row_cap} cannot retain the required "
                f"evaluation horizon of {self.config.evaluation_horizon} rows under the "
                f"fixed {self.config.train_fraction:.0%}/{1 - self.config.train_fraction:.0%} split"
            )
        categories = self._category_maps(train_rows)
        feature_started = time.perf_counter()
        train_x, train_y, test_x, test_y, metadata = self._build_feature_sets(total_rows, train_rows, test_rows, categories)
        feature_seconds = time.perf_counter() - feature_started
        positives = int(train_y.sum())
        scale_pos_weight = (len(train_y) - positives) / positives
        models = self._models(scale_pos_weight)
        model_probabilities: dict[str, np.ndarray] = {}
        model_results: dict[str, dict[str, object]] = {}
        for name, model in list(models.items()):
            started = time.perf_counter()
            model.fit(train_x, train_y)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            probability = model.predict_proba(test_x)[:, 1]
            predict_seconds = time.perf_counter() - started
            predicted = (probability >= self.config.probability_threshold).astype(np.uint8)
            model_probabilities[name] = probability
            model_results[name] = {
                "metrics": _metrics(test_y, probability, predicted),
                "training_seconds": train_seconds,
                "prediction_latency_ms": 1000 * predict_seconds / len(test_y),
                "peak_process_rss_bytes": _peak_rss_bytes(),
            }
            # The benchmark needs probabilities, not fitted estimators, beyond
            # this point. Releasing each model preserves the frozen experiment
            # while preventing three native booster allocations from surviving
            # into the runtime and hybrid stages.
            del models[name]
            del model
            gc.collect()
        del train_x, test_x
        gc.collect()
        runtime_decisions, runtime_scores, runtime_seconds, hybrid_decision_values, hybrid_seconds = self._runtime_results(
            train_rows, test_rows, metadata, model_probabilities
        )
        runtime_predicted = np.asarray([_binary_decision(decision) for decision in runtime_decisions], dtype=np.uint8)
        runtime_result = {
            "metrics": _metrics(test_y, runtime_scores, runtime_predicted),
            "decision_distribution": dict(Counter(item.value for item in runtime_decisions)),
            "decision_latency_ms": 1000 * runtime_seconds / len(test_y),
            "peak_process_rss_bytes": _peak_rss_bytes(),
        }
        hybrid_results: dict[str, dict[str, object]] = {}
        curves: dict[str, np.ndarray] = {"Decision Runtime": runtime_scores}
        for name, probability in model_probabilities.items():
            decision_values = hybrid_decision_values[name]
            prediction = np.asarray([_binary_decision(item) for item in decision_values], dtype=np.uint8)
            score = np.maximum(runtime_scores, probability)
            hybrid_results[f"{name} + Decision Runtime"] = {
                "metrics": _metrics(test_y, score, prediction),
                "decision_distribution": dict(Counter(item.value for item in decision_values)),
                "decision_latency_ms": model_results[name]["prediction_latency_ms"] + 1000 * hybrid_seconds[name] / len(test_y),
                "peak_process_rss_bytes": _peak_rss_bytes(),
                "cases": self._comparison_cases(metadata, test_y, probability, runtime_predicted, prediction, runtime_decisions, decision_values),
            }
            curves[f"{name} + Runtime"] = score
            curves[name] = probability
        curve_path = self.output_directory / "roc_pr_curves.png"
        self._write_curves(test_y, curves, curve_path)
        report = {
            "protocol": {
                "dataset": str(self.transaction_path), "source_total_rows": source_rows, "total_rows": total_rows, "input_row_cap": self.config.input_row_cap, "chronological_train_rows": train_rows,
                "chronological_test_horizon_rows": test_rows, "unscored_post_split_rows": total_rows - train_rows - test_rows,
                "train_fraction": self.config.train_fraction, "test_horizon_policy": "first fixed post-split chronological horizon; bounded to retain exact per-transaction runtime audits within local disk capacity",
                "label_boundary": "labels are separate model training/evaluation targets and are never passed to feature state, runtime transaction/graph/facts/evidence/policies, or base-runtime audits",
            },
            "feature_engineering": {"feature_names": FEATURE_NAMES, "causal_history": "online originator, beneficiary, pair, and trailing-24h aggregates committed after each feature vector", "categorical_encoding": "ordinal maps fit on chronological train partition only; unseen test categories map to -1"},
            "hyperparameters": {**self.config.__dict__, "scale_pos_weight": scale_pos_weight, "classification_threshold": self.config.probability_threshold},
            "class_distribution": {"train": {"negative": int(len(train_y) - train_y.sum()), "positive": positives}, "test": {"negative": int(len(test_y) - test_y.sum()), "positive": int(test_y.sum())}},
            "model_results": model_results,
            "runtime_result": runtime_result,
            "hybrid_results": hybrid_results,
            "feature_engineering_seconds": feature_seconds,
            "curve_path": str(curve_path),
        }
        report_path = self.output_directory / "comparison_results.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        self._write_report(report, self.output_directory / "comparison_report.md")
        return report

    @staticmethod
    def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])

    def _write_report(self, report: dict[str, object], path: Path) -> None:
        protocol = report["protocol"]
        features = report["feature_engineering"]
        parameters = report["hyperparameters"]
        class_distribution = report["class_distribution"]
        models = report["model_results"]
        runtime = report["runtime_result"]
        hybrids = report["hybrid_results"]
        assert isinstance(protocol, dict) and isinstance(features, dict) and isinstance(parameters, dict)
        assert isinstance(class_distribution, dict) and isinstance(models, dict) and isinstance(runtime, dict) and isinstance(hybrids, dict)
        rows: list[tuple[object, ...]] = []
        for name, result in models.items():
            assert isinstance(result, dict)
            metric = result["metrics"]
            assert isinstance(metric, dict)
            rows.append((name, *(f"{metric[key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")), f"{result['prediction_latency_ms']:.4f}", result["peak_process_rss_bytes"]))
        runtime_metric = runtime["metrics"]
        assert isinstance(runtime_metric, dict)
        rows.append(("Decision Runtime", *(f"{runtime_metric[key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")), f"{runtime['decision_latency_ms']:.4f}", runtime["peak_process_rss_bytes"]))
        for name, result in hybrids.items():
            assert isinstance(result, dict)
            metric = result["metrics"]
            assert isinstance(metric, dict)
            rows.append((name, *(f"{metric[key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")), f"{result['decision_latency_ms']:.4f}", result["peak_process_rss_bytes"]))
        confusion_rows: list[tuple[object, ...]] = []
        for name, result in list(models.items()) + [("Decision Runtime", runtime)] + list(hybrids.items()):
            assert isinstance(result, dict)
            matrix = result["metrics"]["confusion_matrix"]
            confusion_rows.append((name, matrix["tn"], matrix["fp"], matrix["fn"], matrix["tp"]))
        decision_rows = [("Decision Runtime", runtime["decision_distribution"])] + [(name, item["decision_distribution"]) for name, item in hybrids.items()]
        hyperparameter_rows = [(key, value) for key, value in parameters.items()]
        sample_cases = []
        for name, result in hybrids.items():
            assert isinstance(result, dict)
            cases = result["cases"]
            assert isinstance(cases, dict)
            sample_cases.append(f"### {name}")
            for category in ("ml_wrong_runtime_correct", "runtime_wrong_ml_correct", "ml_false_positives", "runtime_false_positives", "ml_false_negatives", "runtime_false_negatives"):
                sample_cases.extend((f"#### {category}", "```json", json.dumps(cases[category], indent=2), "```"))
        lines = [
            "# AML Decision Runtime vs Classical ML", "",
            "## Methodology", "",
            "This is a fixed, chronological comparison. Classical models use the laundering label only as their training target and for final evaluation. The deterministic runtime does not read labels. No model or runtime threshold was tuned against holdout labels.", "",
            "## Train/test split", "",
            f"The IBM AML-Data source has {protocol['source_total_rows']:,} rows; this run fixes its first {protocol['total_rows']:,} chronological events as the predeclared practical input cap. The first {protocol['chronological_train_rows']:,} events ({protocol['train_fraction']:.0%}) form training; the next {protocol['chronological_test_horizon_rows']:,} events form the fixed post-split evaluation horizon. {protocol['unscored_post_split_rows']:,} capped-input events are not scored because retaining exact JSON audits for every runtime decision exceeds local storage capacity. This limitation is fixed before fitting and applies to every method.", "",
            self._table(("Partition", "Negative", "Positive"), [("Train", class_distribution["train"]["negative"], class_distribution["train"]["positive"]), ("Evaluation horizon", class_distribution["test"]["negative"], class_distribution["test"]["positive"])]), "",
            "## Feature engineering", "",
            f"Features: `{', '.join(features['feature_names'])}`.", "",
            f"Causality: {features['causal_history']}. Categorical encoding: {features['categorical_encoding']}. The label boundary is: {protocol['label_boundary']}", "",
            "## Hyperparameters", "",
            self._table(("Parameter", "Value"), hyperparameter_rows), "",
            "## Benchmark table", "",
            self._table(("Method", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR", "Decision/predict ms", "Peak process RSS bytes"), rows), "",
            "## Confusion matrices", "",
            self._table(("Method", "TN", "FP", "FN", "TP"), confusion_rows), "",
            "## ROC and PR curves", "",
            "![ROC and precision-recall curves](roc_pr_curves.png)", "",
            "Runtime and hybrid ROC/PR scores use the runtime effective-risk score and `max(runtime effective risk, ML probability)`, respectively. Categorical runtime decisions are evaluated separately in the table above.", "",
            "## Runtime analysis", "",
            self._table(("System", "Decision distribution"), decision_rows), "",
            "The base runtime is the frozen v0.2 implementation. Hybrid systems add an immutable ML probability Evidence object to base evidence, then invoke the unchanged conflict and policy engines. ML evidence never creates a BLOCK by itself; the policy engine remains the final decision selector.", "",
            "## Error and correction cases", "",
            *sample_cases,
            "## Discussion", "",
            "This is an observational benchmark, not a claim that REVIEW/BLOCK equals the laundering label. Reported correction cases mean only that one system's binary alert mapping matched the evaluation label when the other did not. The evaluation horizon, class imbalance, missing KYC/SAR/jurisdiction inputs, and lack of probability calibration constrain external validity. If an ML model outperforms the runtime, that is reported directly by the metrics; if the hybrid changes errors, the listed cases identify the exact policy/evidence context.", "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aml_ml_benchmark"))
    parser.add_argument("--evaluation-horizon", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = BenchmarkConfig(evaluation_horizon=args.evaluation_horizon, threads=args.threads)
    report = AMLMLBenchmark(args.transactions, args.output_dir, config).run()
    print(json.dumps({"report_path": str(args.output_dir / "comparison_report.md"), "results_path": str(args.output_dir / "comparison_results.json"), "model_results": report["model_results"], "runtime_result": report["runtime_result"], "hybrid_results": report["hybrid_results"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
