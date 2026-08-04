"""Did the Semantic Runtime create information, or re-encode the raw features?

One question, three feature spaces, everything else frozen.

Protocol — identical to the window-study run that produced
``ml_only/CatBoost`` recall 0.901 / FP 6,128 at window F:

* priming            rows 0 .. 4,977,237 (all available history, 9 d 11 h)
* ML train partition rows 4,577,237 .. 4,977,237 (400,000 events, 436 positives)
* evaluation         rows 4,977,237 .. 5,077,237 (100,000 events, 253 positives)
* CatBoost           iterations 200, depth 6, learning_rate 0.05, Logloss,
                     random_seed 20260804, scale_pos_weight from the partition,
                     thread_count 4 — taken from ``_models`` unchanged
* alert mapping      probability >= 0.50

Experiment B must reproduce recall 0.901 / FP 6,128 exactly.  That is the
harness's own correctness check and it is asserted in the report.

Nothing in the Runtime, Semantic Context Layer, Behaviour Layer, policies,
routing, thresholds or CatBoost hyperparameters is modified, and no state is
built differently: the priming path is the one proven equivalent to full
evaluation in ``tests/test_window_study.py``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from catboost import Pool
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

from .behaviour import SEMANTIC_FEATURE_NAMES, semantic_feature_vector
from .ml_benchmark import FEATURE_NAMES as RAW_FEATURE_NAMES
from .ml_benchmark import CausalFeatureState, _account, _amount
from .runtime_first_cascade import CascadeConfig, _models
from .semantic import ONTOLOGY_HASH, EntityResolver
from .semantic_benchmark import ML_STANDALONE_THRESHOLD
from .behaviour import BEHAVIOUR_ONTOLOGY_HASH
from .window_study import (
    DENSE_PERIOD_END,
    EVALUATION_ROWS,
    EVALUATION_START,
    ML_TRAIN_CAP,
    TIMESTAMP_FORMAT,
    WindowStudy,
    _event,
)

EXPERIMENT_VERSION = "aml-semantic-vs-raw/1.0"
TRAIN_START = EVALUATION_START - ML_TRAIN_CAP
COMBINED_FEATURE_NAMES = tuple(RAW_FEATURE_NAMES) + tuple(SEMANTIC_FEATURE_NAMES)

#: Which vocabulary each feature belongs to.  ``sem_`` features are the Semantic
#: Context Layer's objects, which the brief calls Context Objects.
GROUPS: dict[str, str] = {}
for _name in RAW_FEATURE_NAMES:
    GROUPS[_name] = "raw"
for _name in SEMANTIC_FEATURE_NAMES:
    if _name.startswith("sem_"):
        GROUPS[_name] = "context"
    elif _name.startswith("beh_"):
        GROUPS[_name] = "behaviour"
    elif _name.startswith("scn_"):
        GROUPS[_name] = "scenario"
    elif _name.startswith("role_"):
        GROUPS[_name] = "role"
    elif _name.startswith("lifecycle_"):
        GROUPS[_name] = "lifecycle"
    elif _name.startswith("evidence_"):
        GROUPS[_name] = "evidence"
    else:  # pragma: no cover - the vocabulary is closed
        raise ValueError(f"ungrouped semantic feature {_name}")

SEMANTIC_GROUPS = ("context", "behaviour", "scenario", "role", "lifecycle", "evidence")


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _metrics(labels: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "alert_rate": float(predicted.mean()),
        "alert_volume": int(predicted.sum()),
    }


class SemanticVsRaw:
    def __init__(self, transactions: str | Path, accounts: str | Path, output_dir: str | Path, threads: int = 4) -> None:
        self.transactions = Path(transactions)
        self.accounts = Path(accounts)
        self.output_dir = Path(output_dir)
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")
        self.config = CascadeConfig(threads=threads)

    # -- input ------------------------------------------------------------
    def _rows(self, start: int, end: int):
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index < start:
                    continue
                if index >= end:
                    return
                yield index, row

    def _category_maps(self) -> tuple[dict[str, int], ...]:
        """Ordinal maps over every pre-evaluation row; unseen test values map to -1.

        Fitted on exactly the history the semantic layer also folded in, so
        neither feature space is given an advantage in coverage.
        """
        values = [set() for _ in range(5)]
        columns = (1, 3, 6, 8, 9)
        for _index, row in self._rows(0, EVALUATION_START):
            for position, column in enumerate(columns):
                values[position].add(row[column])
        return tuple({value: order for order, value in enumerate(sorted(group))} for group in values)

    # -- one pass, both feature spaces --------------------------------------
    def _build(self) -> dict[str, object]:
        resolver = EntityResolver().load(self.accounts)
        categories = self._category_maps()
        runtime = WindowStudy._runtimes(resolver)
        raw_state = CausalFeatureState()

        raw_train = np.zeros((ML_TRAIN_CAP, len(RAW_FEATURE_NAMES)), dtype=np.float32)
        raw_test = np.zeros((EVALUATION_ROWS, len(RAW_FEATURE_NAMES)), dtype=np.float32)
        semantic_train = np.zeros((ML_TRAIN_CAP, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        semantic_test = np.zeros((EVALUATION_ROWS, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        train_y = np.zeros(ML_TRAIN_CAP, dtype=np.uint8)
        test_y = np.zeros(EVALUATION_ROWS, dtype=np.uint8)
        scratch = np.zeros(len(SEMANTIC_FEATURE_NAMES), dtype=np.float32)

        first_day: datetime | None = None
        raw_seconds = semantic_seconds = 0.0
        started_all = time.perf_counter()
        for index, row in self._rows(0, DENSE_PERIOD_END):
            timestamp = datetime.strptime(row[0], TIMESTAMP_FORMAT)
            if first_day is None:
                first_day = timestamp
            originator, beneficiary = _account(row[1], row[2]), _account(row[3], row[4])
            received, paid = _amount(row[5]), _amount(row[7])
            transaction, minute = _event(index, row)

            # `features` is called for every row, exactly as the original raw
            # pipeline does: it is what prunes each account's trailing 24-hour
            # deque.  Skipping it on the priming rows would leave those deques
            # unpruned for 4.5M events.  The vector is simply discarded until
            # the training partition begins.
            started = time.perf_counter()
            raw_vector = raw_state.features(timestamp, originator, beneficiary, received, paid,
                                            categories, row, first_day)
            raw_seconds += time.perf_counter() - started

            if index >= TRAIN_START:
                started = time.perf_counter()
                result = runtime.evaluate(transaction, resolver.resolve(originator), minute, index)
                metrics = result.policies[0].metrics
                semantic_vector = semantic_feature_vector(
                    {item.type: item.confidence for item in result.semantic.context.objects},
                    result.behaviour, result.evidence, len(result.conflicts),
                    int(metrics["independent_unqualified_topics"]), float(metrics["effective_risk"]), scratch,
                )
                semantic_seconds += time.perf_counter() - started

                label = int(row[10] == "1")
                if index < EVALUATION_START:
                    position = index - TRAIN_START
                    raw_train[position] = raw_vector
                    semantic_train[position] = semantic_vector
                    train_y[position] = label
                else:
                    position = index - EVALUATION_START
                    raw_test[position] = raw_vector
                    semantic_test[position] = semantic_vector
                    test_y[position] = label
                runtime.commit(transaction, result, minute)
            else:
                started = time.perf_counter()
                WindowStudy._prime(runtime, resolver, transaction, minute, index)
                semantic_seconds += time.perf_counter() - started
            raw_state.commit(timestamp, originator, beneficiary, received)

        total_seconds = time.perf_counter() - started_all
        del runtime, raw_state, resolver
        gc.collect()
        return {
            "raw_train": raw_train, "raw_test": raw_test,
            "semantic_train": semantic_train, "semantic_test": semantic_test,
            "train_y": train_y, "test_y": test_y,
            "raw_feature_seconds": raw_seconds, "semantic_feature_seconds": semantic_seconds,
            "total_feature_seconds": total_seconds,
        }

    # -- model ---------------------------------------------------------------
    def _catboost(self, scale: float):
        """The frozen CatBoost, taken from `_models` with nothing overridden."""
        return _models(self.config, scale)["CatBoost"]

    def _fit_predict(self, name: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                     test_y: np.ndarray, feature_names: tuple[str, ...], keep_model: bool):
        scale = (len(train_y) - int(train_y.sum())) / int(train_y.sum())
        model = self._catboost(scale)
        started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probability = model.predict_proba(test_x)[:, 1]
        predict_seconds = time.perf_counter() - started
        predicted = (probability >= ML_STANDALONE_THRESHOLD).astype(np.uint8)
        record = {
            "features": name, "feature_count": len(feature_names),
            "metrics": _metrics(test_y, probability, predicted),
            "training_seconds": train_seconds,
            "prediction_seconds": predict_seconds,
            "inference_latency_ms": 1000 * predict_seconds / len(test_y),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        if not keep_model:
            del model
            gc.collect()
            return record, probability, None
        return record, probability, model

    @staticmethod
    def _importance(model, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
        weights = model.get_feature_importance()
        return sorted(zip(feature_names, (float(value) for value in weights)), key=lambda item: -item[1])

    @staticmethod
    def _shap(model, test_x: np.ndarray, test_y: np.ndarray, feature_names: tuple[str, ...]) -> tuple[list[tuple[str, float]], float]:
        """Exact TreeSHAP, computed by CatBoost itself (the `shap` package is absent).

        Returns mean(|SHAP|) per feature over the whole evaluation set, which is
        the standard global attribution summary.
        """
        started = time.perf_counter()
        values = model.get_feature_importance(Pool(test_x, test_y), type="ShapValues")
        seconds = time.perf_counter() - started
        contributions = np.abs(values[:, :-1]).mean(axis=0)
        ranked = sorted(zip(feature_names, (float(value) for value in contributions)), key=lambda item: -item[1])
        return ranked, seconds

    # -- the experiment -------------------------------------------------------
    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = self._build()
        train_y, test_y = data["train_y"], data["test_y"]
        raw_train, raw_test = data["raw_train"], data["raw_test"]
        semantic_train, semantic_test = data["semantic_train"], data["semantic_test"]
        combined_train = np.hstack((raw_train, semantic_train))
        combined_test = np.hstack((raw_test, semantic_test))

        arms: dict[str, dict] = {}
        probabilities: dict[str, np.ndarray] = {}
        models: dict[str, object] = {}
        for name, train_x, test_x, names, keep in (
            ("raw", raw_train, raw_test, tuple(RAW_FEATURE_NAMES), True),
            ("semantic", semantic_train, semantic_test, tuple(SEMANTIC_FEATURE_NAMES), True),
            ("raw_plus_semantic", combined_train, combined_test, COMBINED_FEATURE_NAMES, True),
        ):
            record, probability, model = self._fit_predict(name, train_x, train_y, test_x, test_y, names, keep)
            arms[name] = record
            probabilities[name] = probability
            models[name] = model
            print(json.dumps({"arm": name, **record["metrics"]}), flush=True)

        importances = {
            "semantic": self._importance(models["semantic"], tuple(SEMANTIC_FEATURE_NAMES)),
            "raw_plus_semantic": self._importance(models["raw_plus_semantic"], COMBINED_FEATURE_NAMES),
            "raw": self._importance(models["raw"], tuple(RAW_FEATURE_NAMES)),
        }
        shap_values: dict[str, list[tuple[str, float]]] = {}
        shap_seconds: dict[str, float] = {}
        for name, test_x, names in (
            ("semantic", semantic_test, tuple(SEMANTIC_FEATURE_NAMES)),
            ("raw_plus_semantic", combined_test, COMBINED_FEATURE_NAMES),
            ("raw", raw_test, tuple(RAW_FEATURE_NAMES)),
        ):
            shap_values[name], shap_seconds[name] = self._shap(models[name], test_x, test_y, names)
            print(json.dumps({"shap": name, "seconds": round(shap_seconds[name], 1)}), flush=True)

        ablations = self._ablations(raw_train, semantic_train, raw_test, semantic_test, train_y, test_y)
        correlation = self._correlation(raw_test, semantic_test)
        for model in models.values():
            del model
        models.clear()
        gc.collect()

        report = self._assemble(arms, importances, shap_values, shap_seconds, ablations, correlation, data, test_y)
        self._write(report, importances, shap_values, ablations, correlation, probabilities, test_y)
        return report

    def _ablations(self, raw_train, semantic_train, raw_test, semantic_test, train_y, test_y) -> list[dict]:
        """Drop one vocabulary at a time from Raw+Semantic, everything else frozen."""
        semantic_index = {name: position for position, name in enumerate(SEMANTIC_FEATURE_NAMES)}
        results: list[dict] = []
        for dropped in ("raw",) + SEMANTIC_GROUPS:
            if dropped == "raw":
                train_x, test_x = semantic_train, semantic_test
                names = tuple(SEMANTIC_FEATURE_NAMES)
            else:
                keep = [semantic_index[name] for name in SEMANTIC_FEATURE_NAMES if GROUPS[name] != dropped]
                train_x = np.hstack((raw_train, semantic_train[:, keep]))
                test_x = np.hstack((raw_test, semantic_test[:, keep]))
                names = tuple(RAW_FEATURE_NAMES) + tuple(name for name in SEMANTIC_FEATURE_NAMES if GROUPS[name] != dropped)
            record, _probability, _model = self._fit_predict(f"raw_plus_semantic_minus_{dropped}", train_x, train_y,
                                                             test_x, test_y, names, keep_model=False)
            record["dropped_group"] = dropped
            record["dropped_feature_count"] = len(COMBINED_FEATURE_NAMES) - len(names)
            results.append(record)
            print(json.dumps({"ablation": f"-{dropped}", **record["metrics"]}), flush=True)
            del train_x, test_x
            gc.collect()
        return results

    @staticmethod
    def _correlation(raw_test: np.ndarray, semantic_test: np.ndarray) -> list[dict]:
        """Is each semantic feature a re-encoding of some raw feature?"""
        rows: list[dict] = []
        raw = raw_test.astype(np.float64)
        semantic = semantic_test.astype(np.float64)
        raw_std = raw.std(axis=0)
        semantic_std = semantic.std(axis=0)
        raw_centered = raw - raw.mean(axis=0)
        semantic_centered = semantic - semantic.mean(axis=0)
        for position, name in enumerate(SEMANTIC_FEATURE_NAMES):
            if semantic_std[position] == 0.0:
                rows.append({"semantic_feature": name, "group": GROUPS[name], "constant": 1,
                             "max_abs_pearson_r": 0.0, "closest_raw_feature": "", "r": 0.0})
                continue
            correlations = []
            for raw_position, raw_name in enumerate(RAW_FEATURE_NAMES):
                if raw_std[raw_position] == 0.0:
                    correlations.append((raw_name, 0.0))
                    continue
                value = float(np.dot(semantic_centered[:, position], raw_centered[:, raw_position])
                              / (len(raw) * semantic_std[position] * raw_std[raw_position]))
                correlations.append((raw_name, value))
            closest, best = max(correlations, key=lambda item: abs(item[1]))
            rows.append({"semantic_feature": name, "group": GROUPS[name], "constant": 0,
                         "max_abs_pearson_r": abs(best), "closest_raw_feature": closest, "r": best})
        return rows

    # -- analysis --------------------------------------------------------------
    @staticmethod
    def _group_totals(ranked: list[tuple[str, float]]) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for name, value in ranked:
            totals[GROUPS[name]] += value
        overall = sum(totals.values()) or 1.0
        result = {key: {"absolute": value, "share": value / overall} for key, value in sorted(totals.items())}
        semantic_total = sum(totals[key] for key in SEMANTIC_GROUPS)
        result["_semantic_total"] = {"absolute": semantic_total, "share": semantic_total / overall}
        result["_raw_total"] = {"absolute": totals.get("raw", 0.0), "share": totals.get("raw", 0.0) / overall}
        return result

    def _assemble(self, arms, importances, shap_values, shap_seconds, ablations, correlation, data, test_y) -> dict:
        raw_rank_alone = {name: position for position, (name, _value) in enumerate(importances["raw"], start=1)}
        raw_rank_combined = {name: position for position, (name, _value) in enumerate(importances["raw_plus_semantic"], start=1)
                             if GROUPS[name] == "raw"}
        combined_importance = dict(importances["raw_plus_semantic"])
        raw_alone_importance = dict(importances["raw"])
        demoted = sorted(
            (
                {
                    "feature": name,
                    "importance_raw_only": raw_alone_importance[name],
                    "importance_raw_plus_semantic": combined_importance[name],
                    "rank_raw_only": raw_rank_alone[name],
                    "rank_raw_plus_semantic": raw_rank_combined[name],
                    "importance_retained": (combined_importance[name] / raw_alone_importance[name])
                    if raw_alone_importance[name] else 0.0,
                }
                for name in RAW_FEATURE_NAMES
            ),
            key=lambda item: item["importance_retained"],
        )
        unused_semantic = [
            {"feature": name, "importance": value,
             "shap": dict(shap_values["semantic"])[name],
             "constant_on_evaluation_set": next(item["constant"] for item in correlation if item["semantic_feature"] == name)}
            for name, value in importances["semantic"] if value == 0.0
        ]
        never_used_anywhere = [
            item["feature"] for item in unused_semantic
            if combined_importance.get(item["feature"], 0.0) == 0.0
            and dict(shap_values["raw_plus_semantic"]).get(item["feature"], 0.0) == 0.0
        ]
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "protocol": {
                "priming_rows": f"0..{EVALUATION_START:,}",
                "ml_train_rows": f"{TRAIN_START:,}..{EVALUATION_START:,}",
                "evaluation_rows": f"{EVALUATION_START:,}..{DENSE_PERIOD_END:,}",
                "evaluation_events": EVALUATION_ROWS,
                "evaluation_positives": int(test_y.sum()),
                "train_positives": int(data["train_y"].sum()),
                "alert_threshold": ML_STANDALONE_THRESHOLD,
                "catboost": {"iterations": 200, "depth": 6, "learning_rate": 0.05, "loss_function": "Logloss",
                             "random_seed": 20260804, "thread_count": self.config.threads,
                             "scale_pos_weight": "(n - positives) / positives, computed per partition"},
                "semantic_ontology_hash": ONTOLOGY_HASH,
                "behaviour_ontology_hash": BEHAVIOUR_ONTOLOGY_HASH,
                "frozen": "Runtime, Semantic Context Layer, Behaviour Layer, policies, routing, thresholds, "
                          "CatBoost hyperparameters, chronological split, train/test protocol and history window are unchanged",
                "shap_implementation": "CatBoost exact TreeSHAP (get_feature_importance type=ShapValues); the `shap` package is not installed",
            },
            "reproduction_check": {
                "expected_from_window_study_F": {"recall": 0.901, "fp": 6128},
                "measured_semantic_only": {"recall": arms["semantic"]["metrics"]["recall"],
                                           "fp": arms["semantic"]["metrics"]["fp"]},
                "matches": abs(arms["semantic"]["metrics"]["recall"] - 0.901) < 0.001
                           and arms["semantic"]["metrics"]["fp"] == 6128,
            },
            "arms": arms,
            "feature_generation": {
                "raw_feature_seconds": data["raw_feature_seconds"],
                "semantic_feature_seconds": data["semantic_feature_seconds"],
                "total_pass_seconds": data["total_feature_seconds"],
            },
            "shap_seconds": shap_seconds,
            "group_importance": {name: self._group_totals(ranked) for name, ranked in importances.items()},
            "group_shap": {name: self._group_totals(ranked) for name, ranked in shap_values.items()},
            "ablations": ablations,
            "raw_features_demoted_by_semantic": demoted,
            "unused_semantic_features": unused_semantic,
            "semantic_features_never_used_in_either_model": never_used_anywhere,
            "peak_rss_bytes": _peak_rss_bytes(),
        }

    # -- outputs -----------------------------------------------------------------
    def _write(self, report, importances, shap_values, ablations, correlation, probabilities, test_y) -> None:
        (self.output_dir / "comparison_results.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        with (self.output_dir / "feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("model", "rank", "feature", "group", "catboost_importance"))
            for model_name in ("semantic", "raw_plus_semantic", "raw"):
                for rank, (name, value) in enumerate(importances[model_name][:100], start=1):
                    writer.writerow((model_name, rank, name, GROUPS[name], f"{value:.6f}"))

        with (self.output_dir / "shap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("model", "rank", "feature", "group", "mean_abs_shap"))
            for model_name in ("semantic", "raw_plus_semantic", "raw"):
                for rank, (name, value) in enumerate(shap_values[model_name][:100], start=1):
                    writer.writerow((model_name, rank, name, GROUPS[name], f"{value:.8f}"))

        with (self.output_dir / "feature_groups.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("model", "attribution", "group", "absolute", "share"))
            for attribution, source in (("catboost_importance", report["group_importance"]),
                                        ("mean_abs_shap", report["group_shap"])):
                for model_name, totals in source.items():
                    for group, value in totals.items():
                        writer.writerow((model_name, attribution, group, f"{value['absolute']:.6f}", f"{value['share']:.6f}"))

        with (self.output_dir / "feature_correlation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("semantic_feature", "group", "constant",
                                                        "max_abs_pearson_r", "closest_raw_feature", "r"))
            writer.writeheader()
            for row in sorted(correlation, key=lambda item: -item["max_abs_pearson_r"]):
                writer.writerow(row)

        with (self.output_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("configuration", "dropped_group", "features", "tp", "fp", "fn", "tn",
                             "recall", "precision", "f1", "roc_auc", "pr_auc", "alert_rate"))
            for name in ("raw", "semantic", "raw_plus_semantic"):
                metric = report["arms"][name]["metrics"]
                writer.writerow((name, "-", report["arms"][name]["feature_count"], metric["tp"], metric["fp"],
                                 metric["fn"], metric["tn"], f"{metric['recall']:.6f}", f"{metric['precision']:.6f}",
                                 f"{metric['f1']:.6f}", f"{metric['roc_auc']:.6f}", f"{metric['pr_auc']:.6f}",
                                 f"{metric['alert_rate']:.6f}"))
            for record in ablations:
                metric = record["metrics"]
                writer.writerow((record["features"], record["dropped_group"], record["feature_count"], metric["tp"],
                                 metric["fp"], metric["fn"], metric["tn"], f"{metric['recall']:.6f}",
                                 f"{metric['precision']:.6f}", f"{metric['f1']:.6f}", f"{metric['roc_auc']:.6f}",
                                 f"{metric['pr_auc']:.6f}", f"{metric['alert_rate']:.6f}"))

        for kind in ("roc", "pr"):
            figure, axis = plt.subplots(figsize=(7, 5.5))
            for name, style in (("raw", "-"), ("semantic", "--"), ("raw_plus_semantic", "-.")):
                values = probabilities[name]
                if kind == "roc":
                    x, y, _ = roc_curve(test_y, values)
                    axis.plot(x, y, style, label=f"{name} (AUC={roc_auc_score(test_y, values):.4f})")
                else:
                    y, x, _ = precision_recall_curve(test_y, values)
                    axis.plot(x, y, style, label=f"{name} (AP={average_precision_score(test_y, values):.5f})")
            if kind == "roc":
                axis.plot((0, 1), (0, 1), "k:", linewidth=0.8)
                axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — CatBoost, frozen protocol")
            else:
                axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall — CatBoost, frozen protocol")
                axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=9)
            figure.tight_layout()
            figure.savefig(self.output_dir / f"{kind}_curves.png", dpi=160)
            plt.close(figure)

        self._report(report, importances, shap_values, ablations, correlation)

    def _report(self, report, importances, shap_values, ablations, correlation) -> None:
        arms = report["arms"]

        def table(headers, rows):
            return (["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
                    + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])

        final = [
            (name.replace("raw_plus_semantic", "Raw + Semantic").replace("semantic", "Semantic").replace("raw", "Raw"),
             f"{arms[name]['metrics']['recall']:.4f}", f"{arms[name]['metrics']['precision']:.6f}",
             f"{arms[name]['metrics']['f1']:.6f}", f"{arms[name]['metrics']['fp']:,}", arms[name]["metrics"]["fn"],
             f"{arms[name]['metrics']['roc_auc']:.4f}", f"{arms[name]['metrics']['pr_auc']:.5f}")
            for name in ("raw", "semantic", "raw_plus_semantic")
        ]
        detail = [
            (name, arms[name]["feature_count"], arms[name]["metrics"]["tp"], arms[name]["metrics"]["fp"],
             arms[name]["metrics"]["fn"], arms[name]["metrics"]["tn"], f"{arms[name]['metrics']['alert_rate']:.5f}",
             f"{arms[name]['training_seconds']:.1f}", f"{arms[name]['prediction_seconds']:.2f}",
             f"{arms[name]['inference_latency_ms']:.5f}", f"{arms[name]['peak_rss_bytes']:,}")
            for name in ("raw", "semantic", "raw_plus_semantic")
        ]
        group_rows = []
        for model_name in ("semantic", "raw_plus_semantic"):
            for group, value in report["group_importance"][model_name].items():
                shap_share = report["group_shap"][model_name].get(group, {}).get("share", 0.0)
                group_rows.append((model_name, group.lstrip("_"), f"{value['absolute']:.4f}",
                                   f"{value['share']:.4f}", f"{shap_share:.4f}"))
        ablation_rows = [
            (record["dropped_group"], record["feature_count"], record["metrics"]["tp"], f"{record['metrics']['fp']:,}",
             f"{record['metrics']['recall']:.4f}", f"{record['metrics']['precision']:.6f}",
             f"{record['metrics']['f1']:.6f}", f"{record['metrics']['pr_auc']:.5f}")
            for record in ablations
        ]
        correlated = sorted(correlation, key=lambda item: -item["max_abs_pearson_r"])[:15]
        lines = [
            "# Semantic versus Raw — did the Semantic Runtime create information?", "",
            EXPERIMENT_VERSION, "",
            "## Protocol", "",
            f"Priming rows {report['protocol']['priming_rows']}; ML training rows {report['protocol']['ml_train_rows']} "
            f"({report['protocol']['train_positives']} positives); evaluation rows {report['protocol']['evaluation_rows']} "
            f"({report['protocol']['evaluation_events']:,} events, {report['protocol']['evaluation_positives']} positives).", "",
            report["protocol"]["frozen"] + ".", "",
            f"**Reproduction check.** Semantic-only must reproduce the window study's `ml_only/CatBoost` at window F "
            f"(recall 0.901, FP 6,128). Measured: recall "
            f"{report['reproduction_check']['measured_semantic_only']['recall']:.4f}, FP "
            f"{report['reproduction_check']['measured_semantic_only']['fp']:,} — "
            f"**{'matches' if report['reproduction_check']['matches'] else 'DOES NOT MATCH'}**.", "",
            "## Final table", "",
            *table(("Features", "Recall", "Precision", "F1", "FP", "FN", "ROC", "PR"), final), "",
            "## Full metrics", "",
            *table(("Arm", "Features", "TP", "FP", "FN", "TN", "Alert rate", "Train s", "Predict s",
                    "Inference ms/event", "Peak RSS"), detail), "",
            f"Feature generation: raw {report['feature_generation']['raw_feature_seconds']:.1f} s, "
            f"semantic {report['feature_generation']['semantic_feature_seconds']:.1f} s, "
            f"single pass total {report['feature_generation']['total_pass_seconds']:.1f} s.", "",
            "## Importance by vocabulary", "",
            *table(("Model", "Group", "CatBoost importance", "Importance share", "SHAP share"), group_rows), "",
            "## Ablations — drop one vocabulary from Raw + Semantic", "",
            *table(("Dropped", "Features", "TP", "FP", "Recall", "Precision", "F1", "PR-AUC"), ablation_rows), "",
            "## Most raw-correlated semantic features", "",
            *table(("Semantic feature", "Group", "max |r| with any raw feature", "Closest raw feature"),
                   [(item["semantic_feature"], item["group"], f"{item['max_abs_pearson_r']:.4f}",
                     item["closest_raw_feature"]) for item in correlated]), "",
            "## Files", "",
            "- `comparison_results.json`, `feature_importance.csv`, `shap_summary.csv`",
            "- `feature_groups.csv`, `feature_correlation.csv`, `ablation_results.csv`",
            "- `roc_curves.png`, `pr_curves.png`", "",
            "## Critical analysis and final conclusion", "",
            "The six analysis questions and the single final conclusion are answered from these files in "
            "[`docs/semantic_vs_raw_analysis.md`](../../docs/semantic_vs_raw_analysis.md).", "",
        ]
        (self.output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("data/ibm_aml_data/HI-Small_Trans.csv"))
    parser.add_argument("--accounts", type=Path, default=Path("data/ibm_aml_data/HI-Small_accounts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/semantic_vs_raw"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = SemanticVsRaw(args.transactions, args.accounts, args.output_dir, args.threads).run()
    print(json.dumps({"output_dir": str(args.output_dir),
                      "arms": {name: value["metrics"] for name, value in report["arms"].items()},
                      "reproduction_check": report["reproduction_check"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
