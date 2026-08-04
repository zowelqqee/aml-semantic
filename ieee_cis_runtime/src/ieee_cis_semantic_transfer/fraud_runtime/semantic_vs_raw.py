"""Chronological IEEE-CIS transfer experiment: Raw vs Semantic vs Raw + Semantic.

Labels are read only into ``y`` after causal feature construction.  The
semantic and behaviour folds are committed strictly after each row's feature
vector has been created.  No row from the later Kaggle test file is used.
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
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

from .behaviour import BehaviourDecisionRuntime, SEMANTIC_FEATURE_NAMES, semantic_feature_vector
from .dataset import IEEECISLoader

EXPERIMENT_VERSION = "ieee-cis-semantic-transfer/1.0"
RAW_FEATURE_NAMES = (
    "raw_transaction_dt", "raw_amount", "raw_product_channel", "raw_card_identity", "raw_billing_region", "raw_billing_country",
    "raw_client_signature", "raw_has_identity", "raw_purchaser_email_domain", "raw_recipient_email_domain", "raw_dist1", "raw_dist2",
    "raw_network", "raw_card_type", "raw_device_type",
)
RAW_CATEGORICAL = (2, 3, 4, 5, 6, 8, 9, 12, 13, 14)
COMBINED_FEATURE_NAMES = RAW_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES
SEMANTIC_GROUPS = {
    **{name: "semantic_objects" for name in SEMANTIC_FEATURE_NAMES if name.startswith("sem_")},
    **{name: "behaviours" for name in SEMANTIC_FEATURE_NAMES if name.startswith("beh_")},
    **{name: "scenarios" for name in SEMANTIC_FEATURE_NAMES if name.startswith("scn_")},
    **{name: "roles" for name in ("role_code", "role_confidence", "role_tenure_minutes", "role_transition_count")},
    **{name: "lifecycle" for name in SEMANTIC_FEATURE_NAMES if name.startswith("lifecycle_")},
    **{name: "evidence" for name in SEMANTIC_FEATURE_NAMES if name.startswith("evidence_")},
}
GROUPS = {**{name: "raw" for name in RAW_FEATURE_NAMES}, **SEMANTIC_GROUPS}


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _metrics(y: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=(0, 1)).ravel()
    return {"recall": float(recall_score(y, predicted, zero_division=0)), "precision": float(precision_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)), "false_positives": int(fp), "false_negatives": int(fn),
            "true_positives": int(tp), "true_negatives": int(tn), "alert_rate": float(predicted.mean())}


@dataclass(frozen=True)
class ExperimentConfig:
    train_fraction: float = 0.80
    row_cap: int = 0
    iterations: int = 200
    depth: int = 6
    learning_rate: float = 0.05
    seed: int = 20260804
    threads: int = 4
    threshold: float = 0.50
    shap_rows: int = 25_000


class FraudSemanticTransfer:
    def __init__(self, transactions: str | Path, identity: str | Path, output_dir: str | Path, config: ExperimentConfig | None = None) -> None:
        self.transactions, self.identity, self.output_dir = Path(transactions), Path(identity), Path(output_dir)
        self.config = config or ExperimentConfig()

    @staticmethod
    def _raw_values(transaction) -> tuple[object, ...]:
        return (transaction.transaction_dt, transaction.amount, transaction.product_channel, transaction.card_id, transaction.billing_region,
                transaction.billing_country, transaction.device_id or "(no-identity-join)", int(transaction.has_identity),
                transaction.purchaser_email_domain, transaction.recipient_email_domain, transaction.distance1 if transaction.distance1 is not None else -1.0,
                transaction.distance2 if transaction.distance2 is not None else -1.0, transaction.metadata.get("network", ""),
                transaction.metadata.get("card_type", ""), transaction.metadata.get("device_type", ""))

    def _build(self) -> dict[str, object]:
        loader = IEEECISLoader(self.transactions, self.identity)
        identity = loader.load_identity()
        total = sum(1 for _ in loader.rows(identity, limit=self.config.row_cap or None))
        split = int(total * self.config.train_fraction)
        # Category mappings are fitted solely on the chronological training
        # prefix, without labels.  Compact numerical codes avoid holding three
        # large mixed-object matrices in memory during the combined arm.
        category_values = [set() for _ in RAW_CATEGORICAL]
        for index, transaction in loader.rows(identity, limit=split):
            values = self._raw_values(transaction)
            for position, raw_position in enumerate(RAW_CATEGORICAL):
                category_values[position].add(str(values[raw_position]))
        category_maps = [{value: code for code, value in enumerate(sorted(values))} for values in category_values]
        category_lookup = {raw_position: category_maps[position] for position, raw_position in enumerate(RAW_CATEGORICAL)}
        raw = np.empty((total, len(RAW_FEATURE_NAMES)), dtype=np.float32)
        semantic = np.empty((total, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        labels = np.empty(total, dtype=np.uint8)
        runtime = BehaviourDecisionRuntime()
        raw_seconds = semantic_seconds = 0.0
        started_all = time.perf_counter()
        for index, transaction in loader.rows(identity, limit=self.config.row_cap or None):
            started = time.perf_counter()
            values = self._raw_values(transaction)
            raw[index] = tuple(float(category_lookup[position].get(str(value), -1)) if position in category_lookup else float(value)
                               for position, value in enumerate(values))
            raw_seconds += time.perf_counter() - started
            started = time.perf_counter()
            result = runtime.evaluate(transaction, transaction.transaction_dt // 60, index)
            policy = result.policies[0].metrics
            semantic[index] = semantic_feature_vector({item.type: item.confidence for item in result.semantic.context.objects}, result.behaviour,
                                                       result.evidence, len(result.conflicts), int(policy["independent_unqualified_topics"]),
                                                       float(policy["effective_risk"]))
            runtime.commit(transaction, result, transaction.transaction_dt // 60)
            semantic_seconds += time.perf_counter() - started
            labels[index] = int(transaction.is_fraud)
        return {"raw_train": raw[:split], "raw_test": raw[split:], "semantic_train": semantic[:split], "semantic_test": semantic[split:],
                "y_train": labels[:split], "y_test": labels[split:], "total": total, "split": split,
                "raw_feature_seconds": raw_seconds, "semantic_feature_seconds": semantic_seconds,
                "raw_category_cardinalities": {RAW_FEATURE_NAMES[position]: len(category_lookup[position]) for position in RAW_CATEGORICAL},
                "total_feature_seconds": time.perf_counter() - started_all}

    def _model(self, y: np.ndarray) -> CatBoostClassifier:
        scale = (len(y) - int(y.sum())) / max(1, int(y.sum()))
        return CatBoostClassifier(iterations=self.config.iterations, depth=self.config.depth, learning_rate=self.config.learning_rate,
                                 loss_function="Logloss", eval_metric="AUC", random_seed=self.config.seed, scale_pos_weight=scale,
                                 thread_count=self.config.threads, verbose=False, allow_writing_files=False)

    def _fit(self, name: str, train_x, test_x, y_train, y_test, names: tuple[str, ...], categorical: tuple[int, ...] = (), keep: bool = True):
        model = self._model(y_train)
        train_pool = Pool(train_x, y_train, cat_features=list(categorical)) if categorical else Pool(train_x, y_train)
        test_pool = Pool(test_x, y_test, cat_features=list(categorical)) if categorical else Pool(test_x, y_test)
        started = time.perf_counter(); model.fit(train_pool); training = time.perf_counter() - started
        started = time.perf_counter(); probability = model.predict_proba(test_pool)[:, 1]; inference = time.perf_counter() - started
        record = {"features": name, "feature_count": len(names), "metrics": _metrics(y_test, probability, (probability >= self.config.threshold).astype(np.uint8)),
                  "training_seconds": training, "inference_seconds": inference, "inference_latency_ms": 1000 * inference / len(y_test), "peak_rss_bytes": _peak_rss_bytes()}
        if not keep:
            del model, train_pool, test_pool; gc.collect()
            return record, probability, None
        return record, probability, model

    @staticmethod
    def _importance(model, names: tuple[str, ...]) -> list[tuple[str, float]]:
        return sorted(zip(names, (float(x) for x in model.get_feature_importance())), key=lambda x: -x[1])

    def _shap(self, model, x, y, names, categorical=()) -> tuple[list[tuple[str, float]], float, int]:
        rows = min(self.config.shap_rows, len(y))
        pool = Pool(x[:rows], y[:rows], cat_features=list(categorical)) if categorical else Pool(x[:rows], y[:rows])
        started = time.perf_counter(); values = model.get_feature_importance(pool, type="ShapValues"); seconds = time.perf_counter() - started
        return sorted(zip(names, (float(v) for v in np.abs(values[:, :-1]).mean(axis=0))), key=lambda x: -x[1]), seconds, rows

    @staticmethod
    def _numeric_raw(raw: np.ndarray) -> np.ndarray:
        """Train-only ordinal coding is unnecessary for correlation: categorical raw columns are excluded."""
        return np.asarray(raw[:, (0, 1, 7, 10, 11)], dtype=np.float64)

    def _correlation(self, raw: np.ndarray, semantic: np.ndarray) -> list[dict[str, object]]:
        numeric = self._numeric_raw(raw); raw_names = ("raw_transaction_dt", "raw_amount", "raw_has_identity", "raw_dist1", "raw_dist2")
        rows = []
        for pos, name in enumerate(SEMANTIC_FEATURE_NAMES):
            value = semantic[:, pos].astype(np.float64)
            if value.std() == 0:
                rows.append({"semantic_feature": name, "group": GROUPS[name], "constant": 1, "max_abs_pearson_r": 0.0, "closest_raw_feature": "", "r": 0.0}); continue
            scores = []
            for raw_pos, raw_name in enumerate(raw_names):
                other = numeric[:, raw_pos]
                r = 0.0 if other.std() == 0 else float(np.corrcoef(value, other)[0, 1])
                scores.append((raw_name, r))
            closest, score = max(scores, key=lambda x: abs(x[1]))
            rows.append({"semantic_feature": name, "group": GROUPS[name], "constant": 0, "max_abs_pearson_r": abs(score), "closest_raw_feature": closest, "r": score})
        return rows

    @staticmethod
    def _groups(ranked: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
        totals: dict[str, float] = defaultdict(float)
        for name, value in ranked: totals[GROUPS[name]] += value
        overall = sum(totals.values()) or 1.0
        return {group: {"absolute": value, "share": value / overall} for group, value in sorted(totals.items())}

    def _ablations(self, data: dict[str, object]) -> list[dict[str, object]]:
        out = []; index = {name: i for i, name in enumerate(SEMANTIC_FEATURE_NAMES)}
        for group in ("raw",) + tuple(sorted(set(SEMANTIC_GROUPS.values()))):
            if group == "raw": train, test, names, cats = data["semantic_train"], data["semantic_test"], SEMANTIC_FEATURE_NAMES, ()
            else:
                keep = [index[name] for name in SEMANTIC_FEATURE_NAMES if GROUPS[name] != group]
                train, test = np.hstack((data["raw_train"], data["semantic_train"][:, keep])), np.hstack((data["raw_test"], data["semantic_test"][:, keep]))
                names, cats = RAW_FEATURE_NAMES + tuple(name for name in SEMANTIC_FEATURE_NAMES if GROUPS[name] != group), ()
            record, _p, _m = self._fit(f"raw_plus_semantic_minus_{group}", train, test, data["y_train"], data["y_test"], names, cats, keep=False)
            record["dropped_group"] = group; record["dropped_feature_count"] = len(COMBINED_FEATURE_NAMES) - len(names); out.append(record)
        return out

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = self._build()
        raw_train, raw_test, sem_train, sem_test, y_train, y_test = data["raw_train"], data["raw_test"], data["semantic_train"], data["semantic_test"], data["y_train"], data["y_test"]
        combined_train, combined_test = np.hstack((raw_train, sem_train)), np.hstack((raw_test, sem_test))
        arms = {}; probabilities = {}; models = {}; importances = {}; shaps = {}; shap_seconds = {}
        for name, train, test, names, cats in (("raw", raw_train, raw_test, RAW_FEATURE_NAMES, ()), ("semantic", sem_train, sem_test, SEMANTIC_FEATURE_NAMES, ()), ("raw_plus_semantic", combined_train, combined_test, COMBINED_FEATURE_NAMES, ())):
            arms[name], probabilities[name], models[name] = self._fit(name, train, test, y_train, y_test, names, cats)
            importances[name] = self._importance(models[name], names)
            shaps[name], shap_seconds[name], shap_rows = self._shap(models[name], test, y_test, names, cats)
            print(json.dumps({"arm": name, **arms[name]["metrics"], "shap_seconds": round(shap_seconds[name], 3)}), flush=True)
        ablations = self._ablations(data)
        correlation = self._correlation(raw_test, sem_test)
        report = {"experiment_version": EXPERIMENT_VERSION,
                  "protocol": {"dataset": "IEEE-CIS train_transaction + train_identity", "source_rows": data["total"], "chronology": "input row order verified non-decreasing TransactionDT; chronological 80/20 split", "training_rows": data["split"], "evaluation_rows": data["total"] - data["split"], "training_positives": int(y_train.sum()), "evaluation_positives": int(y_test.sum()), "excluded": "Unlabelled later Kaggle test partition is not read", "raw_category_encoding": "training-prefix-only ordinal codes; no labels and no evaluation rows participate in vocabulary fitting", "raw_category_cardinalities": data["raw_category_cardinalities"], "threshold": self.config.threshold, "catboost": {"iterations": self.config.iterations, "depth": self.config.depth, "learning_rate": self.config.learning_rate, "random_seed": self.config.seed, "thread_count": self.config.threads, "scale_pos_weight": "(n - positives) / positives in the shared chronological training partition"}, "shap": {"implementation": "CatBoost exact TreeSHAP", "evaluation_prefix_rows": shap_rows}},
                  "arms": arms, "feature_generation": {k: data[k] for k in ("raw_feature_seconds", "semantic_feature_seconds", "total_feature_seconds")}, "feature_importance": {k: v[:100] for k, v in importances.items()}, "shap": {k: v[:100] for k, v in shaps.items()}, "shap_seconds": shap_seconds, "feature_group_contribution": {k: self._groups(v) for k, v in importances.items()}, "feature_group_shap": {k: self._groups(v) for k, v in shaps.items()}, "ablations": ablations, "correlation": correlation, "peak_rss_bytes": _peak_rss_bytes()}
        self._write(report, probabilities, importances, shaps, ablations, correlation, y_test)
        return report

    def _write(self, report, probabilities, importances, shaps, ablations, correlation, y) -> None:
        (self.output_dir / "comparison_results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for filename, contents, header in (("feature_importance.csv", importances, ("model", "rank", "feature", "group", "catboost_importance")), ("shap_summary.csv", shaps, ("model", "rank", "feature", "group", "mean_abs_shap"))):
            with (self.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(header)
                for model, ranked in contents.items():
                    for rank, (feature, value) in enumerate(ranked, 1): writer.writerow((model, rank, feature, GROUPS[feature], f"{value:.8f}"))
        with (self.output_dir / "feature_correlation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("semantic_feature", "group", "constant", "max_abs_pearson_r", "closest_raw_feature", "r")); writer.writeheader(); writer.writerows(correlation)
        with (self.output_dir / "feature_groups.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("model", "attribution", "group", "absolute", "share"))
            for attribution, source in (("catboost_importance", report["feature_group_contribution"]), ("mean_abs_shap", report["feature_group_shap"])):
                for model, groups in source.items():
                    for group, values in groups.items(): writer.writerow((model, attribution, group, f"{values['absolute']:.8f}", f"{values['share']:.8f}"))
        with (self.output_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("configuration", "dropped_group", "features", "recall", "precision", "f1", "roc_auc", "pr_auc", "false_positives", "false_negatives", "alert_rate"))
            for name, arm in report["arms"].items():
                m = arm["metrics"]; writer.writerow((name, "-", arm["feature_count"], m["recall"], m["precision"], m["f1"], m["roc_auc"], m["pr_auc"], m["false_positives"], m["false_negatives"], m["alert_rate"]))
            for arm in ablations:
                m = arm["metrics"]; writer.writerow((arm["features"], arm["dropped_group"], arm["feature_count"], m["recall"], m["precision"], m["f1"], m["roc_auc"], m["pr_auc"], m["false_positives"], m["false_negatives"], m["alert_rate"]))
        for kind in ("roc", "pr"):
            figure, axis = plt.subplots(figsize=(7, 5.5))
            for name, style in (("raw", "-"), ("semantic", "--"), ("raw_plus_semantic", "-.")):
                if kind == "roc": x, yv, _ = roc_curve(y, probabilities[name]); axis.plot(x, yv, style, label=f"{name} ({report['arms'][name]['metrics']['roc_auc']:.4f})")
                else: yv, x, _ = precision_recall_curve(y, probabilities[name]); axis.plot(x, yv, style, label=f"{name} ({report['arms'][name]['metrics']['pr_auc']:.4f})")
            axis.set(xlabel="False positive rate" if kind == "roc" else "Recall", ylabel="True positive rate" if kind == "roc" else "Precision", title=f"IEEE-CIS {kind.upper()} — chronological transfer experiment"); axis.grid(alpha=.25); axis.legend(); figure.tight_layout(); figure.savefig(self.output_dir / f"{kind}_curves.png", dpi=160); plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("ieee_cis_data/train_transaction.csv")); parser.add_argument("--identity", type=Path, default=Path("ieee_cis_data/train_identity.csv")); parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fraud_semantic")); parser.add_argument("--row-cap", type=int, default=0); parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); report = FraudSemanticTransfer(args.transactions, args.identity, args.output_dir, ExperimentConfig(row_cap=args.row_cap, threads=args.threads)).run(); print(json.dumps({"arms": {k: v["metrics"] for k, v in report["arms"].items()}}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
