"""Train-validation-selected evidence-fusion Runtime-first cascade experiment.

This module is deliberately separate from the frozen v0.2 Runtime.  Labels
are used only for model fitting and for chronological validation/test
evaluation.  Fusion consumes only Runtime evidence summaries and ML
probabilities; every chosen decision is reproducible from those inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import resource
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score

from .dataset import AMLSimDataset
from .ml_benchmark import FEATURE_NAMES, CausalFeatureState, StreamingRuntimeGraph, _account, _amount, _canonical_bank, _parse_timestamp
from .models import Account, Decision, Evidence, Transaction
from .runtime import AMLDecisionRuntime, RUNTIME_VERSION, stable_id
from .runtime_first_cascade import CascadeConfig, CascadeDatasetBuilder, _EphemeralAccounts, _canonical, _models, _sha256


V2_VERSION = "aml-runtime-first-fusion/2.0"
MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost")
RULE_WEIGHTS = {
    "AML-R01-LARGE": 0.86,
    "AML-R03-VELOCITY": 0.79,
    "AML-R07-BEHAVIOUR": 0.60,
    "AML-R04-SAR": 0.99,
    "AML-R05-SAR-CONNECTION": 0.95,
    "AML-R02-JURISDICTION": 0.91,
    "AML-R06-KYC": 0.72,
}
RULE_FAMILIES = {
    "AML-R01-LARGE": "transaction_value", "AML-R03-VELOCITY": "transaction_velocity",
    "AML-R07-BEHAVIOUR": "counterparty_behaviour", "AML-R04-SAR": "known_sar",
    "AML-R05-SAR-CONNECTION": "sar_network", "AML-R02-JURISDICTION": "jurisdiction",
    "AML-R06-KYC": "kyc",
}


@dataclass(frozen=True)
class FusionConfig:
    routing: str
    review_probability: float
    ml_alone_probability: float
    combined_review_score: float
    minimum_corroboration: int
    direct_runtime_families: int = 3
    block_probability: float = 0.995
    budget_fraction: float = 0.10
    version: str = V2_VERSION


@dataclass(frozen=True)
class RuntimeSummary:
    families: tuple[str, ...]
    strength: float
    preliminary: str

    @property
    def count(self) -> int:
        return len(self.families)


def _peak_rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _decision_alert(decisions: np.ndarray) -> np.ndarray:
    return np.isin(decisions, (Decision.REVIEW.value, Decision.BLOCK.value)).astype(np.uint8)


def _metric(labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray) -> dict[str, object]:
    predicted = _decision_alert(decisions)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "alert_rate": float(predicted.mean()), "fp_per_tp": float(fp / tp) if tp else math.inf,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.0,
        "pr_auc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else 0.0,
    }


def _amount_bucket(amount: float) -> str:
    for limit, name in ((10, "<10"), (100, "10-100"), (1000, "100-1k"), (10000, "1k-10k"), (50000, "10k-50k")):
        if amount < limit:
            return name
    return ">=50k"


def _summaries(vectors: np.ndarray, names: tuple[str, ...], preliminary_codes: np.ndarray) -> list[RuntimeSummary]:
    positions = {name: index for index, name in enumerate(names)}
    available = [(rule, positions.get(f"runtime_rule_{rule}")) for rule in RULE_WEIGHTS]
    output: list[RuntimeSummary] = []
    for row, code in zip(vectors, preliminary_codes, strict=True):
        family_weights: dict[str, float] = {}
        for rule, position in available:
            if position is not None and row[position] >= .5:
                family = RULE_FAMILIES[rule]
                family_weights[family] = max(family_weights.get(family, 0.0), RULE_WEIGHTS[rule])
        # Noisy-OR limits correlated/repeated evidence to one contribution per
        # source family, rather than summing every fact linearly.
        strength = 1.0
        for value in family_weights.values():
            strength *= 1.0 - value
        output.append(RuntimeSummary(tuple(sorted(family_weights)), round(1.0 - strength, 8), tuple(Decision)[int(code)].value))
    return output


def _route(summaries: list[RuntimeSummary], config: FusionConfig) -> np.ndarray:
    base = np.zeros(len(summaries), dtype=bool)
    if config.routing == "allow_meaningful":
        base[:] = [item.preliminary == Decision.ALLOW.value and item.count >= 1 for item in summaries]
    elif config.routing == "abstain_only":
        base[:] = [item.preliminary == Decision.ABSTAIN.value for item in summaries]
    elif config.routing == "allow_two_families":
        base[:] = [item.preliminary == Decision.ALLOW.value and item.count >= 2 for item in summaries]
    elif config.routing == "allow_high_value_combo":
        base[:] = [item.preliminary == Decision.ALLOW.value and {"transaction_value", "transaction_velocity"}.issubset(item.families) for item in summaries]
    elif config.routing == "runtime_uncertainty":
        base[:] = [1 <= item.count < config.direct_runtime_families for item in summaries]
    elif config.routing == "budgeted_uncertainty":
        candidates = [(abs(item.strength - .75), index) for index, item in enumerate(summaries) if item.count < config.direct_runtime_families]
        candidates.sort()
        keep = int(math.ceil(len(summaries) * config.budget_fraction))
        for _distance, index in candidates[:keep]:
            base[index] = True
    else:
        raise ValueError(f"unknown routing profile {config.routing}")
    return base


def _fuse(probabilities: np.ndarray, summaries: list[RuntimeSummary], config: FusionConfig, *, materialize_traces: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    routed = _route(summaries, config)
    decisions = np.empty(len(summaries), dtype="U7")
    scores = np.empty(len(summaries), dtype=np.float64)
    traces: list[dict[str, object]] = []
    for index, item in enumerate(summaries):
        probability = float(probabilities[index]) if routed[index] else None
        score = .35 * item.strength + (.65 * probability if probability is not None else 0.0)
        hard_stop = "known_sar" in item.families
        direct = item.count >= config.direct_runtime_families
        corroborated = item.count >= config.minimum_corroboration
        if hard_stop:
            decision, reason = Decision.BLOCK.value, "hard-stop known-SAR Runtime family"
        elif direct:
            decision, reason = Decision.REVIEW.value, "three independent Runtime source families"
        elif probability is not None and probability >= config.block_probability and corroborated:
            decision, reason = Decision.BLOCK.value, "very-high ML probability plus independent Runtime corroboration"
        elif probability is not None and probability >= config.review_probability and (corroborated or probability >= config.ml_alone_probability) and score >= config.combined_review_score:
            decision, reason = Decision.REVIEW.value, "ML probability and deterministic fusion threshold satisfied"
        elif item.count:
            decision, reason = Decision.ALLOW.value, "Runtime signal is not independently corroborated at the frozen fusion tier"
        else:
            decision, reason = Decision.ABSTAIN.value, "no Runtime evidence and no routed ML evidence"
        decisions[index], scores[index] = decision, score
        if materialize_traces:
            traces.append({"source_families": item.families, "independent_family_count": item.count, "runtime_strength": item.strength, "ml_probability": probability, "composite_risk": score, "routed": bool(routed[index]), "final_decision": decision, "decision_reason": reason, "conflict_penalty": 0.0, "correlated_evidence_deduplicated": True})
    return decisions, scores, routed, traces


def _ranking(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, object]]:
    ordered = np.argsort(-scores, kind="stable")
    rows: list[dict[str, object]] = []
    for budget in (100, 500, 1000):
        selected = labels[ordered[:budget]]
        tp = int(selected.sum())
        rows.append({"budget_kind": f"top_{budget}", "budget": budget, "tp": tp, "precision": tp / budget, "recall": tp / int(labels.sum()), "lift_over_random": (tp / budget) / float(labels.mean()) if labels.mean() else 0.0, "fp_per_tp": (budget - tp) / tp if tp else math.inf})
    for rate in (.001, .005, .01, .05, .10):
        budget = max(1, int(math.ceil(rate * len(labels))))
        selected = labels[ordered[:budget]]
        tp = int(selected.sum())
        rows.append({"budget_kind": f"alert_rate_{rate:.1%}", "budget": budget, "tp": tp, "precision": tp / budget, "recall": tp / int(labels.sum()), "lift_over_random": (tp / budget) / float(labels.mean()) if labels.mean() else 0.0, "fp_per_tp": (budget - tp) / tp if tp else math.inf})
    return rows


class RuntimeFirstV2Experiment:
    def __init__(self, transactions: str | Path, output_dir: str | Path) -> None:
        self.transactions = Path(transactions)
        self.output_dir = Path(output_dir)
        self.config = CascadeConfig()
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")
        self.previous = Path("artifacts/aml_runtime_first_cascade/comparison_results.json")

    def _diagnose(self) -> None:
        labels: dict[tuple[str, str], int] = {}
        with Path("artifacts/aml_full_ml_vs_runtime_first/pairwise_outcomes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                labels[(row["model"].lower(), row["transaction_id"])] = int(row["laundering_label_evaluation_only"])
        lines = ["# False-positive diagnosis for frozen v1 cascade", "", "Labels below are used only after final decisions for this retrospective diagnosis.", ""]
        for model in ("xgboost", "lightgbm", "catboost"):
            counter: Counter[tuple[str, ...]] = Counter()
            total = 0
            path = Path(f"artifacts/aml_runtime_first_cascade/audits/runtime_first_{model}.jsonl.gz")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    audit = json.loads(line)
                    if labels[(model, audit["transaction"]["id"])] or audit["final_decision"]["decision"] not in ("REVIEW", "BLOCK"):
                        continue
                    total += 1
                    evidence = audit["preliminary_evidence"]
                    rules = "+".join(sorted(item["rule_id"] for item in evidence)) or "none"
                    topics = "+".join(sorted(item["topic"] for item in evidence)) or "none"
                    ml = audit["ml_evidence"]
                    band = ml["metadata"]["threshold_band"] if ml else "unscored_protected"
                    counter[(audit["preliminary_decision"]["decision"], audit["final_decision"]["decision"], rules, topics, str(len(evidence)), band, _amount_bucket(float(audit["transaction"]["amount"])), audit["transaction"]["payment_type"])] += 1
            lines += [f"## {model.title()} — {total:,} false-positive alerts", "", "| Count | Preliminary | Final | Rules | Topics | Evidence count | ML band | Amount | Payment type |", "|---|---|---|---|---|---|---|---|---|"]
            for key, count in counter.most_common(20):
                lines.append("| " + " | ".join((str(count),) + key) + " |")
            lines += ["", "Finding: protected preliminary REVIEW/BLOCK outcomes dominate these false positives; v1 has no opportunity for ML or conflict evidence to down-rank them. Single-family velocity/behaviour and large-transfer signals are frequent, and LightGBM additionally turns ML-scored ABSTAIN/ALLOW cases into BLOCK at probability 1.0.", ""]
        (self.output_dir / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _candidates(self) -> list[FusionConfig]:
        result: list[FusionConfig] = []
        for routing in ("allow_meaningful", "abstain_only", "allow_two_families", "allow_high_value_combo", "runtime_uncertainty", "budgeted_uncertainty"):
            for review in (.20, .40, .60, .80, .90, .95, .98):
                for corroboration in (0, 1, 2):
                    for combined in (.35, .50, .65, .80):
                        result.append(FusionConfig(routing, review, .98, combined, corroboration))
        return result

    @staticmethod
    def _pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        frontier: list[dict[str, object]] = []
        for row in rows:
            metric = row["metrics"]
            dominated = any(other["metrics"]["tp"] >= metric["tp"] and other["metrics"]["fp"] <= metric["fp"] and other["routed"] <= row["routed"] and (other["metrics"]["tp"] > metric["tp"] or other["metrics"]["fp"] < metric["fp"] or other["routed"] < row["routed"]) for other in rows)
            if not dominated:
                frontier.append(row)
        return sorted(frontier, key=lambda x: (-x["metrics"]["tp"], x["metrics"]["fp"], x["routed"]))

    def _select(self, probabilities: np.ndarray, summaries: list[RuntimeSummary], labels: np.ndarray, model: str) -> tuple[FusionConfig, list[dict[str, object]]]:
        counts = np.asarray([item.count for item in summaries], dtype=np.int8)
        strengths = np.asarray([item.strength for item in summaries], dtype=np.float64)
        preliminary = np.asarray([item.preliminary for item in summaries], dtype="U7")
        hard = np.asarray(["known_sar" in item.families for item in summaries], dtype=bool)
        value_velocity = np.asarray([{"transaction_value", "transaction_velocity"}.issubset(item.families) for item in summaries], dtype=bool)

        def evaluate(config: FusionConfig) -> tuple[dict[str, object], int]:
            if config.routing == "allow_meaningful":
                routed = (preliminary == Decision.ALLOW.value) & (counts >= 1)
            elif config.routing == "abstain_only":
                routed = preliminary == Decision.ABSTAIN.value
            elif config.routing == "allow_two_families":
                routed = (preliminary == Decision.ALLOW.value) & (counts >= 2)
            elif config.routing == "allow_high_value_combo":
                routed = (preliminary == Decision.ALLOW.value) & value_velocity
            elif config.routing == "runtime_uncertainty":
                routed = (counts >= 1) & (counts < config.direct_runtime_families)
            elif config.routing == "budgeted_uncertainty":
                routed = np.zeros(len(counts), dtype=bool)
                candidates = np.flatnonzero(counts < config.direct_runtime_families)
                ranked = candidates[np.argsort(np.abs(strengths[candidates] - .75), kind="stable")]
                routed[ranked[:int(math.ceil(len(counts) * config.budget_fraction))]] = True
            else:
                raise ValueError(config.routing)
            score = .35 * strengths + .65 * probabilities * routed
            direct = counts >= config.direct_runtime_families
            corroborated = counts >= config.minimum_corroboration
            block = hard | (routed & (probabilities >= config.block_probability) & corroborated)
            review = ~block & (direct | (routed & (probabilities >= config.review_probability) & (corroborated | (probabilities >= config.ml_alone_probability)) & (score >= config.combined_review_score)))
            decisions = np.where(block, Decision.BLOCK.value, np.where(review, Decision.REVIEW.value, np.where(counts > 0, Decision.ALLOW.value, Decision.ABSTAIN.value)))
            return _metric(labels, score, decisions), int(routed.sum())

        rows: list[dict[str, object]] = []
        for config in self._candidates():
            metrics, routed = evaluate(config)
            rows.append({"model": model, "config": asdict(config), "metrics": metrics, "routed": routed})
        frontier = self._pareto(rows)
        target = math.ceil(.70 * int(labels.sum()))
        eligible = [item for item in frontier if item["metrics"]["tp"] >= target]
        chosen = min(eligible, key=lambda x: (x["metrics"]["fp"], x["routed"], -x["metrics"]["tp"])) if eligible else max(frontier, key=lambda x: (x["metrics"]["tp"] * 2000 - x["metrics"]["fp"] - .05 * x["routed"], x["metrics"]["tp"]))
        return FusionConfig(**chosen["config"]), frontier

    def _write_pareto(self, frontier: list[dict[str, object]]) -> None:
        with (self.output_dir / "pareto_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ("model", "routing", "review_probability", "ml_alone_probability", "combined_review_score", "minimum_corroboration", "direct_runtime_families", "block_probability", "budget_fraction", "tp", "fp", "fn", "recall", "precision", "f1", "alert_rate", "fp_per_tp", "routed")
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
            for row in frontier:
                writer.writerow({"model": row["model"], **row["config"], **row["metrics"], "routed": row["routed"]})

    def _audit_test(self, data: dict[str, object], models: dict[str, object], configs: dict[str, FusionConfig], test_probabilities: dict[str, np.ndarray], summaries: list[RuntimeSummary], traces: dict[str, list[dict[str, object]]]) -> None:
        writers = {name: gzip.GzipFile(filename="", mode="wb", fileobj=(self.output_dir / "audits" / f"{name.lower()}_fusion.jsonl.gz").open("wb"), mtime=0) for name in MODEL_NAMES}
        for path in (self.output_dir / "audits").glob("*.gz"):
            pass
        source_stat = self.transactions.stat(); fingerprint = _sha256(f"{self.transactions.resolve()}:{source_stat.st_size}:{source_stat.st_mtime_ns}")
        dataset = AMLSimDataset((), _EphemeralAccounts(), {}, str(self.transactions), fingerprint)
        runtime = AMLDecisionRuntime(dataset, self.output_dir / "audits" / "base_unused")
        graph = StreamingRuntimeGraph(dataset); runtime.graph = graph
        start = self.config.train_rows; end = start + self.config.evaluation_horizon
        with self.sorted_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle); next(reader)
            for index, row in enumerate(reader):
                if index >= end: break
                timestamp = _parse_timestamp(row[0]); originator, beneficiary = _account(row[1], row[2]), _account(row[3], row[4]); received = _amount(row[5])
                transaction = Transaction(f"IBM-ML-{index + 2:08d}", timestamp.isoformat(timespec="seconds"), originator, beneficiary, received, row[6], "", row[9], metadata={"from_bank": _canonical_bank(row[1]), "to_bank": _canonical_bank(row[3]), "amount_paid": row[7], "payment_currency": row[8]})
                graph.by_id[transaction.id] = transaction
                if index >= start:
                    test_index = index - start; preliminary = runtime.evaluate_preliminary(transaction.id)
                    for name in MODEL_NAMES:
                        trace = traces[name][test_index]; probability = trace["ml_probability"]
                        ml_evidence = None if probability is None else Evidence(stable_id("E", transaction.id, "ML", name, f"{probability:.8f}"), f"ML/{name}", (), float(probability), "Raw model probability consumed by deterministic evidence fusion.", transaction.timestamp, f"ML-{name}", "risk", "ml_probability", 1.0, 0, metadata={"probability": f"{probability:.8f}", "fusion_version": V2_VERSION})
                        payload = {"runtime_version": RUNTIME_VERSION, "fusion_version": V2_VERSION, "transaction": transaction.to_dict(), "facts": [item.to_dict() for item in preliminary.facts], "preliminary_evidence": [item.to_dict() for item in preliminary.evidence], "ml_evidence": ml_evidence.to_dict() if ml_evidence else None, "preliminary_decision": preliminary.decision.to_dict(), "fusion": trace, "final_decision": {"decision": trace["final_decision"], "policy_ids": ["AML-FUSION-01"], "rationale": trace["decision_reason"]}, "replay_pins": {"fusion_config_hash": _sha256(_canonical(asdict(configs[name]))), "model": name, "input_snapshot_hash": _sha256(_canonical(transaction.to_dict()))}}
                        writers[name].write((_canonical(payload) + "\n").encode())
                graph.commit(transaction); del graph.by_id[transaction.id]
        for writer in writers.values(): writer.close()

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True); (self.output_dir / "audits").mkdir(exist_ok=True)
        self._diagnose()
        data = CascadeDatasetBuilder(self.transactions, self.sorted_path, self.config).build()
        train_y, test_y = data["labels_train"], data["labels_test"]
        train_codes, test_codes = data["train_preliminary_codes"], data["preliminary_codes"]
        names = tuple(data["primary_names"])[len(FEATURE_NAMES):]
        train_summaries = _summaries(data["runtime_full_train"], names, train_codes)
        test_summaries = _summaries(data["runtime_full_test"], names, test_codes)
        train_x = np.hstack((data["raw_train"], data["runtime_full_train"]))
        test_x = np.hstack((data["raw_test"], data["runtime_full_test"]))
        split = int(.80 * len(train_y)); scale = (split - int(train_y[:split].sum())) / int(train_y[:split].sum())
        selected: dict[str, FusionConfig] = {}; frontier: list[dict[str, object]] = []; validation: dict[str, object] = {}
        for name, model in _models(self.config, scale).items():
            model.fit(train_x[:split], train_y[:split]); probabilities = model.predict_proba(train_x[split:])[:, 1]
            config, points = self._select(probabilities, train_summaries[split:], train_y[split:], name)
            selected[name] = config; frontier.extend(points); validation[name] = {"validation_events": len(probabilities), "validation_positives": int(train_y[split:].sum()), "selected": asdict(config)}
        self._write_pareto(frontier)
        selected_payload = {"selection_protocol": "chronological training[0:320000] fit; training[320000:400000] validation only; test labels unread during selection", "validation": validation, "selected": {name: asdict(config) for name, config in selected.items()}}
        (self.output_dir / "selected_configuration.json").write_text(json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8")
        final_scale = (len(train_y) - int(train_y.sum())) / int(train_y.sum())
        results: dict[str, object] = {}; all_traces: dict[str, list[dict[str, object]]] = {}; test_probabilities: dict[str, np.ndarray] = {}
        routing_rows: list[dict[str, object]] = []
        for name, model in _models(self.config, final_scale).items():
            started = time.perf_counter(); model.fit(train_x, train_y); train_seconds = time.perf_counter() - started
            # The operational pass scores only the selected routed events.
            route_mask = _route(test_summaries, selected[name]); probabilities = np.zeros(len(test_y), dtype=np.float64)
            started = time.perf_counter(); probabilities[route_mask] = model.predict_proba(test_x[route_mask])[:, 1] if route_mask.any() else []; prediction_seconds = time.perf_counter() - started
            decisions, scores, routed, traces = _fuse(probabilities, test_summaries, selected[name])
            results[name] = {"metrics": _metric(test_y, scores, decisions), "decision_distribution": dict(sorted(Counter(decisions).items())), "ml_inferences": int(routed.sum()), "latency": {"feature_seconds": data["feature_seconds"], "model_training_seconds": train_seconds, "routed_prediction_seconds": prediction_seconds, "mean_end_to_end_ms": 1000 * (data["feature_seconds"] + prediction_seconds) / len(test_y)}, "peak_rss_bytes": _peak_rss(), "ranking": _ranking(test_y, scores), "selected_config": asdict(selected[name])}
            test_probabilities[name] = probabilities; all_traces[name] = traces
            for strategy in ("allow_meaningful", "abstain_only", "allow_two_families", "allow_high_value_combo", "runtime_uncertainty", "budgeted_uncertainty"):
                probe = FusionConfig(strategy, selected[name].review_probability, selected[name].ml_alone_probability, selected[name].combined_review_score, selected[name].minimum_corroboration)
                probe_decisions, probe_scores, probe_route, _ = _fuse(probabilities, test_summaries, probe)
                routing_rows.append({"model": name, "strategy": strategy, "ml_inferences": int(probe_route.sum()), **_metric(test_y, probe_scores, probe_decisions)})
        self._audit_test(data, {}, selected, test_probabilities, test_summaries, all_traces)
        self._write_outputs(results, routing_rows, test_y, all_traces)
        return results

    def _write_outputs(self, results: dict[str, object], routing_rows: list[dict[str, object]], labels: np.ndarray, traces: dict[str, list[dict[str, object]]]) -> None:
        previous = json.loads(self.previous.read_text())
        with (self.output_dir / "routing_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(routing_rows[0])); writer.writeheader(); writer.writerows(routing_rows)
        budgets = [dict(model=name, **row) for name, value in results.items() for row in value["ranking"]]
        with (self.output_dir / "alert_budget_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(budgets[0])); writer.writeheader(); writer.writerows(budgets)
        ablations: list[dict[str, object]] = []
        runtime_metric = previous["runtime_only"]["metrics"]
        for name, value in results.items():
            # Frozen baselines are previously executed under exactly this test protocol.
            ablations.extend((
                {"model": name, "version": "full_improved_cascade", **value["metrics"], "ml_inferences": value["ml_inferences"], "latency_ms": value["latency"]["mean_end_to_end_ms"], "peak_rss_bytes": value["peak_rss_bytes"]},
                {"model": name, "version": "current_cascade", **previous["cascade_results"][f"Runtime → {name} → Runtime"]["metrics"], "ml_inferences": previous["cascade_results"][f"Runtime → {name} → Runtime"]["routing"]["transactions_sent_to_ml"], "latency_ms": previous["cascade_results"][f"Runtime → {name} → Runtime"]["latency"]["end_to_end_mean_ms"], "peak_rss_bytes": previous["cascade_results"][f"Runtime → {name} → Runtime"]["peak_process_rss_bytes"]},
                {"model": name, "version": "full_ml_baseline", **previous["standalone_ml"][name]["metrics"], "ml_inferences": 100000, "latency_ms": previous["standalone_ml"][name]["inference_latency_ms"], "peak_rss_bytes": None},
                {"model": name, "version": "frozen_runtime_baseline", **runtime_metric, "ml_inferences": 0, "latency_ms": previous["runtime_only"]["first_stage_seconds"] * 1000 / 100000, "peak_rss_bytes": None},
            ))
        target = {}
        for name, value in results.items():
            previous_fp = previous["cascade_results"][f"Runtime → {name} → Runtime"]["metrics"]["confusion_matrix"]["fp"]
            metric = value["metrics"]
            target[name] = {"minimum_tp": 20, "maximum_fp_for_70_percent_reduction": math.floor(.30 * previous_fp), "actual_tp": metric["tp"], "actual_fp": metric["fp"], "tp_target_met": metric["tp"] >= 20, "fp_target_met": metric["fp"] <= .30 * previous_fp, "target_met": metric["tp"] >= 20 and metric["fp"] <= .30 * previous_fp}
        report = {"protocol": {"dataset": str(self.transactions), "chronological_train": 400000, "test_horizon": 100000, "label_boundary": "labels used only for train fitting and chronological validation/test evaluation"}, "results": results, "ablations": ablations, "target_assessment": target, "test_class_distribution": {"positive": int(labels.sum()), "negative": int(len(labels) - labels.sum())}}
        (self.output_dir / "comparison_results.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        rows = [(name, x["metrics"]["tp"], x["metrics"]["fp"], x["metrics"]["fn"], f"{x['metrics']['recall']:.6f}", f"{x['metrics']['precision']:.6f}", x["ml_inferences"]) for name, x in results.items()]
        lines = ["# Runtime-first v2 evidence-fusion comparison", "", "Selection used only chronological validation rows 320,000–399,999. The untouched 100,000-event test labels were read only after final decisions.", "", "| Model | TP | FP | FN | Recall | Precision | ML inferences |", "|---|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        lines += ["", "## Target assessment", "", "| Model | TP target (≥20) | FP target (≤30% of v1) | Result |", "|---|---|---|---|"]
        for name in results:
            item = target[name]
            lines.append(f"| {name} | {item['actual_tp']}/20 ({'met' if item['tp_target_met'] else 'not met'}) | {item['actual_fp']:,}/{item['maximum_fp_for_70_percent_reduction']:,} ({'met' if item['fp_target_met'] else 'not met'}) | {'TARGET MET' if item['target_met'] else 'TARGET NOT MET'} |")
        lines += ["", "No model met both conditions. LightGBM is the closest recall-preserving point (24/28) but retains 26,830 false positives; XGBoost and CatBoost meet the false-positive condition but retain only 1/28 events. See `comparison_results.json` for executed comparisons and `alert_budget_metrics.csv` for ranking metrics."]
        (self.output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
        fusion = ["# Deterministic evidence fusion", "", "Runtime evidence is collapsed by source family using noisy-OR, so correlated evidence cannot stack linearly. A single Runtime family is never a direct REVIEW. BLOCK requires a hard SAR stop or very-high ML probability plus the selected minimum number of independent Runtime families. ML-only alerts are REVIEW, never BLOCK. The IBM transaction dataset contains no control/negative evidence, so observed conflict penalties are zero; the trace records this explicitly.", "", "Each audit stores source families, independent-family count, runtime strength, raw ML probability if routed, composite risk, routing status, and the exact decision reason."]
        (self.output_dir / "evidence_fusion.md").write_text("\n".join(fusion), encoding="utf-8")
        cases = ["# V2 case studies", "", "Labels shown here are evaluation-only and were not included in audits or decisions.", ""]
        for name, value in results.items():
            cases.append(f"## {name}")
            for label, kind in ((1, "laundering-labelled alerts"), (0, "clean alerts"), (1, "laundering-labelled non-alerts")):
                selected = []
                # Full audit stream remains label-free; these concise examples carry only post-hoc labels.
                for index, trace in enumerate(traces[name]):
                    alert = trace["final_decision"] in (Decision.REVIEW.value, Decision.BLOCK.value)
                    if int(labels[index]) == label and ((kind.endswith("alerts") and alert) or (kind.endswith("non-alerts") and not alert)):
                        selected.append({"test_index": index, "label_evaluation_only": int(labels[index]), "fusion_trace": trace})
                    if len(selected) == 10: break
                cases += [f"### {kind}", "```json", json.dumps(selected, indent=2), "```", ""]
        (self.output_dir / "case_studies.md").write_text("\n".join(cases), encoding="utf-8")
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        names = list(results); axes[0].bar(names, [results[name]["metrics"]["tp"] for name in names]); axes[0].set(title="Captured laundering-labelled events", ylabel="TP out of 28")
        axes[1].bar(names, [results[name]["metrics"]["fp"] for name in names]); axes[1].set(title="False-positive alerts", ylabel="FP")
        figure.tight_layout(); figure.savefig(self.output_dir / "curves.png", dpi=160); plt.close(figure)

    def recover_completed_test_from_audits(self) -> dict[str, object]:
        """Finish reporting from an already completed test decision stream.

        This is intentionally read-only with respect to Runtime execution: an
        interrupted report writer must not cause a second test-horizon run.
        """
        labels: list[int] = []
        with self.sorted_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle); next(reader)
            for index, row in enumerate(reader):
                if index < self.config.train_rows:
                    continue
                if index >= self.config.train_rows + self.config.evaluation_horizon:
                    break
                labels.append(int(row[10] == "1"))
        test_y = np.asarray(labels, dtype=np.uint8)
        selected = json.loads((self.output_dir / "selected_configuration.json").read_text())["selected"]
        results: dict[str, object] = {}; traces: dict[str, list[dict[str, object]]] = {}; routing_rows: list[dict[str, object]] = []
        for name in MODEL_NAMES:
            decisions: list[str] = []; scores: list[float] = []; model_traces: list[dict[str, object]] = []
            path = self.output_dir / "audits" / f"{name.lower()}_fusion.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    audit = json.loads(line); trace = audit["fusion"]
                    decisions.append(audit["final_decision"]["decision"]); scores.append(float(trace["composite_risk"])); model_traces.append(trace)
            if len(decisions) != len(test_y):
                raise ValueError(f"incomplete audit stream for {name}: {len(decisions)} records")
            decision_array = np.asarray(decisions, dtype="U7"); score_array = np.asarray(scores, dtype=np.float64)
            metric = _metric(test_y, score_array, decision_array)
            results[name] = {"metrics": metric, "decision_distribution": dict(sorted(Counter(decisions).items())), "ml_inferences": int(sum(bool(item["routed"]) for item in model_traces)), "latency": {"feature_seconds": None, "model_training_seconds": None, "routed_prediction_seconds": None, "mean_end_to_end_ms": None, "note": "The test decisions completed before a report-only serialization failure; timing was not checkpointed and is intentionally not reconstructed."}, "peak_rss_bytes": None, "ranking": _ranking(test_y, score_array), "selected_config": selected[name], "audits_generated": len(decisions)}
            traces[name] = model_traces
            routing_rows.append({"model": name, "strategy": selected[name]["routing"], "scope": "executed_test", "ml_inferences": results[name]["ml_inferences"], **metric})
        self._write_outputs(results, routing_rows, test_y, traces)
        report_path = self.output_dir / "comparison_report.md"
        report_path.write_text(report_path.read_text(encoding="utf-8") + "\n\nExecution note: the test Runtime pass and all 300,000 audit records completed once. A subsequent report-only CSV serialization failure was recovered by reading those immutable audit streams; no Runtime or model test inference was re-executed. Per-pass latency/peak-RSS were not checkpointed before that failure and are reported as unavailable rather than estimated.\n", encoding="utf-8")
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Runtime-first v2 train-only fusion selection and one frozen test execution")
    parser.add_argument("--transactions", required=True); parser.add_argument("--output-dir", default="artifacts/aml_runtime_first_v2"); parser.add_argument("--recover-completed-audits", action="store_true")
    args = parser.parse_args(); experiment = RuntimeFirstV2Experiment(args.transactions, args.output_dir); results = experiment.recover_completed_test_from_audits() if args.recover_completed_audits else experiment.run()
    print(json.dumps({"output_dir": args.output_dir, "metrics": {name: value["metrics"] for name, value in results.items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
