"""Four-arm comparison: does behavioural reasoning change the decision?

Arms, on the frozen protocol used by every earlier experiment (first 500,000
chronological IBM AML `HI-Small` events; 400,000 train; the next 100,000 as the
evaluation horizon):

1. Runtime                          -- frozen v0.2 transaction-level rules
2. Semantic Runtime                 -- the Semantic Context Layer, unchanged
3. Semantic Behaviour Runtime       -- + the Behaviour Layer
4. Semantic Behaviour Runtime + ML  -- ML over the *semantic feature space*

The model receives no transaction column: its features are behaviour objects,
role objects, scenario objects, lifecycle objects and semantic objects.

Nothing is tuned.  Every constant lives in `semantic/ontology.py` and
`behaviour/ontology.py`, was declared before this module ran, and the frozen
`SemanticPolicyEngine` selects every decision in arms 2-4.
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
from collections import Counter, defaultdict
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

from .behaviour import (
    BEHAVIOUR_ONTOLOGY_HASH,
    BEHAVIOUR_ONTOLOGY_VERSION,
    SEMANTIC_FEATURE_NAMES,
    BehaviourDecisionRuntime,
    BehaviourLayer,
    behaviour_replay_pins,
    semantic_feature_vector,
)
from .behaviour.runtime import BEHAVIOUR_CONFLICT_PAIRS
from .dataset import AMLSimDataset
from .ml_benchmark import StreamingRuntimeGraph, _account, _amount, _canonical_bank
from .models import Transaction
from .runtime import AMLDecisionRuntime
from .runtime_first_cascade import CascadeConfig, _EphemeralAccounts, _models, _sha256
from .semantic import ONTOLOGY_HASH, ONTOLOGY_VERSION, EntityResolver, SemanticContextLayer, SemanticDecisionRuntime, account_key
from .semantic.runtime import SEMANTIC_CONFLICT_PAIRS
from .semantic_benchmark import ML_HIGH_BAND, ML_STANDALONE_THRESHOLD, _alerts, _ml_evidence

BEHAVIOUR_BENCHMARK_VERSION = "aml-behaviour-benchmark/1.0"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost")

#: risk rule -> the declared mitigating rule that would qualify it.  Used to
#: answer "could a missing behaviour object have prevented this alert?".
_QUALIFIERS: dict[str, list[str]] = defaultdict(list)
for _pair in SEMANTIC_CONFLICT_PAIRS + BEHAVIOUR_CONFLICT_PAIRS:
    _QUALIFIERS[_pair.risk_rule_id].append(_pair.mitigating_rule_id)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


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


class BehaviourBenchmark:
    def __init__(self, transactions: str | Path, accounts: str | Path, output_dir: str | Path, config: CascadeConfig | None = None) -> None:
        self.transactions = Path(transactions)
        self.accounts = Path(accounts)
        self.output_dir = Path(output_dir)
        self.config = config or CascadeConfig()
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")

    # -- input ------------------------------------------------------------
    def _rows(self):
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index >= self.config.input_row_cap:
                    return
                yield index, row

    @staticmethod
    def _event(index: int, row: list[str]) -> tuple[Transaction, int]:
        timestamp = datetime.strptime(row[0], TIMESTAMP_FORMAT)
        transaction = Transaction(
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
        return transaction, int(timestamp.timestamp()) // 60

    def _runtimes(self, resolver: EntityResolver) -> BehaviourDecisionRuntime:
        context = SemanticContextLayer(resolver)
        return BehaviourDecisionRuntime(SemanticDecisionRuntime(context), BehaviourLayer())

    # -- pass A: the semantic feature space --------------------------------
    def _feature_pass(self, resolver: EntityResolver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_rows, test_rows = self.config.train_rows, self.config.evaluation_horizon
        train_x = np.zeros((train_rows, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        test_x = np.zeros((test_rows, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        train_y = np.zeros(train_rows, dtype=np.uint8)
        test_y = np.zeros(test_rows, dtype=np.uint8)
        runtime = self._runtimes(resolver)
        scratch = np.zeros(len(SEMANTIC_FEATURE_NAMES), dtype=np.float32)
        for index, row in self._rows():
            transaction, minute = self._event(index, row)
            originator = resolver.resolve(transaction.originator_account_id)
            result = runtime.evaluate(transaction, originator, minute, index)
            metrics = result.policies[0].metrics
            vector = semantic_feature_vector(
                {item.type: item.confidence for item in result.semantic.context.objects},
                result.behaviour, result.evidence, len(result.conflicts),
                int(metrics["independent_unqualified_topics"]), float(metrics["effective_risk"]), scratch,
            )
            if index < train_rows:
                train_x[index] = vector
                train_y[index] = int(row[10] == "1")
            else:
                test_x[index - train_rows] = vector
                test_y[index - train_rows] = int(row[10] == "1")
            runtime.commit(transaction, result, minute)
        del runtime
        gc.collect()
        return train_x, train_y, test_x, test_y

    # -- pass B: the decisions ---------------------------------------------
    def _decision_pass(self, resolver: EntityResolver, probabilities: dict[str, np.ndarray]) -> dict[str, object]:
        fingerprint = _sha256(f"{self.transactions.resolve()}:{self.transactions.stat().st_size}")
        dataset = AMLSimDataset((), _EphemeralAccounts(), {}, str(self.transactions), fingerprint)
        frozen = AMLDecisionRuntime(dataset, self.output_dir / "frozen_runtime_audits_unused")
        graph = StreamingRuntimeGraph(dataset)
        frozen.graph = graph
        runtime = self._runtimes(resolver)

        train_rows, horizon = self.config.train_rows, self.config.evaluation_horizon
        runtime_decisions: list[str] = []
        semantic_decisions: list[str] = []
        behaviour_decisions: list[str] = []
        runtime_scores = np.zeros(horizon, dtype=np.float64)
        semantic_scores = np.zeros(horizon, dtype=np.float64)
        behaviour_scores = np.zeros(horizon, dtype=np.float64)
        behaviour_ml_decisions = {name: [] for name in MODEL_NAMES}
        behaviour_ml_scores = {name: np.zeros(horizon, dtype=np.float64) for name in MODEL_NAMES}
        routed = {name: 0 for name in MODEL_NAMES}

        behaviour_census: Counter[str] = Counter()
        role_census: Counter[str] = Counter()
        scenario_census: Counter[str] = Counter()
        transition_census: Counter[str] = Counter()
        conflict_census: Counter[str] = Counter()
        alert_signatures: Counter[tuple[str, ...]] = Counter()
        alert_signature_labels: Counter[tuple[str, ...]] = Counter()
        positives: list[dict[str, object]] = []
        examples: list[dict[str, object]] = []
        behaviour_objects_total = 0
        transitions_total = 0
        scenarios_total = 0

        audits = self.output_dir / "audits"
        audits.mkdir(parents=True, exist_ok=True)
        behaviour_seconds = frozen_seconds = 0.0

        with gzip.GzipFile(filename="", mode="wb", fileobj=(audits / "behaviour_decisions.jsonl.gz").open("wb"), mtime=0) as stream:
            for index, row in self._rows():
                transaction, minute = self._event(index, row)
                originator = resolver.resolve(transaction.originator_account_id)
                if index < train_rows:
                    graph.commit(transaction)
                    result = runtime.evaluate(transaction, originator, minute, index)
                    runtime.commit(transaction, result, minute)
                    continue
                position = index - train_rows

                started = time.perf_counter()
                graph.by_id[transaction.id] = transaction
                preliminary = frozen.evaluate_preliminary(transaction.id)
                frozen_seconds += time.perf_counter() - started
                runtime_decisions.append(preliminary.decision.decision.value)
                runtime_scores[position] = float(preliminary.policies[0].metrics["effective_risk"])

                started = time.perf_counter()
                result = runtime.evaluate(transaction, originator, minute, index)
                behaviour_seconds += time.perf_counter() - started

                semantic_decisions.append(result.semantic.decision.decision.value)
                semantic_scores[position] = float(result.semantic.policies[0].metrics["effective_risk"])
                behaviour_decisions.append(result.decision.decision.value)
                behaviour_scores[position] = float(result.policies[0].metrics["effective_risk"])

                behaviour_types = sorted(item.type.value for item in result.behaviour.behaviours)
                scenario_types = sorted(item.type.value for item in result.behaviour.scenarios)
                behaviour_census.update(behaviour_types)
                role_census[result.behaviour.role.role.value] += 1
                scenario_census.update(scenario_types)
                behaviour_objects_total += len(result.behaviour.behaviours)
                scenarios_total += len(result.behaviour.scenarios)
                if result.behaviour.transition:
                    transitions_total += 1
                    transition_census[f"{result.behaviour.transition.from_role.value}->{result.behaviour.transition.to_role.value}"] += 1
                conflict_census.update(item.kind for item in result.conflicts)

                qualified = {item.risk_evidence_id for item in result.conflicts}
                signature = tuple(sorted(item.rule_id for item in result.evidence
                                         if item.direction == "risk" and item.id not in qualified))
                label = int(row[10] == "1")
                alert = result.decision.decision.value in ("REVIEW", "BLOCK")
                if alert:
                    alert_signatures[signature] += 1
                    if label:
                        alert_signature_labels[signature] += 1

                route = BehaviourDecisionRuntime.routes_to_ml(result)
                for name in MODEL_NAMES:
                    probability = float(probabilities[name][position])
                    if route:
                        routed[name] += 1
                        fused = runtime.with_ml_evidence(result, _ml_evidence(transaction, name, probability), ML_HIGH_BAND)
                        behaviour_ml_decisions[name].append(fused.decision.decision.value)
                        behaviour_ml_scores[name][position] = max(behaviour_scores[position], probability)
                    else:
                        behaviour_ml_decisions[name].append(result.decision.decision.value)
                        behaviour_ml_scores[name][position] = behaviour_scores[position]

                stream.write((json.dumps({
                    "transaction_id": transaction.id,
                    "frozen_runtime_decision": preliminary.decision.decision.value,
                    "semantic_decision": result.semantic.decision.decision.value,
                    "behaviour_decision": result.decision.decision.value,
                    "role": result.behaviour.role.role.value,
                    "role_transition": (f"{result.behaviour.transition.from_role.value}->{result.behaviour.transition.to_role.value}"
                                        if result.behaviour.transition else ""),
                    "stage": result.behaviour.stage.value,
                    "behaviours": behaviour_types,
                    "scenarios": scenario_types,
                    "semantic_objects": sorted(item.type.value for item in result.semantic.context.objects),
                    "unqualified_risk_rules": list(signature),
                    "conflict_kinds": sorted(item.kind for item in result.conflicts),
                    "routed_to_ml": route,
                }, sort_keys=True, separators=(",", ":")) + "\n").encode())

                if label:
                    record = result.audit_record()
                    record["routed_to_ml"] = route
                    record["replay_pins"] = behaviour_replay_pins(resolver.snapshot_hash, result)
                    record["laundering_label_evaluation_only"] = 1
                    record["frozen_runtime_decision"] = preliminary.decision.decision.value
                    record["semantic_runtime_decision"] = result.semantic.decision.decision.value
                    positives.append(record)
                elif len(examples) < 16 and position % 1213 == 0:
                    record = result.audit_record()
                    record["routed_to_ml"] = route
                    record["replay_pins"] = behaviour_replay_pins(resolver.snapshot_hash, result)
                    record["laundering_label_evaluation_only"] = 0
                    record["frozen_runtime_decision"] = preliminary.decision.decision.value
                    record["semantic_runtime_decision"] = result.semantic.decision.decision.value
                    examples.append(record)

                graph.commit(transaction)
                del graph.by_id[transaction.id]
                runtime.commit(transaction, result, minute)

        return {
            "runtime_decisions": runtime_decisions, "runtime_scores": runtime_scores,
            "semantic_decisions": semantic_decisions, "semantic_scores": semantic_scores,
            "behaviour_decisions": behaviour_decisions, "behaviour_scores": behaviour_scores,
            "behaviour_ml_decisions": behaviour_ml_decisions, "behaviour_ml_scores": behaviour_ml_scores,
            "routed": routed, "behaviour_census": behaviour_census, "role_census": role_census,
            "scenario_census": scenario_census, "transition_census": transition_census,
            "conflict_census": conflict_census, "alert_signatures": alert_signatures,
            "alert_signature_labels": alert_signature_labels, "positives": positives, "examples": examples,
            "behaviour_objects_total": behaviour_objects_total, "transitions_total": transitions_total,
            "scenarios_total": scenarios_total, "frozen_seconds": frozen_seconds,
            "behaviour_seconds": behaviour_seconds,
        }

    # -- orchestration ------------------------------------------------------
    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        keys: set[str] = set()
        for _index, row in self._rows():
            keys.add(account_key(row[1], row[2]))
            keys.add(account_key(row[3], row[4]))
        resolver = EntityResolver().load(self.accounts, restrict_to=keys)
        del keys
        gc.collect()
        resolution_seconds = time.perf_counter() - started

        started = time.perf_counter()
        train_x, train_y, test_x, test_y = self._feature_pass(resolver)
        feature_seconds = time.perf_counter() - started

        train_positives = int(train_y.sum())
        scale = (len(train_y) - train_positives) / train_positives
        probabilities: dict[str, np.ndarray] = {}
        model_results: dict[str, dict[str, object]] = {}
        importances: dict[str, list[tuple[str, float]]] = {}
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
            model_results[name] = {
                "metrics": _metrics(test_y, probability, (probability >= ML_STANDALONE_THRESHOLD).astype(np.uint8)),
                "training_seconds": train_seconds,
                "prediction_latency_ms": 1000 * predict_seconds / len(test_y),
            }
            weights = getattr(model, "feature_importances_", None)
            if weights is not None:
                ranked = sorted(zip(SEMANTIC_FEATURE_NAMES, (float(value) for value in weights)), key=lambda item: -item[1])
                importances[name] = ranked[:15]
            del model
            gc.collect()
        del train_x, test_x, models
        gc.collect()

        outcome = self._decision_pass(resolver, probabilities)
        results = self._assemble(test_y, probabilities, model_results, importances, outcome, resolver,
                                 feature_seconds, resolution_seconds, train_positives)
        self._write(results, test_y, probabilities, outcome)
        return results

    def _assemble(self, labels, probabilities, model_results, importances, outcome, resolver, feature_seconds, resolution_seconds, train_positives) -> dict[str, object]:
        arms: dict[str, object] = {}
        for key, decisions, scores in (
            ("runtime_only", outcome["runtime_decisions"], outcome["runtime_scores"]),
            ("semantic_runtime", outcome["semantic_decisions"], outcome["semantic_scores"]),
            ("semantic_behaviour_runtime", outcome["behaviour_decisions"], outcome["behaviour_scores"]),
        ):
            predicted = _alerts(decisions)
            arms[key] = {
                "metrics": _metrics(labels, scores, predicted),
                "decision_distribution": dict(sorted(Counter(decisions).items())),
                "ml_inferences": 0,
            }
        for name in MODEL_NAMES:
            predicted = _alerts(outcome["behaviour_ml_decisions"][name])
            arms[f"semantic_behaviour_runtime_plus_ml/{name}"] = {
                "metrics": _metrics(labels, outcome["behaviour_ml_scores"][name], predicted),
                "decision_distribution": dict(sorted(Counter(outcome["behaviour_ml_decisions"][name]).items())),
                "ml_inferences": int(outcome["routed"][name]),
                "ml_inference_fraction": outcome["routed"][name] / len(labels),
                "semantic_feature_model": model_results[name],
            }
        return {
            "benchmark_version": BEHAVIOUR_BENCHMARK_VERSION,
            "ontology": {
                "semantic_version": ONTOLOGY_VERSION, "semantic_hash": ONTOLOGY_HASH,
                "behaviour_version": BEHAVIOUR_ONTOLOGY_VERSION, "behaviour_hash": BEHAVIOUR_ONTOLOGY_HASH,
                "entity_snapshot_hash": resolver.snapshot_hash, "resolved_accounts": resolver.resolved_count,
            },
            "protocol": {
                "transactions": str(self.transactions), "accounts": str(self.accounts),
                "input_row_cap": self.config.input_row_cap, "train_rows": self.config.train_rows,
                "evaluation_horizon": self.config.evaluation_horizon,
                "ml_feature_space": "semantic: behaviour, role, scenario, lifecycle and semantic objects only; no transaction column reaches the model",
                "ml_feature_count": len(SEMANTIC_FEATURE_NAMES),
                "ml_standalone_threshold": ML_STANDALONE_THRESHOLD, "ml_high_band": ML_HIGH_BAND,
                "label_boundary": "the laundering column is read into a separate array used for model fitting and post-decision evaluation only; no semantic object, behaviour object, role, scenario, evidence item, policy or audit reads it",
                "tuning_statement": "every semantic and behavioural constant was declared before this benchmark executed; no threshold was selected against the evaluation labels and none was changed after seeing a result",
            },
            "class_distribution": {"train_positive": train_positives, "test_positive": int(labels.sum()),
                                   "test_negative": int(len(labels) - labels.sum())},
            "arms": arms,
            "behaviour_counts": {
                "behaviour_objects_generated": outcome["behaviour_objects_total"],
                "role_transitions": outcome["transitions_total"],
                "scenario_detections": outcome["scenarios_total"],
                "conflicts": sum(outcome["conflict_census"].values()),
            },
            "behaviour_census": dict(outcome["behaviour_census"].most_common()),
            "role_census": dict(outcome["role_census"].most_common()),
            "scenario_census": dict(outcome["scenario_census"].most_common()),
            "role_transition_census": dict(outcome["transition_census"].most_common()),
            "conflict_census": dict(outcome["conflict_census"].most_common()),
            "semantic_feature_importances": importances,
            "timing": {"entity_resolution_seconds": resolution_seconds,
                       "semantic_feature_pass_seconds": feature_seconds,
                       "frozen_runtime_seconds": outcome["frozen_seconds"],
                       "behaviour_runtime_seconds": outcome["behaviour_seconds"]},
            "peak_rss_bytes": _peak_rss_bytes(),
        }

    # -- reporting ----------------------------------------------------------
    def _write(self, results: dict[str, object], labels: np.ndarray, probabilities: dict[str, np.ndarray], outcome: dict[str, object]) -> None:
        (self.output_dir / "comparison_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        samples = self.output_dir / "audits" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        for record in outcome["positives"] + outcome["examples"]:
            (samples / f"{record['transaction']['id']}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        self._write_benchmark_report(results, outcome)
        self._write_comparison_report(results, outcome, labels)
        self._write_case_studies(results, outcome)
        self._write_curves(labels, probabilities, outcome)

    def _write_curves(self, labels: np.ndarray, probabilities: dict[str, np.ndarray], outcome: dict[str, object]) -> None:
        curves = {"Frozen Runtime": outcome["runtime_scores"], "Semantic Runtime": outcome["semantic_scores"],
                  "Semantic Behaviour Runtime": outcome["behaviour_scores"]}
        for name in MODEL_NAMES:
            curves[f"Behaviour + {name}"] = outcome["behaviour_ml_scores"][name]
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

    @staticmethod
    def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
        return (["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
                + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])

    def _write_benchmark_report(self, results: dict[str, object], outcome: dict[str, object]) -> None:
        arms = results["arms"]
        rows = [(key, m["tp"], m["fp"], m["fn"], f"{m['precision']:.6f}", f"{m['recall']:.4f}",
                 f"{m['f1']:.6f}", m["alert_volume"], f"{m['alert_rate']:.4f}", arms[key].get("ml_inferences", 0))
                for key in sorted(arms) for m in (arms[key]["metrics"],)]
        counts = results["behaviour_counts"]
        lines = [
            "# Semantic Behaviour Runtime — benchmark report", "",
            f"Protocol: first {results['protocol']['input_row_cap']:,} chronological events; "
            f"{results['protocol']['train_rows']:,} train; {results['protocol']['evaluation_horizon']:,}-event horizon "
            f"with {results['class_distribution']['test_positive']} laundering-labelled events.", "",
            f"Semantic ontology `{results['ontology']['semantic_hash'][:16]}…`; "
            f"behaviour ontology `{results['ontology']['behaviour_hash'][:16]}…`.", "",
            results["protocol"]["tuning_statement"].capitalize() + ".", "",
            "## Arms", "",
            *self._table(("Arm", "TP", "FP", "FN", "Precision", "Recall", "F1", "Alerts", "Alert rate", "ML inferences"), rows), "",
            "## Behaviour layer output", "",
            *self._table(("Quantity", "Count"), [
                ("Behaviour objects generated", f"{counts['behaviour_objects_generated']:,}"),
                ("Role transitions", f"{counts['role_transitions']:,}"),
                ("Scenario detections", f"{counts['scenario_detections']:,}"),
                ("Conflicts", f"{counts['conflicts']:,}"),
            ]), "",
            "## Behaviour census", "",
            *self._table(("Behaviour", "Emissions"), [(key, f"{value:,}") for key, value in results["behaviour_census"].items()]), "",
            "## Role census", "",
            *self._table(("Role", "Events"), [(key, f"{value:,}") for key, value in results["role_census"].items()]), "",
            "## Role transitions", "",
            *self._table(("Transition", "Count"), [(key, f"{value:,}") for key, value in results["role_transition_census"].items()]), "",
            "## Scenario census", "",
            *self._table(("Scenario", "Detections"), [(key, f"{value:,}") for key, value in results["scenario_census"].items()]), "",
            "## Conflicts", "",
            *self._table(("Conflict kind", "Count"), [(key, f"{value:,}") for key, value in results["conflict_census"].items()]), "",
            "## Semantic feature importances", "",
            "The model sees only behaviour, role, scenario, lifecycle and semantic objects.", "",
        ]
        for name, ranked in results["semantic_feature_importances"].items():
            lines += [f"### {name}", "", *self._table(("Feature", "Importance"), [(item[0], f"{item[1]:.5f}") for item in ranked]), ""]
        lines += ["## Cost", "", *self._table(("Stage", "Seconds"), [(key, f"{value:.2f}") for key, value in results["timing"].items()]),
                  "", f"Peak process RSS: {results['peak_rss_bytes']:,} bytes.", ""]
        (self.output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_comparison_report(self, results: dict[str, object], outcome: dict[str, object], labels: np.ndarray) -> None:
        arms = results["arms"]
        base = arms["runtime_only"]["metrics"]
        semantic = arms["semantic_runtime"]["metrics"]
        behaviour = arms["semantic_behaviour_runtime"]["metrics"]

        def delta(before: dict, after: dict, field: str) -> str:
            if before[field] == 0:
                return "n/a"
            return f"{(after[field] - before[field]) / before[field]:+.1%}"

        movement = Counter(zip(outcome["semantic_decisions"], outcome["behaviour_decisions"]))
        moved = [(f"{a} -> {b}", count) for (a, b), count in movement.most_common() if a != b]

        # False-positive causation: which unqualified risk rule set produced it,
        # and which declared mitigating rule would have qualified it.
        fp_rows = []
        for signature, count in outcome["alert_signatures"].most_common(20):
            true_positives = outcome["alert_signature_labels"].get(signature, 0)
            preventable = sorted({qualifier for rule in signature for qualifier in _QUALIFIERS.get(rule, [])})
            fp_rows.append((
                " + ".join(signature) or "(none)", count - true_positives, true_positives,
                ", ".join(preventable) or "no declared qualifier exists",
            ))

        lines = [
            "# Comparison report — rule-first vs semantic vs behavioural", "",
            "## Headline", "",
            *self._table(("Metric", "Runtime", "Semantic Runtime", "Semantic Behaviour Runtime", "Semantic vs Runtime", "Behaviour vs Semantic"), [
                ("True positives", base["tp"], semantic["tp"], behaviour["tp"], delta(base, semantic, "tp"), delta(semantic, behaviour, "tp")),
                ("False positives", f"{base['fp']:,}", f"{semantic['fp']:,}", f"{behaviour['fp']:,}", delta(base, semantic, "fp"), delta(semantic, behaviour, "fp")),
                ("False negatives", base["fn"], semantic["fn"], behaviour["fn"], delta(base, semantic, "fn"), delta(semantic, behaviour, "fn")),
                ("Precision", f"{base['precision']:.6f}", f"{semantic['precision']:.6f}", f"{behaviour['precision']:.6f}", delta(base, semantic, "precision"), delta(semantic, behaviour, "precision")),
                ("Recall", f"{base['recall']:.4f}", f"{semantic['recall']:.4f}", f"{behaviour['recall']:.4f}", delta(base, semantic, "recall"), delta(semantic, behaviour, "recall")),
                ("F1", f"{base['f1']:.6f}", f"{semantic['f1']:.6f}", f"{behaviour['f1']:.6f}", delta(base, semantic, "f1"), delta(semantic, behaviour, "f1")),
                ("Alert rate", f"{base['alert_rate']:.4f}", f"{semantic['alert_rate']:.4f}", f"{behaviour['alert_rate']:.4f}", delta(base, semantic, "alert_rate"), delta(semantic, behaviour, "alert_rate")),
            ]), "",
            "## What the Behaviour Layer changed relative to the Semantic Runtime", "",
            *self._table(("Decision movement", "Events"), [(key, f"{value:,}") for key, value in moved]), "",
            "## With ML over the semantic feature space", "",
            *self._table(("Model", "TP", "FP", "Recall", "Precision", "Alerts", "ML inferences", "ML share of stream"), [
                (name, arms[f"semantic_behaviour_runtime_plus_ml/{name}"]["metrics"]["tp"],
                 f"{arms[f'semantic_behaviour_runtime_plus_ml/{name}']['metrics']['fp']:,}",
                 f"{arms[f'semantic_behaviour_runtime_plus_ml/{name}']['metrics']['recall']:.4f}",
                 f"{arms[f'semantic_behaviour_runtime_plus_ml/{name}']['metrics']['precision']:.6f}",
                 f"{arms[f'semantic_behaviour_runtime_plus_ml/{name}']['metrics']['alert_volume']:,}",
                 f"{arms[f'semantic_behaviour_runtime_plus_ml/{name}']['ml_inferences']:,}",
                 f"{arms[f'semantic_behaviour_runtime_plus_ml/{name}']['ml_inference_fraction']:.1%}")
                for name in MODEL_NAMES
            ]), "",
            "## False-positive causation", "",
            "Each row is a distinct set of *unqualified* risk evidence that produced alerts, the false and true "
            "positives it produced, and the mitigating behaviour or semantic object that the catalog already declares "
            "as its qualifier. Where the qualifier column names an object, the alert would not have fired had that "
            "object been inferable from this window; where it says none exists, the catalog has a gap.", "",
            *self._table(("Unqualified risk evidence", "False positives", "True positives", "Declared qualifier that would have prevented it"), fp_rows), "",
        ]
        (self.output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_case_studies(self, results: dict[str, object], outcome: dict[str, object]) -> None:
        lines = [
            "# Case studies", "",
            "Labels are evaluation-only and were read after every decision was made.", "",
            "## Every laundering-labelled event in the evaluation horizon", "",
        ]
        for record in outcome["positives"]:
            behaviour = record["behaviour"]
            semantic = record["semantic"]
            decision = record["decision"]["decision"]
            lines += [
                f"### {record['transaction']['id']} — behaviour runtime: **{decision}**", "",
                f"- Frozen Runtime said: `{record['frozen_runtime_decision']}`; "
                f"Semantic Runtime said: `{record['semantic_runtime_decision']}`",
                f"- **Behaviour objects**: {', '.join(item['type'] for item in behaviour['behaviours']) or 'none claimed'}",
                f"- **Scenario objects**: {', '.join(item['type'] for item in behaviour['scenarios']) or 'none matched'}",
                f"- **Role**: {behaviour['role']['role']} (tenure {behaviour['role']['tenure_minutes']} min, "
                f"{behaviour['role']['transition_count']} transitions)"
                + (f"; transition this event: {behaviour['transition']['from_role']} -> {behaviour['transition']['to_role']}"
                   if behaviour.get("transition") else ""),
                f"- **Semantic objects**: {', '.join(item['type'] for item in semantic['objects'])}",
                f"- **Lifecycle**: {behaviour['lifecycle']['observed_events']} observed events, "
                f"{behaviour['lifecycle']['distinct_counterparties']} counterparties, "
                f"age {behaviour['lifecycle']['age_minutes']} min, horizons filled "
                f"{behaviour['lifecycle']['horizons_filled'] or 'none'}",
                f"- **Why**: {record['behaviour_rationale']}",
                f"- **Routed to ML**: {record['routed_to_ml']}",
                "",
            ]
        lines += ["## Sampled negatives", ""]
        for record in outcome["examples"]:
            behaviour = record["behaviour"]
            lines += [
                f"### {record['transaction']['id']} — behaviour runtime: **{record['decision']['decision']}** "
                f"(frozen Runtime: `{record['frozen_runtime_decision']}`)", "",
                f"- Behaviour: {', '.join(item['type'] for item in behaviour['behaviours']) or 'none claimed'}",
                f"- Scenario: {', '.join(item['type'] for item in behaviour['scenarios']) or 'none matched'}",
                f"- Role: {behaviour['role']['role']}",
                f"- Why: {record['behaviour_rationale']}", "",
            ]
        (self.output_dir / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("data/ibm_aml_data/HI-Small_Trans.csv"))
    parser.add_argument("--accounts", type=Path, default=Path("data/ibm_aml_data/HI-Small_accounts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aml_behaviour_v1"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    results = BehaviourBenchmark(args.transactions, args.accounts, args.output_dir, CascadeConfig(threads=args.threads)).run()
    print(json.dumps({"output_dir": str(args.output_dir),
                      "arms": {key: value["metrics"] for key, value in results["arms"].items()},
                      "behaviour_counts": results["behaviour_counts"]},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
