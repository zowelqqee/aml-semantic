"""Does the Semantic Behaviour Layer improve by reasoning, or by having history?

A controlled scaling study.  **No reasoning code is modified, called differently,
or reconfigured.**  The ontologies, inference rules, projection, conflict pairs,
policy engine, routing rule and feature space are byte-identical to the ones the
four-arm benchmark executed.  Exactly one thing varies between runs: how much of
the stream the layers were allowed to fold in before evaluation begins.

Design
------
The evaluation set is held **constant**: the same 100,000 events, the same 253
laundering labels, in every window.  Only the priming window in front of it
changes.  Without this control, a longer window would also mean a different
evaluation set, and any movement in the metrics would be unattributable.

The frozen v0.2 rule runtime is carried through every window as a **control
arm**.  Its only history-dependent rule is a 24-hour velocity count, so it is
close to history-invariant by construction.  If the semantic and behavioural
arms move with the window while the control does not, the movement is caused by
history.  If the control moves too, the evaluation set is behaving differently
and the comparison is contaminated.  That contrast is the answer to the research
question, and it is measured rather than argued.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import resource
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .behaviour import (
    BEHAVIOUR_ONTOLOGY_HASH,
    SEMANTIC_FEATURE_NAMES,
    BehaviourDecisionRuntime,
    BehaviourLayer,
    BehaviourType,
    semantic_feature_vector,
)
from .behaviour_benchmark import MODEL_NAMES, _metrics
from .dataset import AMLSimDataset
from .ml_benchmark import StreamingRuntimeGraph, _account, _amount, _canonical_bank
from .models import Transaction
from .runtime import AMLDecisionRuntime
from .runtime_first_cascade import CascadeConfig, _EphemeralAccounts, _models, _sha256
from .semantic import ONTOLOGY_HASH, EntityResolver, SemanticContextLayer, SemanticDecisionRuntime, SemanticType
from .semantic_benchmark import ML_HIGH_BAND, ML_STANDALONE_THRESHOLD, _alerts, _ml_evidence

WINDOW_STUDY_VERSION = "aml-window-study/1.0"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

#: The dense period of IBM AML `HI-Small` ends here.  Rows beyond it (2022-09-11
#: to 09-18) total 1,108 events carrying 655 laundering labels — a generator
#: artefact of long-running patterns completing, with a 59% base rate.  Including
#: them would make the evaluation set unrepresentative, so the study stops here.
DENSE_PERIOD_END = 5_077_237
EVALUATION_ROWS = 100_000
EVALUATION_START = DENSE_PERIOD_END - EVALUATION_ROWS

#: The ML training partition is the most recent primed events, capped so that
#: model capacity is not itself a variable once the window is large enough.
ML_TRAIN_CAP = 400_000


@dataclass(frozen=True)
class Window:
    name: str
    label: str
    prime_start_row: int
    available: bool = True
    unavailable_reason: str = ""

    @property
    def prime_rows(self) -> int:
        return EVALUATION_START - self.prime_start_row


#: Row indices were resolved from the chronological file by timestamp; they are
#: pinned here so the study is reproducible without re-scanning.
WINDOWS: tuple[Window, ...] = (
    Window("A", "5.6 hours", 4_931_128),
    Window("B", "24 hours", 4_535_375),
    Window("C", "3 days", 3_487_438),
    Window("D", "7 days", 1_975_534),
    Window("E", "14 days", -1, False,
           "The source's dense period spans 2022-09-01 00:00 to 2022-09-10 23:59. "
           "Only 9 days 11 hours of history exist in front of the evaluation set, "
           "so a 14-day window cannot be constructed. Declared, not approximated."),
    Window("F", "maximum available (9 days 11 hours)", 0),
)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


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


class WindowStudy:
    def __init__(self, transactions: str | Path, accounts: str | Path, output_dir: str | Path, threads: int = 4) -> None:
        self.transactions = Path(transactions)
        self.accounts = Path(accounts)
        self.output_dir = Path(output_dir)
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")
        self.config = CascadeConfig(threads=threads)
        self.resolver: EntityResolver | None = None

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

    def _resolve_all(self) -> EntityResolver:
        if self.resolver is None:
            self.resolver = EntityResolver().load(self.accounts)
        return self.resolver

    @staticmethod
    def _runtimes(resolver: EntityResolver) -> BehaviourDecisionRuntime:
        return BehaviourDecisionRuntime(SemanticDecisionRuntime(SemanticContextLayer(resolver)), BehaviourLayer())

    @staticmethod
    def _prime(runtime: BehaviourDecisionRuntime, resolver: EntityResolver, transaction: Transaction, minute: int, index: int) -> None:
        """Fold an event into state without materialising a decision.

        This calls the *same* public methods in the *same* order as
        ``BehaviourDecisionRuntime.evaluate``/``commit``; it simply omits the
        fact, evidence, conflict and policy stages, none of which mutate state.
        Equivalence is asserted by ``test_window_study.py``.
        """
        reading = runtime.semantic.context.observe(transaction, index)
        behaviour = runtime.layer.observe(transaction, reading, resolver.resolve(transaction.originator_account_id), minute, index)
        runtime.semantic.context.commit(transaction)
        runtime.layer.commit(transaction, reading, behaviour, minute)

    # -- pass A: semantic features over the primed state -------------------
    def _feature_pass(self, window: Window, resolver: EntityResolver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        train_start = max(window.prime_start_row, EVALUATION_START - ML_TRAIN_CAP)
        train_rows = EVALUATION_START - train_start
        train_x = np.zeros((train_rows, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        test_x = np.zeros((EVALUATION_ROWS, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
        train_y = np.zeros(train_rows, dtype=np.uint8)
        test_y = np.zeros(EVALUATION_ROWS, dtype=np.uint8)
        runtime = self._runtimes(resolver)
        scratch = np.zeros(len(SEMANTIC_FEATURE_NAMES), dtype=np.float32)
        for index, row in self._rows(window.prime_start_row, DENSE_PERIOD_END):
            transaction, minute = _event(index, row)
            if index < train_start:
                self._prime(runtime, resolver, transaction, minute, index)
                continue
            originator = resolver.resolve(transaction.originator_account_id)
            result = runtime.evaluate(transaction, originator, minute, index)
            metrics = result.policies[0].metrics
            vector = semantic_feature_vector(
                {item.type: item.confidence for item in result.semantic.context.objects},
                result.behaviour, result.evidence, len(result.conflicts),
                int(metrics["independent_unqualified_topics"]), float(metrics["effective_risk"]), scratch,
            )
            if index < EVALUATION_START:
                train_x[index - train_start] = vector
                train_y[index - train_start] = int(row[10] == "1")
            else:
                test_x[index - EVALUATION_START] = vector
                test_y[index - EVALUATION_START] = int(row[10] == "1")
            runtime.commit(transaction, result, minute)
        del runtime
        gc.collect()
        return train_x, train_y, test_x, test_y, train_rows

    # -- pass B: decisions --------------------------------------------------
    def _decision_pass(self, window: Window, resolver: EntityResolver, probabilities: dict[str, np.ndarray]) -> dict[str, object]:
        fingerprint = _sha256(f"{self.transactions.resolve()}:{window.name}")
        dataset = AMLSimDataset((), _EphemeralAccounts(), {}, str(self.transactions), fingerprint)
        frozen = AMLDecisionRuntime(dataset, self.output_dir / "frozen_audits_unused")
        graph = StreamingRuntimeGraph(dataset)
        frozen.graph = graph
        runtime = self._runtimes(resolver)

        frozen_decisions: list[str] = []
        semantic_decisions: list[str] = []
        behaviour_decisions: list[str] = []
        frozen_scores = np.zeros(EVALUATION_ROWS, dtype=np.float64)
        semantic_scores = np.zeros(EVALUATION_ROWS, dtype=np.float64)
        behaviour_scores = np.zeros(EVALUATION_ROWS, dtype=np.float64)
        hybrid_decisions = {name: [] for name in MODEL_NAMES}
        hybrid_scores = {name: np.zeros(EVALUATION_ROWS, dtype=np.float64) for name in MODEL_NAMES}
        routed = {name: 0 for name in MODEL_NAMES}

        behaviour_census: Counter[str] = Counter()
        role_census: Counter[str] = Counter()
        scenario_census: Counter[str] = Counter()
        behaviour_objects = scenario_objects = transitions = 0
        confidence_sum = 0.0
        confidence_count = 0
        no_baseline = insufficient = established_baseline = 0
        positives: dict[str, dict[str, object]] = {}
        conflicts_total = 0

        for index, row in self._rows(window.prime_start_row, DENSE_PERIOD_END):
            transaction, minute = _event(index, row)
            if index < EVALUATION_START:
                graph.commit(transaction)
                self._prime(runtime, resolver, transaction, minute, index)
                continue
            position = index - EVALUATION_START
            originator = resolver.resolve(transaction.originator_account_id)

            graph.by_id[transaction.id] = transaction
            preliminary = frozen.evaluate_preliminary(transaction.id)
            frozen_decisions.append(preliminary.decision.decision.value)
            frozen_scores[position] = float(preliminary.policies[0].metrics["effective_risk"])

            result = runtime.evaluate(transaction, originator, minute, index)
            semantic_decisions.append(result.semantic.decision.decision.value)
            semantic_scores[position] = float(result.semantic.policies[0].metrics["effective_risk"])
            behaviour_decisions.append(result.decision.decision.value)
            behaviour_scores[position] = float(result.policies[0].metrics["effective_risk"])

            semantic_types = result.semantic.context.types()
            behaviour_types = result.behaviour.types()
            if SemanticType.NO_ESTABLISHED_BASELINE in semantic_types:
                no_baseline += 1
            if semantic_types & {SemanticType.VALUE_REGIME, SemanticType.TEMPO_REGIME}:
                established_baseline += 1
            if BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY in behaviour_types:
                insufficient += 1
            behaviour_census.update(item.type.value for item in result.behaviour.behaviours)
            role_census[result.behaviour.role.role.value] += 1
            scenario_census.update(item.type.value for item in result.behaviour.scenarios)
            behaviour_objects += len(result.behaviour.behaviours)
            scenario_objects += len(result.behaviour.scenarios)
            transitions += 1 if result.behaviour.transition else 0
            conflicts_total += len(result.conflicts)
            for item in result.behaviour.behaviours:
                if item.type is not BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY:
                    confidence_sum += item.confidence
                    confidence_count += 1

            route = BehaviourDecisionRuntime.routes_to_ml(result)
            for name in MODEL_NAMES:
                probability = float(probabilities[name][position])
                if route:
                    routed[name] += 1
                    fused = runtime.with_ml_evidence(result, _ml_evidence(transaction, name, probability), ML_HIGH_BAND)
                    hybrid_decisions[name].append(fused.decision.decision.value)
                    hybrid_scores[name][position] = max(behaviour_scores[position], probability)
                else:
                    hybrid_decisions[name].append(result.decision.decision.value)
                    hybrid_scores[name][position] = behaviour_scores[position]

            if row[10] == "1":
                substantive = sorted(item.type.value for item in result.behaviour.behaviours
                                     if item.type is not BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY)
                positives[transaction.id] = {
                    "behaviour_decision": result.decision.decision.value,
                    "semantic_decision": result.semantic.decision.decision.value,
                    "frozen_decision": preliminary.decision.decision.value,
                    "role": result.behaviour.role.role.value,
                    "insufficient_behavioural_history": BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY in behaviour_types,
                    "no_established_baseline": SemanticType.NO_ESTABLISHED_BASELINE in semantic_types,
                    "substantive_behaviours": substantive,
                    "scenarios": sorted(item.type.value for item in result.behaviour.scenarios),
                    "observed_events": result.behaviour.lifecycle.observed_events,
                    "distinct_counterparties": result.behaviour.lifecycle.distinct_counterparties,
                    "routed_to_ml": route,
                }

            graph.commit(transaction)
            del graph.by_id[transaction.id]
            runtime.commit(transaction, result, minute)

        del runtime, frozen, graph
        gc.collect()
        return {
            "frozen_decisions": frozen_decisions, "frozen_scores": frozen_scores,
            "semantic_decisions": semantic_decisions, "semantic_scores": semantic_scores,
            "behaviour_decisions": behaviour_decisions, "behaviour_scores": behaviour_scores,
            "hybrid_decisions": hybrid_decisions, "hybrid_scores": hybrid_scores, "routed": routed,
            "behaviour_census": behaviour_census, "role_census": role_census, "scenario_census": scenario_census,
            "behaviour_objects": behaviour_objects, "scenario_objects": scenario_objects,
            "transitions": transitions, "conflicts": conflicts_total,
            "mean_behaviour_confidence": confidence_sum / confidence_count if confidence_count else 0.0,
            "substantive_behaviour_objects": confidence_count,
            "no_established_baseline": no_baseline, "insufficient_behavioural_history": insufficient,
            "established_baseline": established_baseline, "positives": positives,
        }

    # -- one window ---------------------------------------------------------
    def run_window(self, window: Window) -> dict[str, object]:
        resolver = self._resolve_all()
        started = time.perf_counter()
        train_x, train_y, test_x, test_y, train_rows = self._feature_pass(window, resolver)
        feature_seconds = time.perf_counter() - started

        train_positives = int(train_y.sum())
        probabilities: dict[str, np.ndarray] = {}
        ml_results: dict[str, dict[str, object]] = {}
        if train_positives == 0:
            # Honest failure mode rather than a fabricated model.
            for name in MODEL_NAMES:
                probabilities[name] = np.zeros(EVALUATION_ROWS, dtype=np.float64)
                ml_results[name] = {"metrics": None, "note": "no positive training example inside this window"}
        else:
            scale = (len(train_y) - train_positives) / train_positives
            models = _models(self.config, scale)
            for name in MODEL_NAMES:
                model = models.pop(name)
                model.fit(train_x, train_y)
                probability = model.predict_proba(test_x)[:, 1]
                probabilities[name] = probability
                ml_results[name] = {
                    "metrics": _metrics(test_y, probability, (probability >= ML_STANDALONE_THRESHOLD).astype(np.uint8)),
                }
                del model
                gc.collect()
            del models
        del train_x, test_x
        gc.collect()

        started = time.perf_counter()
        outcome = self._decision_pass(window, resolver, probabilities)
        decision_seconds = time.perf_counter() - started

        arms = {
            "frozen_runtime_control": {
                "metrics": _metrics(test_y, outcome["frozen_scores"], _alerts(outcome["frozen_decisions"])),
                "decision_distribution": dict(sorted(Counter(outcome["frozen_decisions"]).items())),
            },
            "semantic_runtime": {
                "metrics": _metrics(test_y, outcome["semantic_scores"], _alerts(outcome["semantic_decisions"])),
                "decision_distribution": dict(sorted(Counter(outcome["semantic_decisions"]).items())),
            },
            "semantic_behaviour_runtime": {
                "metrics": _metrics(test_y, outcome["behaviour_scores"], _alerts(outcome["behaviour_decisions"])),
                "decision_distribution": dict(sorted(Counter(outcome["behaviour_decisions"]).items())),
            },
        }
        for name in MODEL_NAMES:
            arms[f"ml_only/{name}"] = ml_results[name]
            arms[f"hybrid/{name}"] = {
                "metrics": _metrics(test_y, outcome["hybrid_scores"][name], _alerts(outcome["hybrid_decisions"][name])),
                "decision_distribution": dict(sorted(Counter(outcome["hybrid_decisions"][name]).items())),
                "ml_inferences": int(outcome["routed"][name]),
            }
        behaviour_counter = Counter(outcome["behaviour_decisions"])
        return {
            "window": {"name": window.name, "label": window.label, "prime_start_row": window.prime_start_row,
                       "prime_rows": window.prime_rows, "ml_train_rows": train_rows,
                       "ml_train_positives": train_positives},
            "arms": arms,
            "coverage": {
                "behaviour_objects": outcome["behaviour_objects"],
                "substantive_behaviour_objects": outcome["substantive_behaviour_objects"],
                "scenario_objects": outcome["scenario_objects"],
                "role_transitions": outcome["transitions"],
                "conflicts": outcome["conflicts"],
                "established_baselines": outcome["established_baseline"],
                "established_baseline_rate": outcome["established_baseline"] / EVALUATION_ROWS,
                "no_established_baseline": outcome["no_established_baseline"],
                "no_established_baseline_rate": outcome["no_established_baseline"] / EVALUATION_ROWS,
                "insufficient_behavioural_history": outcome["insufficient_behavioural_history"],
                "insufficient_behavioural_history_rate": outcome["insufficient_behavioural_history"] / EVALUATION_ROWS,
                "mean_behaviour_confidence": outcome["mean_behaviour_confidence"],
                "abstain_rate": behaviour_counter.get("ABSTAIN", 0) / EVALUATION_ROWS,
            },
            "behaviour_census": dict(outcome["behaviour_census"].most_common()),
            "role_census": dict(outcome["role_census"].most_common()),
            "scenario_census": dict(outcome["scenario_census"].most_common()),
            "positives": outcome["positives"],
            "timing": {"feature_pass_seconds": feature_seconds, "decision_pass_seconds": decision_seconds},
            "peak_rss_bytes": _peak_rss_bytes(),
        }

    # -- the study ----------------------------------------------------------
    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, object] = {}
        for window in WINDOWS:
            if not window.available:
                results[window.name] = {"window": {"name": window.name, "label": window.label},
                                        "unavailable": True, "reason": window.unavailable_reason}
                print(json.dumps({"window": window.name, "status": "unavailable"}), flush=True)
                continue
            started = time.perf_counter()
            results[window.name] = self.run_window(window)
            print(json.dumps({"window": window.name, "label": window.label,
                              "prime_rows": window.prime_rows,
                              "seconds": round(time.perf_counter() - started, 1),
                              "behaviour": results[window.name]["arms"]["semantic_behaviour_runtime"]["metrics"]},
                             default=str), flush=True)
            (self.output_dir / f"window_{window.name}.json").write_text(
                json.dumps(results[window.name], indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report = self._assemble(results)
        (self.output_dir / "window_study_results.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        self._plots(results)
        self._report(report, results)
        return report

    def _assemble(self, results: dict[str, object]) -> dict[str, object]:
        available = [name for name in results if not results[name].get("unavailable")]
        migration = self._migration(results, available)
        return {
            "study_version": WINDOW_STUDY_VERSION,
            "protocol": {
                "evaluation_rows": EVALUATION_ROWS,
                "evaluation_start_row": EVALUATION_START,
                "evaluation_end_row": DENSE_PERIOD_END,
                "evaluation_set": "held constant across every window: the last 100,000 events of the source's dense period (all of 2022-09-10)",
                "control_arm": "the frozen v0.2 rule runtime is evaluated on the same events in every window; its only history-dependent rule is a 24-hour velocity count",
                "ml_train_cap": ML_TRAIN_CAP,
                "reasoning_code_changed": "none: ontologies, inference rules, projection, conflict pairs, policy engine, routing rule and feature space are identical to the four-arm benchmark",
                "semantic_ontology_hash": ONTOLOGY_HASH,
                "behaviour_ontology_hash": BEHAVIOUR_ONTOLOGY_HASH,
                "dense_period_note": "rows beyond 5,077,237 (2022-09-11 to 09-18) hold 1,108 events with 655 laundering labels, a 59% base rate; they are a generator artefact and are excluded",
            },
            "windows": {name: results[name] for name in results},
            "migration": migration,
        }

    @staticmethod
    def _migration(results: dict[str, object], available: list[str]) -> dict[str, object]:
        """How many laundering events leave `InsufficientBehaviouralHistory`."""
        baseline = available[0]
        base_positives = results[baseline]["positives"]
        rows: dict[str, object] = {}
        for name in available:
            positives = results[name]["positives"]
            insufficient = sum(1 for item in positives.values() if item["insufficient_behavioural_history"])
            explained = sum(1 for item in positives.values() if item["substantive_behaviours"])
            moved = sum(
                1 for key, item in positives.items()
                if key in base_positives
                and base_positives[key]["insufficient_behavioural_history"]
                and not item["insufficient_behavioural_history"]
                and item["substantive_behaviours"]
            )
            rows[name] = {
                "laundering_events": len(positives),
                "insufficient_behavioural_history": insufficient,
                "fully_explained_behaviour": explained,
                "moved_from_insufficient_to_explained_vs_window_" + baseline: moved,
                "mean_observed_events": (sum(item["observed_events"] for item in positives.values()) / len(positives)) if positives else 0.0,
                "mean_distinct_counterparties": (sum(item["distinct_counterparties"] for item in positives.values()) / len(positives)) if positives else 0.0,
            }
        return {"baseline_window": baseline, "by_window": rows}

    # -- outputs -------------------------------------------------------------
    def _plots(self, results: dict[str, object]) -> None:
        available = [name for name in results if not results[name].get("unavailable")]
        if not available:
            return
        x = [results[name]["window"]["prime_rows"] for name in available]
        labels = [f"{name}\n{results[name]['window']['label']}" for name in available]
        panels = [
            ("Behaviour coverage", "share of evaluated events", {
                "behaviour explained (1 - InsufficientBehaviouralHistory)":
                    [1 - results[name]["coverage"]["insufficient_behavioural_history_rate"] for name in available],
                "semantic baseline established":
                    [results[name]["coverage"]["established_baseline_rate"] for name in available],
            }),
            ("False positives", "FP on the constant evaluation set", {
                "Semantic Behaviour Runtime": [results[name]["arms"]["semantic_behaviour_runtime"]["metrics"]["fp"] for name in available],
                "Semantic Runtime": [results[name]["arms"]["semantic_runtime"]["metrics"]["fp"] for name in available],
                "Frozen Runtime (control)": [results[name]["arms"]["frozen_runtime_control"]["metrics"]["fp"] for name in available],
            }),
            ("Recall", "recall on the constant evaluation set", {
                "Semantic Behaviour Runtime": [results[name]["arms"]["semantic_behaviour_runtime"]["metrics"]["recall"] for name in available],
                "Hybrid (LightGBM)": [results[name]["arms"]["hybrid/LightGBM"]["metrics"]["recall"] for name in available],
                "Frozen Runtime (control)": [results[name]["arms"]["frozen_runtime_control"]["metrics"]["recall"] for name in available],
            }),
            ("ABSTAIN rate", "share of evaluated events", {
                "Semantic Behaviour Runtime": [results[name]["coverage"]["abstain_rate"] for name in available],
                "NoEstablishedBaseline rate": [results[name]["coverage"]["no_established_baseline_rate"] for name in available],
            }),
            ("Mean behaviour confidence", "mean confidence of substantive behaviour objects", {
                "behaviour confidence": [results[name]["coverage"]["mean_behaviour_confidence"] for name in available],
            }),
            ("Behaviour objects emitted", "count over the evaluation set", {
                "substantive behaviour objects": [results[name]["coverage"]["substantive_behaviour_objects"] for name in available],
                "scenario objects": [results[name]["coverage"]["scenario_objects"] for name in available],
                "role transitions": [results[name]["coverage"]["role_transitions"] for name in available],
            }),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(18, 9))
        for axis, (title, ylabel, series) in zip(axes.ravel(), panels, strict=True):
            for name, values in series.items():
                axis.plot(range(len(x)), values, marker="o", label=name)
            axis.set_xticks(range(len(x)))
            axis.set_xticklabels(labels, fontsize=7)
            axis.set(title=title, ylabel=ylabel, xlabel="history window before evaluation")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        figure.suptitle("History window scaling — evaluation set held constant (100,000 events, 253 laundering labels)")
        figure.tight_layout()
        figure.savefig(self.output_dir / "window_evolution.png", dpi=150)
        plt.close(figure)

        for index, (title, ylabel, series) in enumerate(panels):
            single, axis = plt.subplots(figsize=(7, 4.5))
            for name, values in series.items():
                axis.plot(range(len(x)), values, marker="o", label=name)
            axis.set_xticks(range(len(x)))
            axis.set_xticklabels(labels, fontsize=8)
            axis.set(title=title, ylabel=ylabel, xlabel="history window before evaluation")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            single.tight_layout()
            single.savefig(self.output_dir / f"evolution_{index + 1}_{title.lower().replace(' ', '_')}.png", dpi=150)
            plt.close(single)

    def _report(self, report: dict[str, object], results: dict[str, object]) -> None:
        available = [name for name in results if not results[name].get("unavailable")]

        def table(headers, rows):
            return (["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
                    + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])

        coverage_rows = []
        metric_rows = []
        for name in available:
            window = results[name]["window"]
            coverage = results[name]["coverage"]
            arms = results[name]["arms"]
            behaviour = arms["semantic_behaviour_runtime"]["metrics"]
            hybrid = arms["hybrid/LightGBM"]["metrics"]
            ml = arms["ml_only/LightGBM"].get("metrics")
            control = arms["frozen_runtime_control"]["metrics"]
            coverage_rows.append((
                f"{name} — {window['label']}", f"{window['prime_rows']:,}",
                f"{coverage['behaviour_objects']:,}", f"{coverage['scenario_objects']:,}",
                f"{coverage['role_transitions']:,}", f"{coverage['established_baselines']:,}",
                f"{coverage['mean_behaviour_confidence']:.4f}",
                f"{coverage['no_established_baseline_rate']:.4f}",
                f"{coverage['insufficient_behavioural_history_rate']:.4f}",
                f"{coverage['abstain_rate']:.4f}",
            ))
            metric_rows.append((
                f"{name} — {window['label']}",
                f"{behaviour['recall']:.4f}", f"{behaviour['precision']:.6f}", f"{behaviour['fp']:,}", behaviour["fn"],
                f"{ml['recall']:.4f}" if ml else "n/a",
                f"{hybrid['recall']:.4f}", f"{hybrid['precision']:.6f}", f"{hybrid['f1']:.6f}",
                f"{control['recall']:.4f}", f"{control['fp']:,}",
            ))
        migration_rows = [
            (f"{name} — {results[name]['window']['label']}",
             report["migration"]["by_window"][name]["laundering_events"],
             report["migration"]["by_window"][name]["insufficient_behavioural_history"],
             report["migration"]["by_window"][name]["fully_explained_behaviour"],
             report["migration"]["by_window"][name][f"moved_from_insufficient_to_explained_vs_window_{report['migration']['baseline_window']}"],
             f"{report['migration']['by_window'][name]['mean_observed_events']:.1f}",
             f"{report['migration']['by_window'][name]['mean_distinct_counterparties']:.1f}")
            for name in available
        ]
        unavailable = [results[name] for name in results if results[name].get("unavailable")]
        lines = [
            "# History window scaling study", "", WINDOW_STUDY_VERSION, "",
            "## Design", "",
            f"The evaluation set is **held constant**: rows {EVALUATION_START:,}–{DENSE_PERIOD_END:,} "
            f"({EVALUATION_ROWS:,} events, all of 2022-09-10), with the same 253 laundering labels in every window. "
            "Only the priming window in front of it changes.", "",
            "No reasoning code was modified. " + report["protocol"]["reasoning_code_changed"], "",
            "The frozen v0.2 rule runtime is carried as a **control**: it has no behavioural state beyond a "
            "24-hour velocity count, so movement in its numbers would indicate the evaluation set is behaving "
            "differently rather than that history is helping.", "",
        ]
        for item in unavailable:
            lines += [f"**Window {item['window']['name']} ({item['window']['label']}) is unavailable.** {item['reason']}", ""]
        lines += [
            "## Coverage", "",
            *table(("Window", "Prime rows", "Behaviour objects", "Scenario objects", "Role transitions",
                    "Established baselines", "Behaviour confidence", "NoEstablishedBaseline rate",
                    "InsufficientBehaviouralHistory rate", "ABSTAIN rate"), coverage_rows), "",
            "## Decision quality", "",
            *table(("Window", "Runtime recall", "Runtime precision", "Runtime FP", "Runtime FN",
                    "ML recall", "Hybrid recall", "Hybrid precision", "Hybrid F1",
                    "Control recall", "Control FP"), metric_rows), "",
            "\"Runtime\" is the Semantic Behaviour Runtime; \"ML\" is LightGBM alone over the semantic feature "
            "space; \"Hybrid\" is Semantic Behaviour Runtime + LightGBM; \"Control\" is the frozen v0.2 rule runtime.", "",
            "## Migration out of InsufficientBehaviouralHistory", "",
            *table(("Window", "Laundering events", "InsufficientBehaviouralHistory", "Fully explained behaviour",
                    f"Moved vs window {report['migration']['baseline_window']}", "Mean observed events",
                    "Mean counterparties"), migration_rows), "",
            "## Evolution plots", "",
            "![window evolution](window_evolution.png)", "",
        ]
        (self.output_dir / "window_study_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("data/ibm_aml_data/HI-Small_Trans.csv"))
    parser.add_argument("--accounts", type=Path, default=Path("data/ibm_aml_data/HI-Small_accounts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aml_window_study"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", type=str, default="")
    args = parser.parse_args()
    study = WindowStudy(args.transactions, args.accounts, args.output_dir, args.threads)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        globals()["WINDOWS"] = tuple(item for item in WINDOWS if item.name in wanted)
    report = study.run()
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
