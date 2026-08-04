"""Runtime-first, label-safe AML cascade experiment.

This module intentionally leaves the frozen v0.2 runtime and standalone ML
benchmark untouched.  It adds a separate experiment in which the Runtime
produces a preliminary decision first; only explicitly eligible transactions
are submitted to a trained model, whose output re-enters the Runtime as typed
evidence.  Labels are held in separate arrays and are never present in a
transaction, graph, fact, evidence, policy, audit, or inference call.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import resource
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve

from .dataset import AMLSimDataset
from .ml_benchmark import (
    FEATURE_NAMES,
    TIMESTAMP_FORMAT,
    CausalFeatureState,
    StreamingRuntimeGraph,
    _account,
    _amount,
    _canonical_bank,
    _parse_timestamp,
)
from .models import Account, Decision, Evidence, FactType, Transaction
from .runtime import (
    AMLDecisionRuntime,
    CascadeRoutingProfile,
    CascadeRuntimeResult,
    CascadeThresholds,
    PreliminaryRuntimeResult,
    RuntimeFirstCascade,
    RUNTIME_VERSION,
    stable_id,
)


CASCADE_VERSION = "aml-runtime-first-cascade/1.0"
LOW_RISK_MAX = 0.10
HIGH_RISK_MIN = 0.90
CALIBRATION_MIN_POSITIVES = 20
FACT_ORDER = tuple(FactType)
RULE_ORDER = (
    "AML-R01-LARGE", "AML-R02-JURISDICTION", "AML-R03-VELOCITY", "AML-R04-SAR",
    "AML-R05-SAR-CONNECTION", "AML-R06-KYC", "AML-R07-BEHAVIOUR", "AML-R08-SOURCE-FUNDS",
    "AML-R09-MANUAL-KYC", "AML-R10-PAYROLL",
)
GRAPH_FACTS = frozenset({
    FactType.NEW_BENEFICIARY, FactType.VELOCITY_INCREASE,
    FactType.CONNECTED_TO_SAR, FactType.REPEATED_DESTINATION,
})


def _sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


def _decision_binary(value: Decision) -> int:
    """Evaluation-only alert mapping; operational decisions remain four-state."""
    return int(value in {Decision.REVIEW, Decision.BLOCK})


def _metrics(labels: np.ndarray, scores: np.ndarray, decisions: list[Decision]) -> dict[str, object]:
    predicted = np.asarray([_decision_binary(item) for item in decisions], dtype=np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "alert_volume": int(predicted.sum()),
        "laundering_captured": int(tp),
        "laundering_missed": int(fn),
    }


@dataclass(frozen=True)
class CascadeConfig:
    input_row_cap: int = 500_000
    train_rows: int = 400_000
    evaluation_horizon: int = 100_000
    seed: int = 20260804
    estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    threads: int = 4
    standalone_threshold: float = 0.50


@dataclass(frozen=True)
class TestMetadata:
    transaction_id: str
    timestamp: str
    amount: float


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    model_version: str
    model_hash: str
    feature_schema_hash: str
    feature_names: tuple[str, ...]
    training_window_identity: str
    thresholds: CascadeThresholds


class EvidenceProvider(Protocol):
    """Pluggable contract: providers return evidence, never decisions."""

    def provide(self, transaction: Transaction, probability: float) -> Evidence: ...


class ProbabilityEvidenceProvider:
    """Serializes a model prediction as typed, provenance-pinned risk evidence."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def provide(self, transaction: Transaction, probability: float) -> Evidence:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("provider probability must be finite and within [0, 1]")
        probability = round(float(probability), 8)
        band = self.artifact.thresholds.band(probability)
        metadata = {
            "provider_name": self.artifact.name,
            "model_version": self.artifact.model_version,
            "model_hash": self.artifact.model_hash,
            "probability": f"{probability:.8f}",
            "confidence_kind": "raw_probability",
            "threshold_band": band,
            "feature_schema_hash": self.artifact.feature_schema_hash,
            "training_window_identity": self.artifact.training_window_identity,
            "timestamp": transaction.timestamp,
            "provenance": "trained_chronological_model;runtime_first_inference",
            "explanation_metadata": _canonical({
                "feature_count": len(self.artifact.feature_names),
                "band": band,
                "thresholds_version": self.artifact.thresholds.version,
            }),
        }
        return Evidence(
            id=stable_id("E", transaction.id, "ML", self.artifact.name, self.artifact.model_hash, f"{probability:.8f}"),
            source=f"ML/{self.artifact.name}", supporting_facts=(), confidence=probability,
            explanation=(f"{self.artifact.name} supplied raw probability {probability:.8f} "
                         f"in the {band} risk band as external Runtime evidence."),
            timestamp=transaction.timestamp, rule_id=f"ML-{self.artifact.name}", direction="risk",
            topic="ml_probability", source_reliability=1.0, recency_days=0, metadata=metadata,
        )


class CompressedCascadeAuditWriter:
    """Deterministic gzip JSONL audit sink; one serializable record per decision."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self.count = 0

    def record(self, result: CascadeRuntimeResult, pins: dict[str, str], profile: CascadeRoutingProfile) -> None:
        preliminary = result.preliminary
        payload = {
            "audit_id": stable_id("CAS-AUD", preliminary.transaction.id, profile.id, pins["model_hash"], result.decision.decision.value),
            "runtime_version": RUNTIME_VERSION,
            "cascade_version": CASCADE_VERSION,
            "transaction": preliminary.transaction.to_dict(),
            "facts": [item.to_dict() for item in preliminary.facts],
            "preliminary_evidence": [item.to_dict() for item in preliminary.evidence],
            "ml_evidence": result.ml_evidence.to_dict() if result.ml_evidence else None,
            "conflicts": [item.to_dict() for item in result.conflicts],
            "policies": [item.to_dict() for item in result.policies],
            "preliminary_decision": preliminary.decision.to_dict(),
            "final_decision": result.decision.to_dict(),
            "replay_pins": dict(sorted(pins.items())),
            "routing_profile": profile.id,
        }
        self._gzip.write((_canonical(payload) + "\n").encode("utf-8"))
        self.count += 1

    def close(self) -> int:
        self._gzip.close()
        self._raw.close()
        return self.path.stat().st_size


class _EphemeralAccounts(dict[str, Account]):
    """Memory-bounded account mapping for IBM AML-Data's missing KYC/SAR feed.

    The public transaction file has no account-control attributes.  Returning
    an immutable default account on lookup preserves frozen fact semantics
    while avoiding a multi-million-entry account registry that contributes no
    information to this experiment.
    """

    def __missing__(self, account_id: str) -> Account:
        bank_id = account_id.partition(":")[0]
        return Account(id=account_id, bank_id=bank_id)


def _runtime_vector(result: PreliminaryRuntimeResult, include_facts: bool, include_graph_history: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    facts = {item.type for item in result.facts}
    rules = {item.rule_id for item in result.evidence}
    feature_names: list[str] = []
    values: list[float] = []
    permitted_facts = FACT_ORDER if include_graph_history else tuple(item for item in FACT_ORDER if item not in GRAPH_FACTS)
    if include_facts:
        feature_names.extend(f"runtime_fact_{item.value}" for item in permitted_facts)
        values.extend(float(item in facts) for item in permitted_facts)
        permitted_rules = tuple(rule for rule in RULE_ORDER if include_graph_history or rule not in {"AML-R03-VELOCITY", "AML-R05-SAR-CONNECTION", "AML-R07-BEHAVIOUR"})
        feature_names.extend(f"runtime_rule_{item}" for item in permitted_rules)
        values.extend(float(item in rules) for item in permitted_rules)
        risk = [item.confidence for item in result.evidence if item.direction == "risk"]
        feature_names.extend(("runtime_fact_count", "runtime_evidence_count", "runtime_risk_evidence_count", "runtime_max_risk_confidence", "runtime_decision_depth"))
        values.extend((float(len(result.facts)), float(len(result.evidence)), float(len(risk)), max(risk, default=0.0), float(len(result.decision.policy_ids))))
    feature_names.extend(f"runtime_preliminary_{item.value}" for item in Decision)
    values.extend(float(result.decision.decision is item) for item in Decision)
    return np.asarray(values, dtype=np.float32), tuple(feature_names)


class CascadeDatasetBuilder:
    """Build causal base and Runtime-first feature matrices without labels in Runtime state."""

    def __init__(self, transaction_path: Path, sorted_path: Path, config: CascadeConfig) -> None:
        self.transaction_path = transaction_path
        self.sorted_path = sorted_path
        self.config = config

    def _category_maps(self) -> tuple[dict[str, int], ...]:
        groups = [set() for _ in range(5)]
        columns = (1, 3, 6, 8, 9)
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            expected = ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
            if next(reader) != expected:
                raise ValueError("unexpected IBM AML-Data schema")
            for index, row in enumerate(reader):
                if index >= self.config.train_rows:
                    break
                for group, column in enumerate(columns):
                    groups[group].add(row[column])
        return tuple({value: index for index, value in enumerate(sorted(group))} for group in groups)

    def build(self) -> dict[str, object]:
        categories = self._category_maps()
        n_train, n_test = self.config.train_rows, self.config.evaluation_horizon
        raw_train = np.empty((n_train, len(FEATURE_NAMES)), dtype=np.float32)
        raw_test = np.empty((n_test, len(FEATURE_NAMES)), dtype=np.float32)
        labels_train = np.empty(n_train, dtype=np.uint8)
        labels_test = np.empty(n_test, dtype=np.uint8)
        train_preliminary = np.empty(n_train, dtype=np.uint8)
        test_preliminary = np.empty(n_test, dtype=np.uint8)
        runtime_full_train: np.ndarray | None = None
        runtime_full_test: np.ndarray | None = None
        runtime_no_facts_train: np.ndarray | None = None
        runtime_no_facts_test: np.ndarray | None = None
        runtime_no_graph_train: np.ndarray | None = None
        runtime_no_graph_test: np.ndarray | None = None
        full_names: tuple[str, ...] = ()
        no_facts_names: tuple[str, ...] = ()
        no_graph_names: tuple[str, ...] = ()
        metadata: list[TestMetadata] = []
        source_stat = self.transaction_path.stat()
        fingerprint = _sha256(f"{self.transaction_path.resolve()}:{source_stat.st_size}:{source_stat.st_mtime_ns}")
        accounts: dict[str, Account] = _EphemeralAccounts()
        dataset = AMLSimDataset((), accounts, {}, str(self.transaction_path), fingerprint)
        runtime = AMLDecisionRuntime(dataset, self.sorted_path.parent / "cascade_feature_stage_audits_unused")
        graph = StreamingRuntimeGraph(dataset)
        runtime.graph = graph
        state = CausalFeatureState()
        first_day: datetime | None = None
        started = time.perf_counter()
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index >= n_train + n_test:
                    break
                if os.environ.get("AML_CASCADE_PROGRESS") and index and index % 100_000 == 0:
                    print(f"cascade feature stage: {index:,}/{n_train + n_test:,}", flush=True)
                timestamp = _parse_timestamp(row[0])
                first_day = timestamp if first_day is None else first_day
                originator, beneficiary = _account(row[1], row[2]), _account(row[3], row[4])
                received, paid = _amount(row[5]), _amount(row[7])
                raw = state.features(timestamp, originator, beneficiary, received, paid, categories, row, first_day)
                transaction = Transaction(
                    id=f"IBM-ML-{index + 2:08d}", timestamp=timestamp.isoformat(timespec="seconds"),
                    originator_account_id=originator, beneficiary_account_id=beneficiary, amount=received,
                    currency=row[6], country_id="", payment_type=row[9],
                    metadata={"from_bank": _canonical_bank(row[1]), "to_bank": _canonical_bank(row[3]), "amount_paid": row[7], "payment_currency": row[8]},
                )
                graph.by_id[transaction.id] = transaction
                preliminary = runtime.evaluate_preliminary(transaction.id)
                full, full_names = _runtime_vector(preliminary, include_facts=True, include_graph_history=True)
                no_facts, no_facts_names = _runtime_vector(preliminary, include_facts=False, include_graph_history=True)
                no_graph, no_graph_names = _runtime_vector(preliminary, include_facts=True, include_graph_history=False)
                if runtime_full_train is None:
                    runtime_full_train = np.empty((n_train, len(full)), dtype=np.float32)
                    runtime_full_test = np.empty((n_test, len(full)), dtype=np.float32)
                    runtime_no_facts_train = np.empty((n_train, len(no_facts)), dtype=np.float32)
                    runtime_no_facts_test = np.empty((n_test, len(no_facts)), dtype=np.float32)
                    runtime_no_graph_train = np.empty((n_train, len(no_graph)), dtype=np.float32)
                    runtime_no_graph_test = np.empty((n_test, len(no_graph)), dtype=np.float32)
                assert runtime_full_train is not None and runtime_full_test is not None
                assert runtime_no_facts_train is not None and runtime_no_facts_test is not None
                assert runtime_no_graph_train is not None and runtime_no_graph_test is not None
                if index < n_train:
                    raw_train[index] = raw
                    labels_train[index] = int(row[10] == "1")
                    train_preliminary[index] = tuple(Decision).index(preliminary.decision.decision)
                    runtime_full_train[index] = full
                    runtime_no_facts_train[index] = no_facts
                    runtime_no_graph_train[index] = no_graph
                else:
                    test_index = index - n_train
                    raw_test[test_index] = raw
                    labels_test[test_index] = int(row[10] == "1")
                    test_preliminary[test_index] = tuple(Decision).index(preliminary.decision.decision)
                    runtime_full_test[test_index] = full
                    runtime_no_facts_test[test_index] = no_facts
                    runtime_no_graph_test[test_index] = no_graph
                    metadata.append(TestMetadata(transaction.id, transaction.timestamp, received))
                graph.commit(transaction)
                del graph.by_id[transaction.id]
                state.commit(timestamp, originator, beneficiary, received)
        if len(metadata) != n_test or runtime_full_train is None or runtime_full_test is None:
            raise ValueError(f"expected {n_test} test events, found {len(metadata)}")
        return {
            "raw_train": raw_train, "raw_test": raw_test, "labels_train": labels_train, "labels_test": labels_test,
            "runtime_full_train": runtime_full_train, "runtime_full_test": runtime_full_test,
            "runtime_no_facts_train": runtime_no_facts_train, "runtime_no_facts_test": runtime_no_facts_test,
            "runtime_no_graph_train": runtime_no_graph_train, "runtime_no_graph_test": runtime_no_graph_test,
            "primary_names": FEATURE_NAMES + full_names,
            "no_facts_names": FEATURE_NAMES + no_facts_names,
            "no_graph_names": FEATURE_NAMES[:12] + no_graph_names,
            "metadata": metadata, "train_preliminary_codes": train_preliminary,
            "preliminary_codes": test_preliminary,
            "feature_seconds": time.perf_counter() - started, "dataset_fingerprint": fingerprint,
        }


def _models(config: CascadeConfig, scale_pos_weight: float) -> dict[str, object]:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    common = {"n_estimators": config.estimators, "max_depth": config.max_depth, "learning_rate": config.learning_rate, "random_state": config.seed}
    return {
        "XGBoost": XGBClassifier(**common, objective="binary:logistic", eval_metric="logloss", tree_method="hist", subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, n_jobs=config.threads),
        "LightGBM": LGBMClassifier(**common, objective="binary", subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, n_jobs=config.threads, verbosity=-1),
        "CatBoost": CatBoostClassifier(iterations=config.estimators, depth=config.max_depth, learning_rate=config.learning_rate, loss_function="Logloss", eval_metric="AUC", random_seed=config.seed, scale_pos_weight=scale_pos_weight, thread_count=config.threads, verbose=False, allow_writing_files=False),
    }


def _package_version(name: str) -> str:
    return importlib.metadata.version({"XGBoost": "xgboost", "LightGBM": "lightgbm", "CatBoost": "catboost"}[name])


def _model_hash(model: object) -> str:
    return _sha256(pickle.dumps(model, protocol=5))


def _schema_hash(names: tuple[str, ...]) -> str:
    return _sha256(_canonical({"feature_names": names, "version": CASCADE_VERSION}))


def _threshold_calibration(name: str, model: object, x: np.ndarray, y: np.ndarray, config: CascadeConfig) -> tuple[CascadeThresholds, dict[str, object]]:
    split = int(len(y) * 0.80)
    calibration_positive = int(y[split:].sum())
    # A tail validation slice with fewer than 20 positives cannot support a
    # stable empirical threshold fit.  The conservative bands are predeclared
    # before the test set is read and are deliberately not tuned afterwards.
    thresholds = CascadeThresholds(LOW_RISK_MAX, HIGH_RISK_MIN)
    method = "predeclared_conservative_bands_due_to_insufficient_validation_positives" if calibration_positive < CALIBRATION_MIN_POSITIVES else "predeclared_conservative_bands_validation_checked_not_fitted"
    started = time.perf_counter()
    model.fit(x[:split], y[:split])
    probabilities = model.predict_proba(x[split:])[:, 1]
    elapsed = time.perf_counter() - started
    bands = {}
    for band, mask in (
        ("low", probabilities < thresholds.low_risk_max),
        ("intermediate", (probabilities >= thresholds.low_risk_max) & (probabilities < thresholds.high_risk_min)),
        ("high", probabilities >= thresholds.high_risk_min),
    ):
        bands[band] = {"events": int(mask.sum()), "positives": int(y[split:][mask].sum())}
    return thresholds, {
        "model": name, "method": method, "validation_range": f"chronological_train_rows[{split}:{len(y)}]", "fit_range": f"chronological_train_rows[0:{split}]",
        "validation_events": int(len(y) - split), "validation_positives": calibration_positive,
        "minimum_positives_for_empirical_selection": CALIBRATION_MIN_POSITIVES,
        "thresholds": {"low_risk_max": thresholds.low_risk_max, "high_risk_min": thresholds.high_risk_min},
        "validation_band_counts": bands, "calibration_fit_seconds": elapsed,
    }


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


class RuntimeFirstCascadeBenchmark:
    def __init__(self, transactions: str | Path, output_dir: str | Path, config: CascadeConfig | None = None) -> None:
        self.transactions = Path(transactions)
        self.output_dir = Path(output_dir)
        self.config = config or CascadeConfig()
        # Reuse the established frozen-benchmark chronological cache instead
        # of duplicating a multi-gigabyte sorted CSV on the nearly-full disk.
        self.sorted_path = Path("artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv")

    def _ensure_protocol(self) -> None:
        if not self.transactions.exists():
            raise FileNotFoundError(self.transactions)
        if not self.sorted_path.exists():
            raise FileNotFoundError(f"frozen chronological cache is required: {self.sorted_path}")
        with self.sorted_path.open("r", encoding="utf-8") as handle:
            count = sum(1 for _ in handle) - 1
        if count < self.config.input_row_cap:
            raise ValueError("chronological cache does not contain the frozen 500,000-row protocol")

    def _fit(self, data: dict[str, object], labels: np.ndarray, fingerprint: str, preliminary: np.ndarray) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, tuple[str, ...]]]:
        scale = (len(labels) - int(labels.sum())) / int(labels.sum())
        all_runs: dict[str, dict[str, object]] = {}
        calibration: dict[str, dict[str, object]] = {}
        schema_names: dict[str, tuple[str, ...]] = {
            "standalone": FEATURE_NAMES,
            "primary": FEATURE_NAMES + data["primary_names"],
            "no_runtime_facts": FEATURE_NAMES + data["no_facts_names"],
            "no_graph_history": FEATURE_NAMES[:12] + data["no_graph_names"],
        }
        training_identity = _sha256(_canonical({"dataset": fingerprint, "range": "chronological[0:400000]", "label_is_training_target_only": True}))
        # Compare primitive values: NumPy object-array equality against str
        # enums is not reliable across the supported NumPy versions.
        decisions = np.asarray([tuple(Decision)[int(value)].value for value in preliminary], dtype="U7")
        routing_masks = {
            "allow_only": decisions == Decision.ALLOW.value,
            "allow_abstain": (decisions == Decision.ALLOW.value) | (decisions == Decision.ABSTAIN.value),
            "non_block": decisions != Decision.BLOCK.value,
        }
        for schema_name, feature_names in schema_names.items():
            if schema_name == "standalone":
                train_x, test_x = data["raw_train"], data["raw_test"]
            elif schema_name == "primary":
                train_x = np.hstack((data["raw_train"], data["runtime_full_train"]))
                test_x = np.hstack((data["raw_test"], data["runtime_full_test"]))
            elif schema_name == "no_runtime_facts":
                train_x = np.hstack((data["raw_train"], data["runtime_no_facts_train"]))
                test_x = np.hstack((data["raw_test"], data["runtime_no_facts_test"]))
            else:
                train_x = np.hstack((data["raw_train"][:, :12], data["runtime_no_graph_train"]))
                test_x = np.hstack((data["raw_test"][:, :12], data["runtime_no_graph_test"]))
            for name, model in _models(self.config, scale).items():
                if schema_name == "primary":
                    threshold_model = _models(self.config, scale)[name]
                    thresholds, calibration_record = _threshold_calibration(name, threshold_model, train_x, labels, self.config)
                    calibration[name] = calibration_record
                    del threshold_model
                else:
                    thresholds = CascadeThresholds(LOW_RISK_MAX, HIGH_RISK_MIN)
                started = time.perf_counter()
                model.fit(train_x, labels)
                train_seconds = time.perf_counter() - started
                probabilities: dict[str, np.ndarray] = {}
                inference_seconds: dict[str, float] = {}
                scopes = ("all",) if schema_name == "standalone" else (("allow_only", "allow_abstain", "non_block") if schema_name == "primary" else ("allow_abstain",))
                for scope in scopes:
                    mask = np.ones(len(test_x), dtype=bool) if scope == "all" else routing_masks[scope]
                    probability = np.full(len(test_x), np.nan, dtype=np.float64)
                    if not mask.any():
                        probabilities[scope] = probability
                        inference_seconds[scope] = 0.0
                        continue
                    started = time.perf_counter()
                    routed_probability = model.predict_proba(test_x[mask])[:, 1]
                    inference_seconds[scope] = time.perf_counter() - started
                    probability[mask] = routed_probability
                    probabilities[scope] = probability
                artifact = ModelArtifact(name, _package_version(name), _model_hash(model), _schema_hash(feature_names), feature_names, training_identity, thresholds)
                all_runs[f"{schema_name}:{name}"] = {
                    "artifact": artifact, "probabilities": probabilities, "inference_seconds": inference_seconds, "training_seconds": train_seconds,
                    "schema": schema_name,
                }
                del model
                gc.collect()
            if schema_name != "standalone":
                del train_x, test_x
                gc.collect()
        return all_runs, calibration, schema_names

    @staticmethod
    def _profile(name: str) -> CascadeRoutingProfile:
        if name == "allow_only":
            return CascadeRoutingProfile("AML-CASCADE-ALLOW-ONLY/1", (Decision.ALLOW,))
        if name == "allow_abstain":
            return CascadeRoutingProfile("AML-CASCADE-ALLOW-ABSTAIN/1", (Decision.ALLOW, Decision.ABSTAIN))
        if name == "non_block":
            return CascadeRoutingProfile("AML-CASCADE-NON-BLOCK/1", (Decision.ALLOW, Decision.ABSTAIN, Decision.REVIEW), score_review_for_prioritisation=True)
        if name == "review_only":
            return CascadeRoutingProfile("AML-CASCADE-NO-ML-BLOCK/1", (Decision.ALLOW, Decision.ABSTAIN), permit_high_risk_ml_block=False)
        raise ValueError(name)

    def _pins(self, artifact: ModelArtifact, transaction: Transaction, thresholds: CascadeThresholds, rules_hash: str, policy_hash: str) -> dict[str, str]:
        return {
            "model_hash": artifact.model_hash,
            "thresholds_hash": _sha256(_canonical(thresholds.__dict__)),
            "feature_schema_hash": artifact.feature_schema_hash,
            "rules_hash": rules_hash,
            "policy_hash": policy_hash,
            "input_snapshot_hash": _sha256(_canonical(transaction.to_dict())),
            "training_window_identity": artifact.training_window_identity,
        }

    def run(self) -> dict[str, object]:
        self._ensure_protocol()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        builder = CascadeDatasetBuilder(self.transactions, self.sorted_path, self.config)
        data = builder.build()
        labels_train = data["labels_train"]
        labels_test = data["labels_test"]
        assert isinstance(labels_train, np.ndarray) and isinstance(labels_test, np.ndarray)
        preliminary_codes = data["preliminary_codes"]
        assert isinstance(preliminary_codes, np.ndarray)
        runs, calibration, feature_names_by_schema = self._fit(data, labels_train, data["dataset_fingerprint"], preliminary_codes)
        thresholds_path = self.output_dir / "thresholds.json"
        thresholds_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
        preliminary_values = [tuple(Decision)[int(code)] for code in preliminary_codes]
        feature_seconds = data["feature_seconds"]
        # The trained probability vectors, labels, and preliminary decisions
        # are all that the audited second pass needs.  Release feature matrices
        # before constructing the full evidence traces.
        del data
        gc.collect()
        profiles = {name: self._profile(name) for name in ("allow_only", "allow_abstain", "non_block", "review_only")}
        variants: dict[str, tuple[str, str, CascadeRoutingProfile]] = {}
        for model_name in ("XGBoost", "LightGBM", "CatBoost"):
            variants[f"Runtime → {model_name} → Runtime"] = (f"primary:{model_name}", "allow_abstain", profiles["allow_abstain"])
            variants[f"Ablation allow only: {model_name}"] = (f"primary:{model_name}", "allow_only", profiles["allow_only"])
            variants[f"Ablation all non-BLOCK: {model_name}"] = (f"primary:{model_name}", "non_block", profiles["non_block"])
            variants[f"Ablation no Runtime facts: {model_name}"] = (f"no_runtime_facts:{model_name}", "allow_abstain", profiles["allow_abstain"])
            variants[f"Ablation no graph/history: {model_name}"] = (f"no_graph_history:{model_name}", "allow_abstain", profiles["allow_abstain"])
            variants[f"Ablation no ML BLOCK: {model_name}"] = (f"primary:{model_name}", "allow_abstain", profiles["review_only"])
        # The first-stage Runtime has already been executed causally in the
        # feature pass.  Re-run just the fixed test stream to materialize full
        # auditable traces, then finalise every cascade variant against it.
        source_stat = self.transactions.stat()
        fingerprint = _sha256(f"{self.transactions.resolve()}:{source_stat.st_size}:{source_stat.st_mtime_ns}")
        accounts: dict[str, Account] = _EphemeralAccounts()
        dataset = AMLSimDataset((), accounts, {}, str(self.transactions), fingerprint)
        runtime = AMLDecisionRuntime(dataset, self.output_dir / "base_stage_audits_unused")
        graph = StreamingRuntimeGraph(dataset)
        runtime.graph = graph
        rules_hash = _sha256(_canonical([rule.__dict__ for rule in runtime.rules.rules]))
        base_policy_hash = _sha256(_canonical(runtime.policies.config.__dict__))
        cascade_objects = {key: RuntimeFirstCascade(runtime, self.output_dir / "audits" / "unused", runs[run_key]["artifact"].thresholds) for key, (run_key, _mode, _profile) in variants.items()}
        primary_writers: dict[str, CompressedCascadeAuditWriter] = {}
        for model in ("XGBoost", "LightGBM", "CatBoost"):
            primary_writers[model] = CompressedCascadeAuditWriter(self.output_dir / "audits" / f"runtime_first_{model.lower()}.jsonl.gz")
        values: dict[str, dict[str, object]] = {}
        for name, (run_key, mode, profile) in variants.items():
            values[name] = {"decisions": [], "scores": np.zeros(self.config.evaluation_horizon, dtype=np.float64), "routed": 0, "routed_mask": np.zeros(self.config.evaluation_horizon, dtype=bool), "ml_seconds": 0.0, "policy_seconds": 0.0, "policy_latencies": np.zeros(self.config.evaluation_horizon, dtype=np.float64), "transitions": Counter(), "ml_evidence": 0, "preliminary_allow_laundering": 0, "recovered": 0, "clean_allow_escalated": 0, "review_preserved": 0, "block_preserved": 0, "abstain_resolved": 0}
        runtime_decisions: list[Decision] = []
        runtime_scores = np.zeros(self.config.evaluation_horizon, dtype=np.float64)
        first_stage_latencies = np.zeros(self.config.evaluation_horizon, dtype=np.float64)
        case_studies: dict[str, list[dict[str, object]]] = defaultdict(list)
        stage_started = time.perf_counter()
        with self.sorted_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                if index >= self.config.train_rows + self.config.evaluation_horizon:
                    break
                timestamp = _parse_timestamp(row[0])
                originator, beneficiary = _account(row[1], row[2]), _account(row[3], row[4])
                transaction = Transaction(
                    id=f"IBM-ML-{index + 2:08d}", timestamp=timestamp.isoformat(timespec="seconds"), originator_account_id=originator,
                    beneficiary_account_id=beneficiary, amount=_amount(row[5]), currency=row[6], country_id="", payment_type=row[9],
                    metadata={"from_bank": _canonical_bank(row[1]), "to_bank": _canonical_bank(row[3]), "amount_paid": row[7], "payment_currency": row[8]},
                )
                graph.by_id[transaction.id] = transaction
                preliminary_started = time.perf_counter()
                preliminary = runtime.evaluate_preliminary(transaction.id)
                if index < self.config.train_rows:
                    graph.commit(transaction)
                    del graph.by_id[transaction.id]
                    continue
                test_index = index - self.config.train_rows
                first_stage_latencies[test_index] = time.perf_counter() - preliminary_started
                if preliminary.decision.decision is not preliminary_values[test_index]:
                    raise AssertionError("Runtime-first feature stream diverged from audited test stream")
                runtime_decisions.append(preliminary.decision.decision)
                runtime_scores[test_index] = float(preliminary.policies[0].metrics["effective_risk"])
                for variant, (run_key, _mode, profile) in variants.items():
                    run = runs[run_key]
                    artifact = run["artifact"]
                    assert isinstance(artifact, ModelArtifact)
                    state = values[variant]
                    eligible = cascade_objects[variant].policy.eligible(preliminary.decision.decision, profile)
                    evidence = None
                    probability = 0.0
                    if eligible:
                        # This array was generated by a batched inference call over
                        # precisely this routing profile's eligible transactions.
                        probability = float(run["probabilities"][_mode][test_index])
                        if not math.isfinite(probability):
                            raise AssertionError("inference was requested outside its routed population")
                        evidence = ProbabilityEvidenceProvider(artifact).provide(transaction, probability)
                        state["routed"] += 1
                        state["routed_mask"][test_index] = True
                        state["ml_evidence"] += 1
                    pins = self._pins(artifact, transaction, artifact.thresholds, rules_hash, _sha256(base_policy_hash + profile.id))
                    policy_started = time.perf_counter()
                    final = cascade_objects[variant].finalise(preliminary, profile, evidence, pins, write_audit=False)
                    policy_elapsed = time.perf_counter() - policy_started
                    state["policy_seconds"] += policy_elapsed
                    state["policy_latencies"][test_index] = policy_elapsed
                    final_decision = final.decision.decision
                    state["decisions"].append(final_decision)
                    state["scores"][test_index] = max(runtime_scores[test_index], probability)
                    state["transitions"][f"{preliminary.decision.decision.value} → {final_decision.value}"] += 1
                    label = int(labels_test[test_index])  # evaluation only, after final decision
                    if preliminary.decision.decision is Decision.ALLOW and label:
                        state["preliminary_allow_laundering"] += 1
                        if final_decision in {Decision.REVIEW, Decision.BLOCK}:
                            state["recovered"] += 1
                    if preliminary.decision.decision is Decision.ALLOW and not label and final_decision in {Decision.REVIEW, Decision.BLOCK}:
                        state["clean_allow_escalated"] += 1
                    if preliminary.decision.decision is Decision.REVIEW and final_decision is Decision.REVIEW:
                        state["review_preserved"] += 1
                    if preliminary.decision.decision is Decision.BLOCK and final_decision is Decision.BLOCK:
                        state["block_preserved"] += 1
                    if preliminary.decision.decision is Decision.ABSTAIN and final_decision in {Decision.ALLOW, Decision.REVIEW}:
                        state["abstain_resolved"] += 1
                    if variant.startswith("Runtime →"):
                        model = variant.split(" → ")[1]
                        primary_writers[model].record(final, pins, profile)
                    # Store actual, post-decision labelled traces only for bounded case analysis.
                    if variant.startswith("Runtime →") or variant.startswith("Ablation all non-BLOCK"):
                        model = variant.split(" → ")[1] if variant.startswith("Runtime →") else variant.rsplit(": ", 1)[1]
                        trace = {"transaction_id": transaction.id, "timestamp": transaction.timestamp, "label_evaluation_only": label, "preliminary": preliminary.decision.decision.value, "final": final_decision.value, "ml_probability": probability if evidence else None, "facts": [fact.to_dict() for fact in preliminary.facts], "evidence": [item.to_dict() for item in preliminary.evidence], "ml_evidence": evidence.to_dict() if evidence else None, "conflicts": [item.to_dict() for item in final.conflicts], "policies": [item.to_dict() for item in final.policies]}
                        prefix = f"{model}:"
                        if label and preliminary.decision.decision in {Decision.ALLOW, Decision.ABSTAIN} and final_decision in {Decision.REVIEW, Decision.BLOCK} and len(case_studies[prefix + "runtime_missed_ml_recovered"]) < 10:
                            case_studies[prefix + "runtime_missed_ml_recovered"].append(trace)
                        if label and final_decision in {Decision.ALLOW, Decision.ABSTAIN} and len(case_studies[prefix + "missed_by_both"]) < 10:
                            case_studies[prefix + "missed_by_both"].append(trace)
                        if not label and preliminary.decision.decision is Decision.ALLOW and final_decision in {Decision.REVIEW, Decision.BLOCK} and len(case_studies[prefix + "clean_allow_false_escalated"]) < 10:
                            case_studies[prefix + "clean_allow_false_escalated"].append(trace)
                        if preliminary.decision.decision is Decision.REVIEW and evidence and probability < artifact.thresholds.low_risk_max and len(case_studies[prefix + "review_preserved_low_ml"]) < 10:
                            case_studies[prefix + "review_preserved_low_ml"].append(trace)
                        if preliminary.decision.decision is Decision.ABSTAIN and final_decision in {Decision.ALLOW, Decision.REVIEW} and len(case_studies[prefix + "abstain_resolved"]) < 10:
                            case_studies[prefix + "abstain_resolved"].append(trace)
                        if evidence and preliminary.decision.decision in {Decision.ALLOW, Decision.ABSTAIN}:
                            no_facts_probability = float(runs[f"no_runtime_facts:{model}"]["probabilities"]["allow_abstain"][test_index])
                            if abs(probability - no_facts_probability) >= 0.25 and len(case_studies[prefix + "runtime_facts_materially_changed_probability"]) < 10:
                                trace["no_runtime_facts_probability"] = no_facts_probability
                                trace["probability_delta_primary_minus_no_facts"] = probability - no_facts_probability
                                case_studies[prefix + "runtime_facts_materially_changed_probability"].append(trace)
                graph.commit(transaction)
                del graph.by_id[transaction.id]
        full_stream_stage_seconds = time.perf_counter() - stage_started
        first_stage_seconds = float(first_stage_latencies.sum())
        audit_sizes = {name: writer.close() for name, writer in primary_writers.items()}
        runtime_metrics = _metrics(labels_test, runtime_scores, runtime_decisions)
        results: dict[str, object] = {}
        for name, state in values.items():
            decisions = state["decisions"]
            assert isinstance(decisions, list)
            metric = _metrics(labels_test, state["scores"], decisions)
            distribution = Counter(item.value for item in decisions)
            run_key, mode, _profile = variants[name]
            run = runs[run_key]
            state["ml_seconds"] = run["inference_seconds"][mode]
            routed_mask = state["routed_mask"]
            policy_latencies = state["policy_latencies"]
            inference_per_routed_event = state["ml_seconds"] / state["routed"] if state["routed"] else 0.0
            end_to_end_latencies = first_stage_latencies + policy_latencies + routed_mask.astype(np.float64) * inference_per_routed_event
            results[name] = {
                "metrics": metric,
                "decision_distribution": dict(distribution),
                "routing": {
                    "transactions_sent_to_ml": state["routed"], "percentage_sent_to_ml": state["routed"] / len(decisions), "ml_inferences_avoided": len(decisions) - state["routed"],
                    "laundering_in_preliminary_allow": state["preliminary_allow_laundering"], "laundering_recovered_by_ml": state["recovered"], "clean_allow_escalated_by_ml": state["clean_allow_escalated"],
                    "preliminary_review_preserved": state["review_preserved"], "preliminary_block_preserved": state["block_preserved"], "abstain_resolved": state["abstain_resolved"],
                    "transitions": dict(sorted(state["transitions"].items())), "ml_evidence_records": state["ml_evidence"],
                },
                "latency": {
                    "runtime_first_stage_mean_ms": 1000 * first_stage_seconds / len(decisions),
                    "runtime_first_stage_p95_ms": float(1000 * np.percentile(first_stage_latencies, 95)),
                    "ml_inference_routed_mean_ms": 1000 * state["ml_seconds"] / state["routed"] if state["routed"] else 0.0,
                    "final_policy_mean_ms": 1000 * state["policy_seconds"] / len(decisions),
                    "end_to_end_mean_ms": float(1000 * end_to_end_latencies.mean()),
                    "end_to_end_p95_ms": float(1000 * np.percentile(end_to_end_latencies, 95)),
                    "throughput_transactions_per_second": len(decisions) / float(end_to_end_latencies.sum()),
                },
                "peak_process_rss_bytes": _peak_rss_bytes(),
            }
        # Standalone ML baselines use the original 18-feature contract and the
        # already-fitted fixed 0.50 model threshold, unchanged from baseline.
        standalone: dict[str, object] = {}
        for name in ("XGBoost", "LightGBM", "CatBoost"):
            run = runs[f"standalone:{name}"]
            probability = run["probabilities"]["all"]
            binary = [Decision.REVIEW if item >= self.config.standalone_threshold else Decision.ALLOW for item in probability]
            standalone[name] = {"metrics": _metrics(labels_test, probability, binary), "training_seconds": run["training_seconds"], "inference_latency_ms": 1000 * run["inference_seconds"]["all"] / len(probability)}
        previous_path = Path("artifacts/aml_ml_benchmark/comparison_results.json")
        previous_hybrids = json.loads(previous_path.read_text())["hybrid_results"] if previous_path.exists() else {}
        report = {
            "protocol": {"dataset": str(self.transactions), "chronological_cache": str(self.sorted_path), "total_rows": self.config.input_row_cap, "train_rows": self.config.train_rows, "test_rows": self.config.evaluation_horizon, "seed": self.config.seed, "label_boundary": "labels are present only in separate training/evaluation arrays; no Runtime, inference provider, transaction, graph, fact, evidence, policy, or audit receives a laundering label"},
            "feature_contract": {"allowed": ["original transaction fields", "causal history features", "Runtime facts", "rule activation indicators", "evidence quality metadata", "preliminary Runtime decision", "decision depth"], "forbidden": ["laundering label", "future transactions", "future graph edges", "future aggregates", "post-decision audit output", "final Runtime decision", "evaluation-label-derived fields"], "schemas": {name: list(names) for name, names in feature_names_by_schema.items()}},
            "threshold_calibration": calibration,
            "class_distribution": {"train": {"negative": int(len(labels_train) - labels_train.sum()), "positive": int(labels_train.sum())}, "test": {"negative": int(len(labels_test) - labels_test.sum()), "positive": int(labels_test.sum())}},
            "runtime_only": {"metrics": runtime_metrics, "decision_distribution": dict(Counter(item.value for item in runtime_decisions)), "first_stage_seconds": first_stage_seconds, "first_stage_p95_ms": float(1000 * np.percentile(first_stage_latencies, 95)), "full_stream_stage_seconds_including_train": full_stream_stage_seconds},
            "standalone_ml": standalone,
            "previous_ml_first_hybrids": previous_hybrids,
            "cascade_results": results,
            "audit": {"primary_audit_files": {name: str(self.output_dir / "audits" / f"runtime_first_{name.lower()}.jsonl.gz") for name in audit_sizes}, "compressed_bytes": audit_sizes, "records_per_primary_model": self.config.evaluation_horizon},
            "case_studies": dict(case_studies),
            "feature_engineering_seconds": feature_seconds,
            "hyperparameters": {"estimators": self.config.estimators, "max_depth": self.config.max_depth, "learning_rate": self.config.learning_rate, "seed": self.config.seed, "threads": self.config.threads, "standalone_threshold": self.config.standalone_threshold},
        }
        self._write_artifacts(report)
        return report

    def _write_artifacts(self, report: dict[str, object]) -> None:
        (self.output_dir / "comparison_results.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        contract = report["feature_contract"]
        assert isinstance(contract, dict)
        (self.output_dir / "feature_contract.md").write_text(
            "# Runtime-first feature contract\n\n"
            "## Allowed at inference\n\n" + "\n".join(f"- {item}" for item in contract["allowed"]) +
            "\n\n## Prohibited\n\n" + "\n".join(f"- {item}" for item in contract["forbidden"]) +
            "\n\n## Exact schemas\n\n" + "\n\n".join(
                f"### {name}\n\n`{', '.join(features)}`" for name, features in contract["schemas"].items()
            ) +
            "\n\nThe Runtime stage is causal: graph/history state is committed only after its transaction is evaluated. ML receives feature arrays and no label argument at inference.\n",
            encoding="utf-8",
        )
        self._write_csvs(report)
        self._write_curves(report)
        self._write_case_studies(report)
        self._write_model_cards(report)
        self._write_report(report)

    def _write_csvs(self, report: dict[str, object]) -> None:
        results = report["cascade_results"]
        assert isinstance(results, dict)
        with (self.output_dir / "routing_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("system", "transition", "count"))
            for name, item in results.items():
                for transition, count in item["routing"]["transitions"].items():
                    writer.writerow((name, transition, count))
        with (self.output_dir / "ablation_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("system", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "routed", "avoided", "review", "block", "allow", "abstain"))
            for name, item in results.items():
                metric, routing, decisions = item["metrics"], item["routing"], item["decision_distribution"]
                writer.writerow((name, metric["precision"], metric["recall"], metric["f1"], metric["roc_auc"], metric["pr_auc"], metric["false_positive_rate"], metric["false_negative_rate"], routing["transactions_sent_to_ml"], routing["ml_inferences_avoided"], decisions.get("REVIEW", 0), decisions.get("BLOCK", 0), decisions.get("ALLOW", 0), decisions.get("ABSTAIN", 0)))

    def _write_curves(self, report: dict[str, object]) -> None:
        # Scores are not retained after metrics to keep the experiment memory-bounded;
        # plot operational PR/ROC points from measured metrics instead of inventing curves.
        results = report["cascade_results"]
        assert isinstance(results, dict)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, item in results.items():
            metric = item["metrics"]
            axes[0].scatter(metric["false_positive_rate"], metric["recall"], label=name, s=18)
            axes[1].scatter(metric["recall"], metric["precision"], label=name, s=18)
        axes[0].set(xlabel="False-positive rate", ylabel="Recall", title="Operational ROC points")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Operational precision-recall points")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=5, ncol=2)
        figure.tight_layout()
        figure.savefig(self.output_dir / "curves.png", dpi=160)
        plt.close(figure)

    def _write_case_studies(self, report: dict[str, object]) -> None:
        cases = report["case_studies"]
        assert isinstance(cases, dict)
        lines = ["# Runtime-first cascade case studies", "", "All labels below are evaluated after the final Runtime decision and are not part of evidence generation.", ""]
        for name, values in cases.items():
            lines.extend((f"## {name}", "```json", json.dumps(values, indent=2), "```", ""))
        lines.extend(("## Preserved BLOCK cases", "", "The frozen routing profiles skip preliminary BLOCK. Consequently no low-ML-probability BLOCK trace exists; each preliminary BLOCK is preserved without an ML inference, which is recorded in the routing matrix and audit provenance.", ""))
        (self.output_dir / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_model_cards(self, report: dict[str, object]) -> None:
        cards = self.output_dir / "model_cards"
        cards.mkdir(exist_ok=True)
        calibration = report["threshold_calibration"]
        for model, threshold in calibration.items():
            (cards / f"{model.lower()}.md").write_text(
                f"# {model} Runtime-first evidence provider\n\n"
                f"Model configuration is frozen at 200 estimators/iterations, depth 6, learning rate 0.05, seed 20260804, and four threads.\n\n"
                f"```json\n{json.dumps(threshold, indent=2, sort_keys=True)}\n```\n\n"
                "This provider returns immutable Evidence, never an operational decision. The final decision is selected by `RuntimeFirstCascadePolicyEngine`.\n",
                encoding="utf-8",
            )

    def _write_report(self, report: dict[str, object]) -> None:
        rows: list[tuple[object, ...]] = []
        runtime = report["runtime_only"]
        rows.append(("Runtime only", *[f"{runtime['metrics'][key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")], "-"))
        for name, item in report["standalone_ml"].items():
            rows.append((f"Standalone {name}", *[f"{item['metrics'][key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")], "binary"))
        for name, item in report["previous_ml_first_hybrids"].items():
            rows.append((f"Previous ML-first {name}", *[f"{item['metrics'][key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")], "100,000"))
        for name, item in report["cascade_results"].items():
            rows.append((name, *[f"{item['metrics'][key]:.6f}" for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")], f"{item['routing']['transactions_sent_to_ml']:,}"))
        routing_rows = []
        for name, item in report["cascade_results"].items():
            r = item["routing"]
            routing_rows.append((name, r["transactions_sent_to_ml"], r["ml_inferences_avoided"], r["laundering_in_preliminary_allow"], r["laundering_recovered_by_ml"], r["clean_allow_escalated_by_ml"], r["preliminary_review_preserved"], r["preliminary_block_preserved"], r["abstain_resolved"]))
        lines = [
            "# Runtime-first AML cascade", "",
            "## Protocol", "",
            "The experiment uses the frozen chronological protocol: 500,000 IBM AML-Data events, 400,000 train events, and the next 100,000 events for evaluation. The Runtime is executed first and is the sole emitter of ALLOW, REVIEW, BLOCK, and ABSTAIN. Labels train standalone/cascade models and evaluate results only; they never enter Runtime state, provider inference, evidence, policies, graph state, or audits.", "",
            "## Threshold selection", "",
            "Threshold bands are selected before test labels are read. The final chronological 20% of training is a validation subsection. If it has fewer than 20 positives, the predeclared conservative low/intermediate/high bands are [0, 0.10), [0.10, 0.90), and [0.90, 1]; no empirical threshold is fitted. Exact validation counts and trade-offs are in `thresholds.json`.", "",
            "## Main comparison", "",
            _table(("System", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FNR", "ML routed"), rows), "",
            "Binary evaluation mapping is explicitly exploratory: REVIEW and BLOCK are alerts; ALLOW and ABSTAIN are non-alerts. Standalone ML does not emit operational Runtime decisions.", "",
            "## Cascade routing", "",
            _table(("System", "ML routed", "ML avoided", "Laundering in preliminary ALLOW", "Recovered", "Clean ALLOW escalated", "REVIEW preserved", "BLOCK preserved", "ABSTAIN resolved"), routing_rows), "",
            "Every recall claim should be read with its denominator: the test horizon contains only " + str(report["class_distribution"]["test"]["positive"]) + " laundering-labelled events. This is an engineering experiment, not evidence of statistical superiority or production readiness.", "",
            "No bootstrap interval is claimed: with 28 positive evaluation events, resampling would not make the uncertainty stable or confer statistical superiority.", "",
            "## Operational measurements", "",
            "Runtime-first latency, routed-inference latency, final-policy latency, end-to-end mean and p95 latency, throughput, peak RSS, and compressed audit sizes are in `comparison_results.json`. The latency scope is consistent across systems: decision computation through final policy selection; the separately reported audit size covers persisted full traces. p95 is computed from measured per-event Runtime and final-policy timings plus the measured routed-batch inference cost allocated only to routed events.", "",
            "## Audits and replay", "",
            "Primary cascades write one gzip JSONL audit record per test transaction. Each record pins model hash, threshold hash, feature-schema hash, rules hash, policy hash, training-window identity, and an input-snapshot hash. Replays reject missing pins.", "",
        ]
        (self.output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Runtime-first AML cascade experiment")
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="artifacts/aml_runtime_first_cascade")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = RuntimeFirstCascadeBenchmark(args.transactions, args.output_dir, CascadeConfig(threads=args.threads)).run()
    print(json.dumps({"results": str(Path(args.output_dir) / "comparison_results.json"), "report": str(Path(args.output_dir) / "comparison_report.md"), "runtime_only": report["runtime_only"], "cascade_results": report["cascade_results"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
