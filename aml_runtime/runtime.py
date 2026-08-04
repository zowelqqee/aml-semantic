"""The complete deterministic AML decision pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Mapping

from .dataset import AMLSimDataset
from .graph import TransactionGraph, parse_timestamp
from .models import Conflict, Decision, DecisionRecord, Evidence, Fact, FactType, PolicyOutcome, Transaction

RUNTIME_VERSION = "aml-decision-runtime/0.2.0"


def stable_id(prefix: str, *parts: str) -> str:
    value = "|".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(value).hexdigest()[:12]}"


def _fact(transaction: Transaction, kind: FactType, value: str, confidence: float, explanation: str, provenance: str) -> Fact:
    return Fact(stable_id("F", transaction.id, kind.value, value), kind, transaction.id, value, confidence, explanation, transaction.timestamp, provenance)


class FactExtractor:
    HIGH_RISK_COUNTRIES = frozenset({"AF", "BY", "IR", "KP", "MM", "RU", "SY", "VE", "YE"})
    LARGE_TRANSFER_THRESHOLD = 50_000.0
    KYC_MAX_AGE_DAYS = 730

    def extract(self, transaction: Transaction, graph: TransactionGraph) -> tuple[Fact, ...]:
        facts: list[Fact] = []
        account = graph.dataset.accounts[transaction.originator_account_id]
        prior_24h = graph.prior_outbound(transaction, hours=24)
        prior_30d = graph.prior_outbound(transaction, days=30)
        pair_history = graph.prior_pair(transaction, days=365)
        if transaction.amount >= self.LARGE_TRANSFER_THRESHOLD:
            facts.append(_fact(transaction, FactType.LARGE_TRANSFER, f"{transaction.amount:.2f}", 0.91, f"Amount {transaction.amount:.2f} exceeds the {self.LARGE_TRANSFER_THRESHOLD:.0f} monitoring threshold.", "transaction.amount"))
        if not pair_history:
            facts.append(_fact(transaction, FactType.NEW_BENEFICIARY, transaction.beneficiary_account_id, 0.84, "No earlier payment from this originator to this beneficiary exists in the loaded history.", "transaction-history"))
        if transaction.country_id in self.HIGH_RISK_COUNTRIES:
            facts.append(_fact(transaction, FactType.HIGH_RISK_COUNTRY, transaction.country_id, 0.95, f"Destination jurisdiction {transaction.country_id} is on the configured high-risk jurisdiction list.", "jurisdiction-list/v1"))
        if len(prior_24h) >= 3:
            facts.append(_fact(transaction, FactType.VELOCITY_INCREASE, str(len(prior_24h) + 1), 0.88, f"{len(prior_24h) + 1} outbound transfers occurred within 24 hours.", "transaction-history"))
        if account.kyc_updated_at:
            try:
                age = (parse_timestamp(transaction.timestamp) - parse_timestamp(account.kyc_updated_at)).days
                if age > self.KYC_MAX_AGE_DAYS:
                    facts.append(_fact(transaction, FactType.OLD_KYC, str(age), 0.86, f"KYC was last updated {age} days before this transaction.", "account.kyc_updated_at"))
            except ValueError:
                pass
        if account.is_sar:
            facts.append(_fact(transaction, FactType.PREVIOUS_SAR, account.id, 0.99, "The originator account has a known SAR/laundering flag in the supplied AMLSim data.", "account.is_sar"))
        sar_path = graph.connected_to_sar(transaction.beneficiary_account_id)
        if sar_path:
            facts.append(_fact(transaction, FactType.CONNECTED_TO_SAR, " -> ".join(sar_path), 0.98, f"Beneficiary is connected to a SAR-flagged account by path {' -> '.join(sar_path)}.", "transaction-graph"))
        if len(graph.prior_pair(transaction, days=30)) >= 2:
            facts.append(_fact(transaction, FactType.REPEATED_DESTINATION, transaction.beneficiary_account_id, 0.82, "The same beneficiary has already received at least two payments in the last 30 days.", "transaction-history"))
        if account.verified_source_of_funds:
            facts.append(_fact(transaction, FactType.VERIFIED_SOURCE_OF_FUNDS, account.id, 0.94, "Account has a verified source-of-funds control.", "account.verified_source_of_funds"))
        if account.recently_manually_verified:
            facts.append(_fact(transaction, FactType.RECENT_MANUAL_VERIFICATION, account.id, 0.96, "Account was recently manually KYC verified.", "account.recently_manually_verified"))
        if account.expected_payroll_day or transaction.payment_type.strip().lower() == "payroll":
            facts.append(_fact(transaction, FactType.EXPECTED_PAYROLL_DAY, transaction.originator_account_id, 0.90, "The transaction is attributed to an expected payroll event.", "account.expected_payroll_day/payment_type"))
        return tuple(sorted(facts, key=lambda fact: (fact.type.value, fact.id)))


@dataclass(frozen=True)
class Rule:
    id: str
    fact_types: tuple[FactType, ...]
    confidence: float
    direction: str
    description: str
    topic: str
    source_reliability: float

    def apply(self, transaction: Transaction, facts: tuple[Fact, ...]) -> Evidence | None:
        supporting = tuple(fact for fact in facts if fact.type in self.fact_types)
        if not supporting:
            return None
        fact_ids = tuple(sorted(fact.id for fact in supporting))
        confidence = round(min(0.99, max(self.confidence, sum(fact.confidence for fact in supporting) / len(supporting))), 2)
        recency_days = None
        old_kyc_values = [int(fact.value) for fact in supporting if fact.type is FactType.OLD_KYC and fact.value.isdigit()]
        if old_kyc_values:
            recency_days = max(old_kyc_values)
        return Evidence(
            stable_id("E", transaction.id, self.id, *fact_ids), self.id, fact_ids, confidence,
            self.description.format(facts=", ".join(fact.type.value for fact in supporting)), transaction.timestamp,
            self.id, self.direction, self.topic, self.source_reliability, recency_days,
        )


class AMLRuleEngine:
    def __init__(self) -> None:
        self.rules = (
            Rule("AML-R01-LARGE", (FactType.LARGE_TRANSFER,), 0.91, "risk", "Large-transfer rule matched: {facts}.", "transaction_value", 0.98),
            Rule("AML-R02-JURISDICTION", (FactType.HIGH_RISK_COUNTRY,), 0.95, "risk", "High-risk-jurisdiction rule matched: {facts}.", "jurisdiction", 0.99),
            Rule("AML-R03-VELOCITY", (FactType.VELOCITY_INCREASE,), 0.88, "risk", "Rapid-transfer rule matched: {facts}.", "transaction_velocity", 0.90),
            Rule("AML-R04-SAR", (FactType.PREVIOUS_SAR,), 0.99, "risk", "Known-suspicious-account rule matched: {facts}.", "known_suspicion", 0.99),
            Rule("AML-R05-SAR-CONNECTION", (FactType.CONNECTED_TO_SAR,), 0.98, "risk", "Connection-to-SAR rule matched: {facts}.", "known_suspicion", 0.86),
            Rule("AML-R06-KYC", (FactType.OLD_KYC,), 0.86, "risk", "Stale-KYC rule matched: {facts}.", "kyc_recency", 0.78),
            Rule("AML-R07-BEHAVIOUR", (FactType.NEW_BENEFICIARY, FactType.REPEATED_DESTINATION), 0.84, "risk", "Abnormal-beneficiary-behaviour rule matched: {facts}.", "counterparty_behaviour", 0.72),
            Rule("AML-R08-SOURCE-FUNDS", (FactType.VERIFIED_SOURCE_OF_FUNDS,), 0.94, "mitigation", "Verified-source-of-funds control matched: {facts}.", "financial_integrity", 0.98),
            Rule("AML-R09-MANUAL-KYC", (FactType.RECENT_MANUAL_VERIFICATION,), 0.96, "mitigation", "Recent-manual-verification control matched: {facts}.", "kyc_recency", 0.97),
            Rule("AML-R10-PAYROLL", (FactType.EXPECTED_PAYROLL_DAY,), 0.90, "mitigation", "Expected-payroll control matched: {facts}.", "transaction_velocity", 0.91),
        )

    def evaluate(self, transaction: Transaction, facts: tuple[Fact, ...]) -> tuple[Evidence, ...]:
        evidence = (rule.apply(transaction, facts) for rule in self.rules)
        return tuple(item for item in evidence if item is not None)


@dataclass(frozen=True)
class ConflictSpecification:
    risk_rule_id: str
    mitigating_rule_id: str
    kind: str
    explanation: str
    temporal_supersession: bool = False


class ConflictEngine:
    """Detect explicit, semantically declared risk/control disagreements.

    A conflict never deletes either evidence item.  It describes polarity,
    confidence, source-strength, and (where applicable) temporal disagreement
    so policy can qualify risk transparently.
    """

    PAIRS = (
        ConflictSpecification("*", "AML-R08-SOURCE-FUNDS", "source_of_funds", "Risk evidence is qualified by a verified source-of-funds control."),
        ConflictSpecification("AML-R06-KYC", "AML-R09-MANUAL-KYC", "kyc_recency", "Stale KYC evidence conflicts with recent manual verification.", True),
        ConflictSpecification("AML-R03-VELOCITY", "AML-R10-PAYROLL", "expected_payroll", "Velocity evidence conflicts with an expected payroll control."),
    )

    def __init__(self, pairs: tuple[ConflictSpecification, ...] | None = None) -> None:
        # The detection semantics are fixed; only the declared vocabulary of
        # risk/control pairs varies between the transaction-level rule set and
        # the semantic rule set.
        self.pairs = self.PAIRS if pairs is None else pairs

    def detect(self, evidence: tuple[Evidence, ...]) -> tuple[Conflict, ...]:
        by_rule = {item.rule_id: item for item in evidence}
        conflicts: list[Conflict] = []
        for specification in self.pairs:
            mitigation = by_rule.get(specification.mitigating_rule_id)
            if mitigation is None:
                continue
            risks = (
                [item for item in evidence if item.direction == "risk"]
                if specification.risk_rule_id == "*"
                else [by_rule[specification.risk_rule_id]] if specification.risk_rule_id in by_rule else []
            )
            for risk in risks:
                dimensions = ["positive_negative"]
                if abs(risk.confidence - mitigation.confidence) >= 0.08:
                    dimensions.append("confidence_asymmetry")
                if abs(risk.source_reliability - mitigation.source_reliability) >= 0.08:
                    dimensions.append("source_strength_asymmetry")
                if specification.temporal_supersession or (
                    risk.recency_days is not None and mitigation.recency_days in (None, 0)
                ):
                    dimensions.append("old_vs_recent")
                explanation = (
                    f"{specification.explanation} Risk confidence={risk.confidence:.2f}, "
                    f"source reliability={risk.source_reliability:.2f}; mitigating confidence="
                    f"{mitigation.confidence:.2f}, source reliability={mitigation.source_reliability:.2f}. "
                    f"Conflict dimensions: {', '.join(dimensions)}."
                )
                conflicts.append(Conflict(
                    stable_id("C", risk.id, mitigation.id, specification.kind), risk.id, mitigation.id,
                    specification.kind, explanation, tuple(dimensions), risk.confidence, mitigation.confidence,
                    risk.source_reliability, mitigation.source_reliability,
                ))
        return tuple(sorted(conflicts, key=lambda item: item.id))


@dataclass(frozen=True)
class PolicyConfig:
    """Versioned, deterministic decision thresholds; never fitted to labels."""

    review_risk_threshold: float = 0.90
    block_risk_threshold: float = 0.995
    composite_block_rules: tuple[str, ...] = ("AML-R01-LARGE", "AML-R03-VELOCITY", "AML-R07-BEHAVIOUR")
    singleton_review_rules: tuple[str, ...] = ("AML-R02-JURISDICTION", "AML-R03-VELOCITY", "AML-R05-SAR-CONNECTION", "AML-R06-KYC")


class AMLPolicyEngine:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    @staticmethod
    def _noisy_or(items: tuple[Evidence, ...]) -> float:
        complement = 1.0
        for item in items:
            complement *= 1.0 - item.confidence
        return round(1.0 - complement, 6)

    def evaluate(self, evidence: tuple[Evidence, ...], conflicts: tuple[Conflict, ...]) -> tuple[PolicyOutcome, ...]:
        rules = {item.rule_id: item for item in evidence}
        qualified_risk_ids = {item.risk_evidence_id for item in conflicts}
        unqualified_risk = tuple(item for item in evidence if item.direction == "risk" and item.id not in qualified_risk_ids)
        effective_risk = self._noisy_or(unqualified_risk)
        unqualified_rules = frozenset(item.rule_id for item in unqualified_risk)
        policy_metrics: dict[str, float | int | str] = {
            "effective_risk": effective_risk,
            "unqualified_risk_count": len(unqualified_risk),
            "qualified_risk_count": len(qualified_risk_ids),
            "configured_review_threshold": self.config.review_risk_threshold,
            "configured_block_threshold": self.config.block_risk_threshold,
        }
        def has(rule: str) -> bool: return rule in unqualified_rules
        hard_sar = has("AML-R04-SAR")
        sar_connection = has("AML-R05-SAR-CONNECTION") and (has("AML-R01-LARGE") or has("AML-R03-VELOCITY"))
        composite = (
            all(rule in unqualified_rules for rule in self.config.composite_block_rules)
            and effective_risk >= self.config.block_risk_threshold
        )
        review = (
            effective_risk >= self.config.review_risk_threshold
            or any(rule in unqualified_rules for rule in self.config.singleton_review_rules)
        )
        allow = bool(evidence) and not (hard_sar or sar_connection or composite or review)
        outcomes = (
            PolicyOutcome("AML-01", Decision.BLOCK, hard_sar, "Block an unqualified known-SAR originator.", (rules["AML-R04-SAR"].id,) if "AML-R04-SAR" in rules else (), policy_metrics),
            PolicyOutcome("AML-02", Decision.BLOCK, sar_connection, "Block an unqualified SAR connection corroborated by transfer severity.", tuple(rules[key].id for key in ("AML-R05-SAR-CONNECTION", "AML-R01-LARGE", "AML-R03-VELOCITY") if key in rules), policy_metrics),
            PolicyOutcome("AML-20", Decision.BLOCK, composite, "Block only when large value, velocity, and behavioural evidence independently corroborate a high effective-risk score.", tuple(rules[key].id for key in self.config.composite_block_rules if key in rules), policy_metrics),
            PolicyOutcome("AML-21", Decision.REVIEW, not (hard_sar or sar_connection or composite) and review, "Review material unqualified risk; isolated low-severity behavioural novelty is not sufficient.", tuple(item.id for item in unqualified_risk), policy_metrics),
            PolicyOutcome("AML-22", Decision.ALLOW, allow, "Allow when available evidence remains below the review threshold after explicit conflict qualification.", tuple(item.id for item in evidence), policy_metrics),
            PolicyOutcome("AML-00", Decision.ABSTAIN, not evidence, "Abstain when the supplied data yields neither risk nor control evidence.", (), policy_metrics),
        )
        return outcomes


@dataclass(frozen=True)
class PreliminaryRuntimeResult:
    """Label-free output of the frozen runtime before cascade routing."""

    transaction: Transaction
    facts: tuple[Fact, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    policies: tuple[PolicyOutcome, ...]
    decision: DecisionRecord


@dataclass(frozen=True)
class CascadeRoutingProfile:
    """Versioned, explicit eligibility and escalation policy for a cascade."""

    id: str
    eligible_preliminary_decisions: tuple[Decision, ...]
    permit_high_risk_ml_block: bool = True
    score_review_for_prioritisation: bool = False


@dataclass(frozen=True)
class CascadeThresholds:
    """Frozen probability bands selected before evaluation labels are read."""

    low_risk_max: float
    high_risk_min: float
    version: str = "aml-cascade-thresholds/1"

    def band(self, probability: float) -> str:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("ML probability must be between zero and one")
        if probability < self.low_risk_max:
            return "low"
        if probability < self.high_risk_min:
            return "intermediate"
        return "high"


class RuntimeFirstCascadePolicyEngine:
    """The only cascade component permitted to select an operational decision.

    The engine always re-runs the existing conflict and base policy engines on
    the combined evidence.  Routing and the no-downgrade guarantees are then
    declared as versioned policy outcomes rather than hidden in a benchmark.
    """

    VERSION = "aml-runtime-first-cascade-policy/1.0"

    def __init__(self, base_policy: AMLPolicyEngine, conflicts: ConflictEngine, thresholds: CascadeThresholds) -> None:
        self.base_policy = base_policy
        self.conflicts = conflicts
        self.thresholds = thresholds

    @staticmethod
    def _record(outcome: Decision, policy_id: str, triggered: bool, explanation: str, evidence_ids: tuple[str, ...], metrics: dict[str, float | int | str]) -> PolicyOutcome:
        return PolicyOutcome(policy_id, outcome, triggered, explanation, evidence_ids, metrics)

    def eligible(self, preliminary: Decision, profile: CascadeRoutingProfile) -> bool:
        return preliminary in profile.eligible_preliminary_decisions

    def evaluate(
        self,
        preliminary: PreliminaryRuntimeResult,
        profile: CascadeRoutingProfile,
        ml_evidence: Evidence | None,
    ) -> tuple[tuple[Conflict, ...], tuple[PolicyOutcome, ...], DecisionRecord]:
        if ml_evidence is not None and not self.eligible(preliminary.decision.decision, profile):
            raise ValueError("ML evidence supplied for a preliminary decision excluded by the routing profile")
        all_evidence = preliminary.evidence + ((ml_evidence,) if ml_evidence else ())
        all_conflicts = self.conflicts.detect(all_evidence)
        base_outcomes = self.base_policy.evaluate(all_evidence, all_conflicts)
        initial = preliminary.decision.decision
        metrics: dict[str, float | int | str] = {
            "cascade_policy_version": self.VERSION,
            "routing_profile": profile.id,
            "preliminary_decision": initial.value,
            "ml_scored": int(ml_evidence is not None),
        }
        if initial is Decision.BLOCK:
            outcome = self._record(Decision.BLOCK, "AML-CASCADE-01", True, "Preserve preliminary BLOCK; no downstream evidence may downgrade it.", (), metrics)
        elif initial is Decision.REVIEW:
            outcome = self._record(Decision.REVIEW, "AML-CASCADE-02", True, "Preserve preliminary REVIEW; ML may add auditable prioritisation evidence but cannot downgrade it.", (), metrics)
        elif ml_evidence is None:
            outcome = self._record(initial, "AML-CASCADE-03", True, "No ML inference was eligible under the explicit routing profile; preserve the preliminary Runtime decision.", (), metrics)
        else:
            raw_probability = float(ml_evidence.metadata.get("probability", ml_evidence.confidence))
            band = self.thresholds.band(raw_probability)
            qualified = any(conflict.risk_evidence_id == ml_evidence.id for conflict in all_conflicts)
            metrics.update({"ml_probability": raw_probability, "ml_band": band, "ml_evidence_qualified_by_conflict": int(qualified)})
            if qualified:
                final = Decision.ALLOW if initial is Decision.ALLOW else Decision.ABSTAIN
                explanation = "ML risk evidence is explicitly qualified by conflicting control evidence; preserve the eligible preliminary decision."
                policy_id = "AML-CASCADE-04"
            elif band == "low":
                final = Decision.ALLOW
                explanation = "Low-band ML evidence preserves ALLOW and resolves ABSTAIN to ALLOW under the frozen cascade policy."
                policy_id = "AML-CASCADE-05"
            elif band == "intermediate":
                final = Decision.REVIEW
                explanation = "Intermediate-band ML evidence triggers REVIEW through the versioned Runtime cascade policy."
                policy_id = "AML-CASCADE-06"
            elif profile.permit_high_risk_ml_block:
                final = Decision.BLOCK
                explanation = "High-band ML evidence triggers BLOCK only through the explicit, versioned Runtime cascade policy."
                policy_id = "AML-CASCADE-07"
            else:
                final = Decision.REVIEW
                explanation = "High-band ML evidence is constrained to REVIEW because this routing profile prohibits ML-initiated BLOCK."
                policy_id = "AML-CASCADE-08"
            outcome = self._record(final, policy_id, True, explanation, (ml_evidence.id,), metrics)
        policies = base_outcomes + (outcome,)
        rationale = outcome.explanation
        decision = DecisionRecord(outcome.outcome, rationale, (outcome.policy_id,))
        return all_conflicts, policies, decision


@dataclass(frozen=True)
class RuntimeResult:
    transaction: Transaction
    facts: tuple[Fact, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    policies: tuple[PolicyOutcome, ...]
    decision: DecisionRecord
    audit: dict

    def to_dict(self) -> dict:
        return {
            "transaction": self.transaction.to_dict(), "facts": [item.to_dict() for item in self.facts],
            "evidence": [item.to_dict() for item in self.evidence], "conflicts": [item.to_dict() for item in self.conflicts],
            "policies": [item.to_dict() for item in self.policies], "decision": self.decision.to_dict(), "audit": self.audit,
        }


class AuditEngine:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, dataset: AMLSimDataset, transaction: Transaction, facts: tuple[Fact, ...], evidence: tuple[Evidence, ...], conflicts: tuple[Conflict, ...], policies: tuple[PolicyOutcome, ...], decision: DecisionRecord, extra: Mapping[str, object] | None = None) -> dict:
        extra = dict(extra or {})
        audit_identity = str(extra.get("audit_identity", ""))
        audit_id = stable_id("AUD", dataset.fingerprint, transaction.id, decision.decision.value, *decision.policy_ids, audit_identity)
        record = {
            "audit_id": audit_id, "transaction_id": transaction.id, "dataset_fingerprint": dataset.fingerprint,
            "runtime_version": RUNTIME_VERSION, "execution_time_ms": 0, "execution_time_model": "deterministic-logical-clock",
            "facts": [item.to_dict() for item in facts], "evidence": [item.to_dict() for item in evidence],
            "rules": [item.rule_id for item in evidence], "conflicts": [item.to_dict() for item in conflicts],
            "policies": [item.to_dict() for item in policies], "decision": decision.to_dict(),
        }
        if extra:
            record["cascade"] = extra
        path = self.directory / f"{transaction.id}-{audit_id}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"id": audit_id, "path": str(path), "execution_time_ms": 0, "runtime_version": RUNTIME_VERSION}


class AMLDecisionRuntime:
    def __init__(self, dataset: AMLSimDataset, audit_directory: str | Path = "artifacts/aml_audit") -> None:
        self.dataset = dataset
        self.graph = TransactionGraph(dataset)
        self.facts = FactExtractor()
        self.rules = AMLRuleEngine()
        self.conflicts = ConflictEngine()
        self.policies = AMLPolicyEngine()
        self.audit_engine = AuditEngine(audit_directory)

    def evaluate_preliminary(self, transaction_id: str) -> PreliminaryRuntimeResult:
        """Evaluate facts through policy selection without creating an audit.

        Streaming experiments use this label-free method for chronological
        training rows.  Operational `execute` remains the audited entrypoint.
        """
        transaction = self.graph.by_id.get(transaction_id)
        if transaction is None:
            raise KeyError(f"Unknown transaction ID: {transaction_id}")
        facts = self.facts.extract(transaction, self.graph)
        evidence = self.rules.evaluate(transaction, facts)
        conflicts = self.conflicts.detect(evidence)
        policies = self.policies.evaluate(evidence, conflicts)
        triggered = tuple(item for item in policies if item.triggered)
        selected = next((decision for decision in (Decision.BLOCK, Decision.REVIEW, Decision.ALLOW, Decision.ABSTAIN) if any(item.outcome == decision for item in triggered)), Decision.ABSTAIN)
        selected_policies = tuple(item.policy_id for item in triggered if item.outcome == selected)
        rationale = next((item.explanation for item in triggered if item.outcome == selected), "No policy was triggered.")
        decision = DecisionRecord(selected, rationale, selected_policies)
        return PreliminaryRuntimeResult(transaction, facts, evidence, conflicts, policies, decision)

    def execute(self, transaction_id: str) -> RuntimeResult:
        preliminary = self.evaluate_preliminary(transaction_id)
        transaction = preliminary.transaction
        facts, evidence, conflicts, policies, decision = (
            preliminary.facts,
            preliminary.evidence,
            preliminary.conflicts,
            preliminary.policies,
            preliminary.decision,
        )
        audit = self.audit_engine.record(self.dataset, transaction, facts, evidence, conflicts, policies, decision)
        return RuntimeResult(transaction, facts, evidence, conflicts, policies, decision, audit)

    def replay(self, transaction_id: str) -> RuntimeResult:
        """Re-execute from the immutable loaded dataset; deterministic fields remain identical."""
        return self.execute(transaction_id)


@dataclass(frozen=True)
class CascadeRuntimeResult:
    preliminary: PreliminaryRuntimeResult
    ml_evidence: Evidence | None
    conflicts: tuple[Conflict, ...]
    policies: tuple[PolicyOutcome, ...]
    decision: DecisionRecord
    audit: dict

    def to_dict(self) -> dict:
        return {
            "preliminary": {
                "transaction": self.preliminary.transaction.to_dict(),
                "facts": [item.to_dict() for item in self.preliminary.facts],
                "evidence": [item.to_dict() for item in self.preliminary.evidence],
                "conflicts": [item.to_dict() for item in self.preliminary.conflicts],
                "policies": [item.to_dict() for item in self.preliminary.policies],
                "decision": self.preliminary.decision.to_dict(),
            },
            "ml_evidence": self.ml_evidence.to_dict() if self.ml_evidence else None,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "policies": [item.to_dict() for item in self.policies],
            "decision": self.decision.to_dict(),
            "audit": self.audit,
        }


class RuntimeFirstCascade:
    """Audited Runtime → typed evidence → Runtime decision orchestrator."""

    REQUIRED_REPLAY_PINS = frozenset({
        "model_hash", "thresholds_hash", "feature_schema_hash", "rules_hash",
        "policy_hash", "input_snapshot_hash", "training_window_identity",
    })

    def __init__(self, runtime: AMLDecisionRuntime, audit_directory: str | Path, thresholds: CascadeThresholds) -> None:
        self.runtime = runtime
        self.policy = RuntimeFirstCascadePolicyEngine(runtime.policies, runtime.conflicts, thresholds)
        self.audit_engine = AuditEngine(audit_directory)

    def finalise(self, preliminary: PreliminaryRuntimeResult, profile: CascadeRoutingProfile, ml_evidence: Evidence | None, pins: Mapping[str, str], write_audit: bool = True) -> CascadeRuntimeResult:
        missing = self.REQUIRED_REPLAY_PINS - set(pins)
        if missing:
            raise ValueError(f"cascade audit pins missing required values: {sorted(missing)}")
        conflicts, policies, decision = self.policy.evaluate(preliminary, profile, ml_evidence)
        audit_extra = {
            "audit_identity": stable_id("CAS", profile.id, pins["model_hash"], pins["thresholds_hash"], pins["feature_schema_hash"]),
            "routing_profile": profile.id,
            "preliminary_decision": preliminary.decision.decision.value,
            "pins": dict(sorted(pins.items())),
            "ml_evidence_id": ml_evidence.id if ml_evidence else "",
        }
        audit = (
            self.audit_engine.record(self.runtime.dataset, preliminary.transaction, preliminary.facts,
                                     preliminary.evidence + ((ml_evidence,) if ml_evidence else ()),
                                     conflicts, policies, decision, audit_extra)
            if write_audit else {"id": stable_id("AUD", preliminary.transaction.id, decision.decision.value), "runtime_version": RUNTIME_VERSION}
        )
        return CascadeRuntimeResult(preliminary, ml_evidence, conflicts, policies, decision, audit)

    def replay(self, preliminary: PreliminaryRuntimeResult, profile: CascadeRoutingProfile, ml_evidence: Evidence | None, pins: Mapping[str, str]) -> CascadeRuntimeResult:
        """Reproduce a cascade decision only when every version pin is present."""
        return self.finalise(preliminary, profile, ml_evidence, pins, write_audit=False)
