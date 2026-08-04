from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest

from aml_runtime import AMLDecisionRuntime, AMLSimLoader
from aml_runtime.models import Decision, Evidence, FactType
from aml_runtime.research import ResearchMetrics
from aml_runtime.runtime import AMLPolicyEngine
from aml_runtime.runtime import CascadeRoutingProfile, CascadeThresholds, RuntimeFirstCascade
from aml_runtime.runtime_first_cascade import ModelArtifact, ProbabilityEvidenceProvider, StreamingRuntimeGraph
from aml_runtime.runtime_first_v2 import FusionConfig, RuntimeSummary, _fuse
from aml_runtime.visualize import write_decision_graph


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "amlsim_sample"


def runtime(tmp_path):
    return AMLDecisionRuntime(AMLSimLoader().load(DATA), tmp_path / "audit")


def test_loader_builds_entities_and_uses_stable_dataset_fingerprint():
    first = AMLSimLoader().load(DATA)
    second = AMLSimLoader().load(DATA)
    assert len(first.transactions) == 8
    assert first.fingerprint == second.fingerprint
    assert first.accounts["A-500"].is_sar
    assert first.countries["IR"].high_risk


def test_explicit_aml_rules_produce_evidence_not_decisions(tmp_path):
    result = runtime(tmp_path).execute("TX-1004")
    fact_types = {item.type for item in result.facts}
    assert {FactType.LARGE_TRANSFER, FactType.HIGH_RISK_COUNTRY, FactType.OLD_KYC}.issubset(fact_types)
    assert any(item.rule_id == "AML-R01-LARGE" for item in result.evidence)
    assert any(item.rule_id == "AML-R02-JURISDICTION" for item in result.evidence)
    assert result.decision.decision is Decision.REVIEW


def test_known_sar_and_connection_block(tmp_path):
    result = runtime(tmp_path).execute("TX-1006")
    assert result.decision.decision is Decision.BLOCK
    assert "AML-01" in result.decision.policy_ids
    assert any(item.type is FactType.PREVIOUS_SAR for item in result.facts)
    assert any(item.type is FactType.CONNECTED_TO_SAR for item in result.facts)


def test_conflict_is_explicit_and_can_allow_qualified_risk(tmp_path):
    result = runtime(tmp_path).execute("TX-1008")
    assert result.decision.decision is Decision.ALLOW
    conflict = next(item for item in result.conflicts if item.kind == "kyc_recency")
    assert {"positive_negative", "confidence_asymmetry", "source_strength_asymmetry", "old_vs_recent"}.issubset(conflict.dimensions)
    assert conflict.mitigating_confidence > conflict.risk_confidence
    assert any(item.policy_id == "AML-22" and item.triggered for item in result.policies)


def test_v02_policy_has_deterministic_allow_review_block_and_abstain_paths(tmp_path):
    subject = runtime(tmp_path)
    assert subject.execute("TX-1004").decision.decision is Decision.REVIEW
    assert subject.execute("TX-1006").decision.decision is Decision.BLOCK
    isolated_behaviour = Evidence("E-test", "test", (), 0.84, "isolated new beneficiary", "2026-01-01T00:00:00", "AML-R07-BEHAVIOUR", "risk", "counterparty_behaviour", 0.72)
    allow = AMLPolicyEngine().evaluate((isolated_behaviour,), ())
    assert next(item for item in allow if item.policy_id == "AML-22").triggered
    abstain = AMLPolicyEngine().evaluate((), ())
    assert next(item for item in abstain if item.policy_id == "AML-00").triggered


def test_research_metrics_are_label_free_and_measure_rule_contribution(tmp_path):
    subject = runtime(tmp_path)
    metrics = ResearchMetrics(tuple(rule.id for rule in subject.rules.rules))
    for transaction_id in ("TX-1001", "TX-1004", "TX-1006", "TX-1008"):
        metrics.observe(subject.execute(transaction_id))
    summary = metrics.to_dict()
    assert summary["transactions_measured"] == 4
    assert summary["decision_distribution"]["ALLOW"] == 1
    assert summary["decision_distribution"]["REVIEW"] == 2
    assert summary["decision_distribution"]["BLOCK"] == 1
    assert summary["rule_contribution"]["AML-R01-LARGE"]["contribution_to_review"] == 1
    assert "laundering" not in repr(summary).lower()


def test_audit_replay_and_graph_are_deterministic(tmp_path):
    subject = runtime(tmp_path)
    first = subject.execute("TX-1004")
    replay = subject.replay("TX-1004")
    assert first.to_dict() == replay.to_dict()
    audit_file = Path(first.audit["path"])
    assert audit_file.exists()
    assert '"runtime_version": "aml-decision-runtime/0.2.0"' in audit_file.read_text(encoding="utf-8")
    assert '"effective_risk"' in audit_file.read_text(encoding="utf-8")
    graph = write_decision_graph(first, tmp_path / "trace.dot")
    assert "digraph aml_decision" in graph.read_text(encoding="utf-8")


def test_ibm_amlsim_column_layout_is_accepted(tmp_path):
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text(
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Is Laundering\n"
        "2026-01-01 10:00,B1,origin,B2,destination,1234.50,USD,0\n", encoding="utf-8"
    )
    dataset = AMLSimLoader().load(csv_path)
    transaction = dataset.transactions[0]
    assert transaction.originator_account_id == "origin"
    assert transaction.beneficiary_account_id == "destination"
    assert transaction.amount == 1234.50


def _cascade_pins():
    return {
        "model_hash": "model-sha", "thresholds_hash": "thresholds-sha", "feature_schema_hash": "features-sha",
        "rules_hash": "rules-sha", "policy_hash": "policy-sha", "input_snapshot_hash": "input-sha",
        "training_window_identity": "chronological[0:400000]",
    }


def _provider():
    return ProbabilityEvidenceProvider(ModelArtifact(
        "TestModel", "1.0", "model-sha", "features-sha", ("amount",), "chronological[0:400000]",
        CascadeThresholds(0.10, 0.90),
    ))


def test_runtime_first_provider_never_receives_labels_and_serializes_provenance(tmp_path):
    subject = runtime(tmp_path)
    preliminary = subject.evaluate_preliminary("TX-1004")
    evidence = _provider().provide(preliminary.transaction, 0.61)
    rendered = evidence.to_dict()
    assert "laundering" not in repr(rendered).lower()
    assert rendered["metadata"]["threshold_band"] == "intermediate"
    assert rendered["metadata"]["model_hash"] == "model-sha"
    cascade = RuntimeFirstCascade(subject, tmp_path / "cascade", CascadeThresholds(0.10, 0.90))
    profile = CascadeRoutingProfile("test", (Decision.ALLOW, Decision.ABSTAIN, Decision.REVIEW))
    result = cascade.finalise(preliminary, profile, evidence, _cascade_pins())
    audit = json.loads(Path(result.audit["path"]).read_text(encoding="utf-8"))
    assert audit["evidence"][-1]["metadata"]["feature_schema_hash"] == "features-sha"
    assert audit["cascade"]["pins"]["model_hash"] == "model-sha"


def test_runtime_first_cascade_preserves_block_and_review_and_has_no_future_graph_state(tmp_path):
    subject = runtime(tmp_path)
    cascade = RuntimeFirstCascade(subject, tmp_path / "cascade", CascadeThresholds(0.10, 0.90))
    allow_abstain = CascadeRoutingProfile("allow-abstain", (Decision.ALLOW, Decision.ABSTAIN))
    block = subject.evaluate_preliminary("TX-1006")
    assert cascade.finalise(block, allow_abstain, None, _cascade_pins(), write_audit=False).decision.decision is Decision.BLOCK
    review = subject.evaluate_preliminary("TX-1004")
    low = _provider().provide(review.transaction, 0.01)
    review_profile = CascadeRoutingProfile("non-block", (Decision.ALLOW, Decision.ABSTAIN, Decision.REVIEW))
    assert cascade.finalise(review, review_profile, low, _cascade_pins(), write_audit=False).decision.decision is Decision.REVIEW
    dataset = AMLSimLoader().load(DATA)
    graph = StreamingRuntimeGraph(dataset)
    future = dataset.transactions[-1]
    graph.by_id[future.id] = future
    # Nothing is committed, so the graph facade cannot expose the future event.
    assert len(graph.prior_outbound(future, hours=24)) == 0


def test_runtime_first_replay_requires_pins_and_identical_artifacts_reproduce_decision(tmp_path):
    subject = runtime(tmp_path)
    cascade = RuntimeFirstCascade(subject, tmp_path / "cascade", CascadeThresholds(0.10, 0.90))
    preliminary = subject.evaluate_preliminary("TX-1004")
    profile = CascadeRoutingProfile("review", (Decision.REVIEW,))
    evidence = _provider().provide(preliminary.transaction, 0.11)
    with pytest.raises(ValueError, match="pins missing"):
        cascade.replay(preliminary, profile, evidence, {})
    first = cascade.replay(preliminary, profile, evidence, _cascade_pins())
    second = cascade.replay(preliminary, profile, evidence, _cascade_pins())
    assert first.decision == second.decision
    assert first.conflicts == second.conflicts


def test_v2_fusion_does_not_escalate_a_single_weak_runtime_family():
    config = FusionConfig("runtime_uncertainty", 0.80, 0.98, 0.65, 1)
    summaries = [RuntimeSummary(("counterparty_behaviour",), 0.60, Decision.REVIEW.value)]
    first = _fuse(np.array([0.50]), summaries, config)
    second = _fuse(np.array([0.50]), summaries, config)
    assert first[0][0] == Decision.ALLOW.value
    assert first[0].tolist() == second[0].tolist()
    assert first[3][0]["correlated_evidence_deduplicated"] is True
