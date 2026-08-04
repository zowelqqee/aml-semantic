"""Tests for the Semantic Context Layer.

The properties that matter here are causality (no object may read the event it
describes or anything later), honesty (absent inputs are recorded, never
defaulted), and determinism (the same prior state yields the same objects).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aml_runtime.models import Decision, Evidence, Transaction
from aml_runtime.semantic import (
    ONTOLOGY_HASH,
    EntityResolver,
    SemanticContextLayer,
    SemanticDecisionRuntime,
    SemanticType,
    account_key,
    confidence_for,
    jurisdiction_of,
    semantic_replay_pins,
)
from aml_runtime.semantic.ontology import (
    BASELINE_MINIMUM_EVENTS,
    ESTABLISHED_RELATIONSHIP_MINIMUM,
    SPECIFICATIONS,
    VALUE_REGIME_BREAK_MULTIPLE,
)
from aml_runtime.semantic.runtime import SEMANTIC_RULES, STRUCTURAL_EXPLANATIONS

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = ROOT / "data" / "ibm_aml_data" / "HI-Small_accounts.csv"

ALICE = "10:AAA"
BOB = "20:BBB"
CAROL = "30:CCC"
DMITRI = "50:RUS"


class _StubResolver(EntityResolver):
    """A hand-built reference snapshot, so tests do not depend on the corpus."""

    def __init__(self, records: dict[str, tuple[str, str, str]]) -> None:
        super().__init__()
        from aml_runtime.semantic.entities import ResolvedAccount
        from aml_runtime.semantic.ontology import EntityForm

        for key, (customer, form, jurisdiction) in records.items():
            self._accounts[key] = ResolvedAccount(key, customer, EntityForm(form), key.split(":")[0], jurisdiction)
        self.snapshot_hash = "stub-snapshot"


def _resolver() -> _StubResolver:
    return _StubResolver({
        ALICE: ("cust-1", "Corporation", "United States"),
        BOB: ("cust-2", "Corporation", "Germany"),
        CAROL: ("cust-1", "Corporation", "United States"),
        DMITRI: ("cust-3", "Partnership", "Russia"),
    })


def _transaction(index: int, originator: str, beneficiary: str, amount: float, minute: int = 0, currency: str = "US Dollar", payment_type: str = "Cheque", payment_currency: str | None = None) -> Transaction:
    return Transaction(
        id=f"T-{index:04d}",
        timestamp=f"2022-09-01T{minute // 60:02d}:{minute % 60:02d}:00",
        originator_account_id=originator,
        beneficiary_account_id=beneficiary,
        amount=amount,
        currency=currency,
        payment_type=payment_type,
        metadata={"payment_currency": payment_currency or currency},
    )


def _layer() -> SemanticContextLayer:
    return SemanticContextLayer(_resolver())


def _prime(layer: SemanticContextLayer, count: int, originator: str = ALICE, beneficiary: str = BOB, amount: float = 100.0) -> None:
    for index in range(count):
        layer.commit(_transaction(index, originator, beneficiary, amount, minute=index))


# -- ontology ---------------------------------------------------------------

def test_every_semantic_type_has_a_specification():
    assert set(SPECIFICATIONS) == set(SemanticType)


def test_every_semantic_rule_consumes_a_declared_type():
    for rule in SEMANTIC_RULES:
        assert rule.consumes in SPECIFICATIONS
        assert rule.direction in {"risk", "mitigation"}


def test_confidence_is_prior_times_support_times_coverage():
    value, explanation = confidence_for(SemanticType.VALUE_REGIME, observations=8, present_inputs=2)
    specification = SPECIFICATIONS[SemanticType.VALUE_REGIME]
    expected = specification.prior * (8 / (8 + specification.half_support)) * 1.0
    assert value == pytest.approx(round(expected, 6))
    assert "prior" in explanation and "support" in explanation and "coverage" in explanation


def test_missing_inputs_reduce_confidence_rather_than_defaulting():
    full, _ = confidence_for(SemanticType.HIGH_RISK_JURISDICTION_EXPOSURE, 1, 3)
    partial, _ = confidence_for(SemanticType.HIGH_RISK_JURISDICTION_EXPOSURE, 1, 1)
    assert 0.0 < partial < full


def test_ontology_hash_is_stable_across_imports():
    import importlib

    module = importlib.import_module("aml_runtime.semantic.ontology")
    assert module.ONTOLOGY_HASH == ONTOLOGY_HASH


# -- entity resolution ------------------------------------------------------

def test_jurisdiction_comes_from_the_bank_name():
    assert jurisdiction_of("Germany Bank #4507") == "Germany"
    assert jurisdiction_of("Savings Bank of Topeka #12") == "United States"
    assert jurisdiction_of("Crytpo Bank #7") == "virtual-asset"


def test_account_key_matches_the_ml_feature_join_key():
    assert account_key("010", "8000ebd30") == "10:8000EBD30"


@pytest.mark.skipif(not ACCOUNTS.exists(), reason="IBM AML account reference data not present")
def test_reference_data_resolves_a_restricted_working_set():
    with ACCOUNTS.open(newline="", encoding="utf-8-sig") as handle:
        rows = [next(csv.DictReader(handle)) for _ in range(1)]
    key = account_key(rows[0]["Bank ID"], rows[0]["Account Number"])
    resolver = EntityResolver().load(ACCOUNTS, restrict_to={key})
    assert resolver.resolved_count == 1
    assert resolver.resolve(key).customer_id == rows[0]["Entity ID"]
    assert resolver.snapshot_hash


# -- causality --------------------------------------------------------------

def test_observation_cannot_see_the_event_it_describes():
    layer = _layer()
    transaction = _transaction(0, ALICE, BOB, 100.0)
    reading = layer.observe(transaction, 0)
    assert reading.by_type(SemanticType.NON_INFORMATIVE_NOVELTY) is not None
    assert layer.events_committed == 0
    layer.commit(transaction)
    assert layer.events_committed == 1


def test_replaying_the_same_prior_state_reproduces_the_object_set():
    first, second = _layer(), _layer()
    _prime(first, 10)
    _prime(second, 10)
    transaction = _transaction(99, ALICE, CAROL, 400.0, minute=200)
    left = first.observe(transaction, 99)
    right = second.observe(transaction, 99)
    assert [item.id for item in left.objects] == [item.id for item in right.objects]
    assert left.context_state_hash == right.context_state_hash


# -- profile and event inference -------------------------------------------

def test_no_baseline_blocks_every_regime_claim():
    layer = _layer()
    _prime(layer, BASELINE_MINIMUM_EVENTS - 1)
    reading = layer.observe(_transaction(50, ALICE, BOB, 10_000_000.0, minute=100), 50)
    types = reading.types()
    assert SemanticType.NO_ESTABLISHED_BASELINE in types
    assert SemanticType.UNSCALED_VALUE in types
    # A magnitude with no reference frame is never called large.
    assert SemanticType.UNEXPECTED_LARGE_TRANSFER not in types


def test_value_is_situated_in_the_party_own_regime():
    layer = _layer()
    _prime(layer, 10, amount=100.0)
    routine = layer.observe(_transaction(50, ALICE, BOB, 50.0, minute=100), 50)
    assert SemanticType.ROUTINE_VALUE_TRANSFER in routine.types()
    broken = layer.observe(_transaction(51, ALICE, BOB, 100.0 * VALUE_REGIME_BREAK_MULTIPLE + 1, minute=100), 51)
    assert SemanticType.UNEXPECTED_LARGE_TRANSFER in broken.types()


def test_the_same_amount_means_different_things_to_different_parties():
    """The property a fixed 50,000 threshold cannot express."""
    small = _layer()
    _prime(small, 10, amount=100.0)
    large = _layer()
    _prime(large, 10, amount=1_000_000.0)
    amount = 5_000.0
    assert SemanticType.UNEXPECTED_LARGE_TRANSFER in small.observe(_transaction(60, ALICE, BOB, amount, minute=100), 60).types()
    assert SemanticType.ROUTINE_VALUE_TRANSFER in large.observe(_transaction(60, ALICE, BOB, amount, minute=100), 60).types()


def test_self_posting_is_a_book_entry_with_no_counterparty():
    layer = _layer()
    reading = layer.observe(_transaction(0, ALICE, ALICE, 900.0), 0)
    types = reading.types()
    assert SemanticType.INTERNAL_BOOK_ENTRY in types
    assert not (types & {SemanticType.FIRST_CONTACT, SemanticType.NON_INFORMATIVE_NOVELTY,
                         SemanticType.RECENTLY_CREATED_RELATIONSHIP, SemanticType.ESTABLISHED_RELATIONSHIP})


def test_two_accounts_of_one_customer_are_an_intra_customer_transfer():
    layer = _layer()
    reading = layer.observe(_transaction(0, ALICE, CAROL, 900.0), 0)
    assert SemanticType.INTRA_CUSTOMER_TRANSFER in reading.types()


def test_novelty_is_informative_only_against_a_counterparty_baseline():
    sparse = _layer()
    assert SemanticType.NON_INFORMATIVE_NOVELTY in sparse.observe(_transaction(0, ALICE, BOB, 100.0), 0).types()

    known = _layer()
    for index in range(BASELINE_MINIMUM_EVENTS):
        known.commit(_transaction(index, ALICE, f"90:CP{index}", 100.0, minute=index))
    assert SemanticType.FIRST_CONTACT in known.observe(_transaction(50, ALICE, BOB, 100.0, minute=60), 50).types()


def test_repeated_pair_becomes_an_established_relationship():
    layer = _layer()
    for index in range(ESTABLISHED_RELATIONSHIP_MINIMUM):
        layer.commit(_transaction(index, ALICE, BOB, 100.0, minute=index))
    assert SemanticType.ESTABLISHED_RELATIONSHIP in layer.observe(_transaction(50, ALICE, BOB, 100.0, minute=60), 50).types()


def test_cross_jurisdiction_and_virtual_asset_come_from_reference_data():
    layer = _layer()
    reading = layer.observe(_transaction(0, ALICE, BOB, 100.0), 0)
    assert SemanticType.CROSS_JURISDICTION_TRANSFER in reading.types()
    assert SemanticType.VIRTUAL_ASSET_EXPOSURE not in reading.types()


def test_currency_conversion_is_read_from_both_legs():
    layer = _layer()
    reading = layer.observe(_transaction(0, ALICE, BOB, 100.0, currency="Euro", payment_currency="US Dollar"), 0)
    assert SemanticType.CURRENCY_CONVERSION_TRANSFER in reading.types()


# -- honesty ----------------------------------------------------------------

def test_absent_inputs_are_recorded_as_withheld_claims_and_coverage_gaps():
    reading = _layer().observe(_transaction(0, ALICE, BOB, 100.0), 0)
    withheld = {item.type for item in reading.withheld}
    assert {"SalaryDistribution", "MortgagePayment", "TaxPayment"}.issubset(withheld)
    assert "sar_feed" in reading.coverage_gaps and "kyc_dates" in reading.coverage_gaps
    for item in reading.withheld:
        assert item.missing_inputs


def test_unresolvable_account_withholds_identity_rather_than_assuming_it():
    reading = _layer().observe(_transaction(0, ALICE, "99:UNKNOWN", 100.0), 0)
    assert any(item.type == "ResolvedCounterpartyIdentity" for item in reading.withheld)
    assert SemanticType.CROSS_JURISDICTION_TRANSFER not in reading.types()


# -- decisions --------------------------------------------------------------

def _runtime() -> SemanticDecisionRuntime:
    return SemanticDecisionRuntime(_layer())


def test_undetermined_state_abstains_instead_of_allowing():
    runtime = _runtime()
    result = runtime.evaluate(_transaction(0, ALICE, "40:DDD", 100.0), 0)
    assert result.decision.decision is Decision.ABSTAIN
    assert result.policies[-1].metrics["semantic_state"] == "undetermined"
    assert SemanticDecisionRuntime.routes_to_ml(result)


def test_structural_explanation_allows_and_is_never_routed_to_ml():
    runtime = _runtime()
    result = runtime.evaluate(_transaction(0, ALICE, ALICE, 10_000_000.0), 0)
    assert result.decision.decision is Decision.ALLOW
    assert result.context.types() & STRUCTURAL_EXPLANATIONS
    assert not SemanticDecisionRuntime.routes_to_ml(result)


def test_review_requires_two_independent_semantic_topics():
    runtime = _runtime()
    for index in range(10):
        runtime.context.commit(_transaction(index, ALICE, f"90:CP{index}", 100.0, minute=index))
    # Jurisdiction exposure alone: the counterparty novelty is qualified by a
    # value inside the party's own regime, leaving one unqualified topic.
    single = runtime.evaluate(_transaction(50, ALICE, DMITRI, 10.0, minute=100), 50)
    assert single.policies[0].metrics["independent_unqualified_topics"] == 1
    assert single.decision.decision is Decision.ALLOW
    # Add a value-regime break and the novelty stops being qualified: REVIEW.
    corroborated = runtime.evaluate(
        _transaction(51, ALICE, DMITRI, 100.0 * VALUE_REGIME_BREAK_MULTIPLE + 1, minute=100), 51)
    assert corroborated.policies[0].metrics["independent_unqualified_topics"] >= 2
    assert corroborated.decision.decision is Decision.REVIEW


def test_conflicts_actually_fire_in_the_semantic_vocabulary():
    """The v0.2 conflict engine measured 0.0 conflicts on this source."""
    runtime = _runtime()
    result = runtime.evaluate(_transaction(0, ALICE, ALICE, 5_000.0, payment_type="Cash"), 0)
    assert any(item.kind == "structural_explanation" for item in result.conflicts)
    qualified = {item.risk_evidence_id for item in result.conflicts}
    assert any(item.id in qualified for item in result.evidence if item.direction == "risk")


def test_every_decision_is_explainable_in_semantic_objects():
    runtime = _runtime()
    result = runtime.evaluate(_transaction(0, ALICE, BOB, 100.0), 0)
    rationale = result.semantic_rationale
    assert not any(token in rationale for token in ("AML-R0", "AML-R1"))
    for item in result.evidence:
        assert item.metadata["semantic_type"] in {value.value for value in SemanticType}
        assert item.source.startswith("semantic-context/")


def test_ml_evidence_cannot_manufacture_the_second_independent_topic():
    runtime = _runtime()
    base = runtime.evaluate(_transaction(0, ALICE, BOB, 100.0), 0)
    evidence = Evidence("E-ml", "ML/Test", (), 0.80, "probability", base.transaction.timestamp,
                        "ML-Test", "risk", "ml_probability", 1.0, 0, {"probability": "0.80"})
    fused = runtime.with_ml_evidence(base, evidence, high_band=0.90)
    assert fused.decision.decision is base.decision.decision
    assert fused.policies[-1].policy_id == "SEM-ML-03"


def test_high_band_ml_lifts_an_abstention_to_review_only_through_policy():
    runtime = _runtime()
    base = runtime.evaluate(_transaction(0, ALICE, "40:DDD", 100.0), 0)
    assert base.decision.decision is Decision.ABSTAIN
    evidence = Evidence("E-ml", "ML/Test", (), 0.99, "probability", base.transaction.timestamp,
                        "ML-Test", "risk", "ml_probability", 1.0, 0, {"probability": "0.99"})
    fused = runtime.with_ml_evidence(base, evidence, high_band=0.90)
    assert fused.decision.decision is Decision.REVIEW
    assert fused.decision.policy_ids == ("SEM-ML-02",)
    assert fused.routed_to_ml


# -- audit and replay -------------------------------------------------------

def test_audit_record_carries_objects_withheld_and_coverage():
    runtime = _runtime()
    record = runtime.evaluate(_transaction(0, ALICE, BOB, 100.0), 0).audit_record()
    semantic = record["semantic"]
    assert semantic["ontology_hash"] == ONTOLOGY_HASH
    assert semantic["objects"] and semantic["withheld"] and semantic["coverage_gaps"]
    for item in semantic["objects"]:
        assert item["causal_evidence"]["window"] == "prior-only"
        assert item["confidence_explanation"]
        assert item["meaning"]


def test_replay_pins_cover_every_declared_input():
    runtime = _runtime()
    result = runtime.evaluate(_transaction(0, ALICE, BOB, 100.0), 0)
    pins = semantic_replay_pins("stub-snapshot", result)
    assert {"ontology_hash", "inference_rules_hash", "semantic_rules_hash", "policy_hash",
            "entity_snapshot_hash", "context_state_hash", "semantic_object_set_hash",
            "input_snapshot_hash"} == set(pins)
    assert all(pins.values())
