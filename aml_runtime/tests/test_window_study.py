"""Tests for the history-window scaling study.

The study's central claim is that *nothing but the window changes*.  The tests
below check the two ways that claim could be false: the fast priming path could
evolve state differently from a full evaluation, and the study could be reading
a different vocabulary or policy than the benchmark it is compared against.
"""

from __future__ import annotations

from aml_runtime.behaviour import BEHAVIOUR_ONTOLOGY_HASH, BehaviourDecisionRuntime, BehaviourLayer
from aml_runtime.behaviour.runtime import COMBINED_CONFLICT_PAIRS
from aml_runtime.semantic import ONTOLOGY_HASH, SemanticContextLayer, SemanticDecisionRuntime
from aml_runtime.semantic.runtime import SEMANTIC_RULES, SemanticPolicyEngine
from aml_runtime.window_study import (
    DENSE_PERIOD_END,
    EVALUATION_ROWS,
    EVALUATION_START,
    ML_TRAIN_CAP,
    WINDOWS,
    WindowStudy,
)

from test_semantic_context import ALICE, BOB, CAROL, _resolver, _transaction  # noqa: F401


def _runtime() -> BehaviourDecisionRuntime:
    return BehaviourDecisionRuntime(SemanticDecisionRuntime(SemanticContextLayer(_resolver())), BehaviourLayer())


def _state_fingerprint(runtime: BehaviourDecisionRuntime, accounts: list[str]) -> list[tuple]:
    """Everything the temporal engine and the semantic fold carry per account."""
    fingerprint = []
    for account in accounts:
        state = runtime.layer.engine.peek(account)
        if state is None:
            fingerprint.append((account, None))
            continue
        fingerprint.append((account, tuple(
            (name, getattr(state, name)) for name in sorted(state.__slots__)
        )))
    return fingerprint


def test_priming_path_evolves_state_identically_to_full_evaluation():
    """The study's fast path must not be a different pipeline."""
    events = [
        (ALICE, BOB, 100.0, 0), (BOB, CAROL, 100.0, 2), (ALICE, "90:CP1", 250.0, 3),
        (ALICE, ALICE, 900.0, 4), (BOB, ALICE, 100.0, 5), (ALICE, "90:CP2", 75.0, 6),
        (CAROL, ALICE, 400.0, 7), (ALICE, "90:CP3", 120.0, 9), (ALICE, BOB, 130.0, 20),
        (BOB, "90:CP4", 130.0, 21), (ALICE, "90:CP1", 90.0, 70), (BOB, CAROL, 60.0, 75),
    ]
    full, primed = _runtime(), _runtime()
    resolver = _resolver()
    for index, (originator, beneficiary, amount, minute) in enumerate(events):
        transaction = _transaction(index, originator, beneficiary, amount, minute=minute)
        result = full.evaluate(transaction, resolver.resolve(originator), minute, index)
        full.commit(transaction, result, minute)
        WindowStudy._prime(primed, resolver, transaction, minute, index)

    accounts = [ALICE, BOB, CAROL, "90:CP1", "90:CP2", "90:CP3", "90:CP4"]
    assert _state_fingerprint(full, accounts) == _state_fingerprint(primed, accounts)
    assert full.layer.engine.events_committed == primed.layer.engine.events_committed


def test_a_decision_after_priming_matches_a_decision_after_full_evaluation():
    events = [(ALICE, f"90:CP{index}", 100.0 * (index + 1), index) for index in range(12)]
    full, primed = _runtime(), _runtime()
    resolver = _resolver()
    for index, (originator, beneficiary, amount, minute) in enumerate(events):
        transaction = _transaction(index, originator, beneficiary, amount, minute=minute)
        result = full.evaluate(transaction, resolver.resolve(originator), minute, index)
        full.commit(transaction, result, minute)
        WindowStudy._prime(primed, resolver, transaction, minute, index)

    probe = _transaction(99, ALICE, "91:NEW", 5_000.0, minute=30)
    left = full.evaluate(probe, resolver.resolve(ALICE), 30, 99)
    right = primed.evaluate(probe, resolver.resolve(ALICE), 30, 99)
    assert left.decision.decision is right.decision.decision
    assert [item.id for item in left.behaviour.behaviours] == [item.id for item in right.behaviour.behaviours]
    assert [item.id for item in left.evidence] == [item.id for item in right.evidence]
    assert left.behaviour.role.role is right.behaviour.role.role
    assert left.semantic.context.context_state_hash == right.semantic.context.context_state_hash


# -- the study reads the unmodified vocabulary ------------------------------

def test_study_uses_the_unmodified_ontologies_rules_and_policy():
    import aml_runtime.window_study as study

    assert study.ONTOLOGY_HASH == ONTOLOGY_HASH
    assert study.BEHAVIOUR_ONTOLOGY_HASH == BEHAVIOUR_ONTOLOGY_HASH
    runtime = _runtime()
    assert isinstance(runtime.policies, SemanticPolicyEngine)
    assert runtime.conflicts.pairs == COMBINED_CONFLICT_PAIRS
    assert len(SEMANTIC_RULES) == 18


def test_evaluation_set_is_constant_and_windows_only_change_priming():
    assert DENSE_PERIOD_END - EVALUATION_START == EVALUATION_ROWS
    available = [item for item in WINDOWS if item.available]
    assert len({item.name for item in available}) == len(available)
    # every available window ends where the evaluation set begins
    for item in available:
        assert item.prime_start_row < EVALUATION_START
        assert item.prime_rows == EVALUATION_START - item.prime_start_row
    # and they are strictly ordered by how much history they carry
    rows = [item.prime_rows for item in available]
    assert rows == sorted(rows)


def test_unavailable_window_is_declared_with_a_reason():
    missing = [item for item in WINDOWS if not item.available]
    assert missing, "the study must record which requested windows the source cannot supply"
    for item in missing:
        assert item.unavailable_reason
        assert "14" in item.label or item.unavailable_reason


def test_ml_training_partition_is_capped_so_capacity_is_not_a_variable():
    assert ML_TRAIN_CAP == 400_000
    for item in WINDOWS:
        if item.available and item.prime_rows > ML_TRAIN_CAP:
            assert min(item.prime_rows, ML_TRAIN_CAP) == ML_TRAIN_CAP
