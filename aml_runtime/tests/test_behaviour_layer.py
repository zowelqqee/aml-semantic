"""Tests for the Semantic Behaviour Layer.

The properties under test are: causality (behaviour reads only prior events),
composition (the Semantic Context Layer is used, not modified), honesty (claims
below the observation minimum are withheld and counter-evidence is mandatory),
and that behaviour objects reach the decision as evidence rather than as a new
rule layer.
"""

from __future__ import annotations

import pytest

from aml_runtime.models import Decision, Evidence, Transaction
from aml_runtime.behaviour import (
    BEHAVIOUR_ONTOLOGY_HASH,
    BEHAVIOUR_SPECIFICATIONS,
    SEMANTIC_FEATURE_NAMES,
    BehaviourDecisionRuntime,
    BehaviourLayer,
    BehaviourType,
    Horizon,
    RoleType,
    ScenarioType,
    Stage,
    TemporalEngine,
    behaviour_confidence,
    behaviour_replay_pins,
    behaviour_rule_id,
    semantic_feature_vector,
)
from aml_runtime.behaviour.layer import _is_subsequence
from aml_runtime.behaviour.ontology import (
    BEHAVIOUR_MINIMUM_OBSERVATIONS,
    FAN_MINIMUM_COUNTERPARTIES,
    FORWARD_EVENT_MINIMUM,
    HORIZON_MINUTES,
    SCENARIO_PATTERNS,
    UNFILLABLE_HORIZONS_ON_SOURCE,
)
from aml_runtime.behaviour.runtime import BEHAVIOUR_CONFLICT_PAIRS, COMBINED_CONFLICT_PAIRS
from aml_runtime.semantic import SemanticContextLayer, SemanticDecisionRuntime, SemanticType
from aml_runtime.semantic.runtime import SEMANTIC_CONFLICT_PAIRS, SEMANTIC_RULES, SemanticPolicyEngine

from test_semantic_context import ALICE, BOB, CAROL, DMITRI, _resolver, _transaction  # noqa: F401


def _runtime() -> BehaviourDecisionRuntime:
    resolver = _resolver()
    return BehaviourDecisionRuntime(SemanticDecisionRuntime(SemanticContextLayer(resolver)), BehaviourLayer())


def _run(runtime: BehaviourDecisionRuntime, transaction: Transaction, minute: int, index: int):
    resolver = runtime.semantic.context.resolver
    originator = resolver.resolve(transaction.originator_account_id)
    return runtime.evaluate(transaction, originator, minute, index)


def _feed(runtime: BehaviourDecisionRuntime, events: list[tuple[str, str, float, int]], start: int = 0):
    """Commit a list of (originator, beneficiary, amount, minute) events."""
    for offset, (originator, beneficiary, amount, minute) in enumerate(events):
        transaction = _transaction(start + offset, originator, beneficiary, amount, minute=minute)
        result = _run(runtime, transaction, minute, start + offset)
        runtime.commit(transaction, result, minute)


# -- ontology ---------------------------------------------------------------

def test_every_behaviour_type_has_a_specification():
    assert set(BEHAVIOUR_SPECIFICATIONS) == set(BehaviourType)


def test_specifications_declare_a_usable_direction_and_horizon():
    for type_, specification in BEHAVIOUR_SPECIFICATIONS.items():
        assert specification.direction in {"risk", "mitigation", "none"}
        assert specification.horizon in Horizon
        assert 0.0 < specification.prior <= 1.0
        assert specification.half_support >= 1
        assert specification.meaning


def test_confidence_is_prior_times_support_times_coverage():
    value, explanation = behaviour_confidence(BehaviourType.TRANSIT_BEHAVIOUR, observations=10, present_inputs=2, declared_inputs=2)
    specification = BEHAVIOUR_SPECIFICATIONS[BehaviourType.TRANSIT_BEHAVIOUR]
    assert value == pytest.approx(round(specification.prior * (10 / (10 + specification.half_support)), 6))
    assert "prior" in explanation and "support" in explanation and "coverage" in explanation


def test_motif_behaviours_reuse_the_existing_high_concern_topic():
    """So the frozen semantic policy needs no change to treat them as severe."""
    for type_ in (BehaviourType.CIRCULAR_MONEY_MOVEMENT, BehaviourType.RAPID_LAYERING_BEHAVIOUR,
                  BehaviourType.HIGH_VELOCITY_LAYERING):
        assert BEHAVIOUR_SPECIFICATIONS[type_].topic == "network_motif"


def test_unfillable_horizons_are_declared_not_hidden():
    assert set(UNFILLABLE_HORIZONS_ON_SOURCE) == {Horizon.DAYS, Horizon.WEEKS}
    assert HORIZON_MINUTES[Horizon.DAYS] > HORIZON_MINUTES[Horizon.HOURS] > HORIZON_MINUTES[Horizon.MINUTES]


# -- composition: the semantic layer is used, not modified ------------------

def test_semantic_layer_is_composed_not_replaced():
    runtime = _runtime()
    assert isinstance(runtime.semantic, SemanticDecisionRuntime)
    assert isinstance(runtime.policies, SemanticPolicyEngine)
    # the semantic rule set is untouched
    assert len(SEMANTIC_RULES) == 18
    # and the conflict engine is the same class with the union of declared pairs
    assert runtime.conflicts.pairs == COMBINED_CONFLICT_PAIRS
    assert set(SEMANTIC_CONFLICT_PAIRS).issubset(set(COMBINED_CONFLICT_PAIRS))


def test_behaviour_evidence_is_a_projection_of_the_catalog():
    """No behaviour meaning is declared twice: it all comes from the ontology."""
    runtime = _runtime()
    _feed(runtime, [(ALICE, f"90:CP{index}", 100.0, index) for index in range(12)])
    result = _run(runtime, _transaction(50, ALICE, "91:NEW", 100.0, minute=12), 12, 50)
    for item in result.evidence:
        if not item.rule_id.startswith("BEH-"):
            continue
        type_ = BehaviourType(item.rule_id.removeprefix("BEH-"))
        specification = BEHAVIOUR_SPECIFICATIONS[type_]
        assert item.direction == specification.direction
        assert item.topic == specification.topic
        assert item.source_reliability == specification.prior


def test_behaviour_conflicts_can_qualify_semantic_risk():
    """The point of the layer: an event explained by the account's behaviour."""
    pairs = {(item.risk_rule_id, item.mitigating_rule_id) for item in BEHAVIOUR_CONFLICT_PAIRS}
    assert ("SEM-R03-INFORMATIVE-NOVELTY", behaviour_rule_id(BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR)) in pairs
    assert ("SEM-R01-VALUE-REGIME-BREAK", behaviour_rule_id(BehaviourType.EXPECTED_BUSINESS_CYCLE)) in pairs
    assert ("*", behaviour_rule_id(BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR)) in pairs


# -- causality ---------------------------------------------------------------

def test_observation_cannot_see_the_event_it_describes():
    runtime = _runtime()
    transaction = _transaction(0, ALICE, BOB, 100.0)
    runtime.evaluate(transaction, _resolver().resolve(ALICE), 0, 0)
    assert runtime.layer.engine.events_committed == 0
    result = _run(runtime, transaction, 0, 0)
    runtime.commit(transaction, result, 0)
    assert runtime.layer.engine.events_committed == 1


def test_same_prior_state_reproduces_the_same_behaviour_objects():
    events = [(ALICE, f"90:CP{index}", 100.0, index) for index in range(12)]
    first, second = _runtime(), _runtime()
    _feed(first, events)
    _feed(second, events)
    probe = _transaction(99, ALICE, "91:NEW", 400.0, minute=20)
    left = _run(first, probe, 20, 99)
    right = _run(second, probe, 20, 99)
    assert [item.id for item in left.behaviour.behaviours] == [item.id for item in right.behaviour.behaviours]
    assert left.behaviour.role.role is right.behaviour.role.role
    assert left.decision.decision is right.decision.decision


# -- honesty -----------------------------------------------------------------

def test_below_the_observation_minimum_only_ignorance_is_claimed():
    runtime = _runtime()
    _feed(runtime, [(ALICE, BOB, 100.0, index) for index in range(2)])
    result = _run(runtime, _transaction(50, ALICE, CAROL, 100.0, minute=5), 5, 50)
    assert result.behaviour.types() == {BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY}
    assert result.behaviour.role.role is RoleType.UNKNOWN


def test_ignorance_is_not_projected_as_evidence_in_either_direction():
    assert BEHAVIOUR_SPECIFICATIONS[BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY].direction == "none"
    runtime = _runtime()
    result = _run(runtime, _transaction(0, ALICE, BOB, 100.0), 0, 0)
    assert not any(item.rule_id == behaviour_rule_id(BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY)
                   for item in result.evidence)


def test_every_behaviour_object_carries_an_interval_and_supporting_observations():
    runtime = _runtime()
    _feed(runtime, [(ALICE, f"90:CP{index}", 100.0, index) for index in range(12)])
    result = _run(runtime, _transaction(50, ALICE, "91:NEW", 100.0, minute=12), 12, 50)
    assert result.behaviour.behaviours
    for item in result.behaviour.behaviours:
        assert item.supporting_observations
        assert item.causal_explanation
        assert item.interval.end_minute >= item.interval.start_minute
        assert item.confidence_explanation


def test_hypotheses_can_state_what_would_weaken_them():
    """Counter-evidence is what separates a hypothesis from a label."""
    runtime = _runtime()
    # A fan-out account that also has one inbound counterparty, spread over
    # three hour buckets so the shape counts as sustained.
    events = [("80:P0", ALICE, 100.0, 1)]
    events += [(ALICE, f"90:CP{index}", 100.0 * (index + 1), 5 + index) for index in range(FAN_MINIMUM_COUNTERPARTIES)]
    events += [(ALICE, f"92:CP{index}", 100.0 * (index + 3), 70 + index) for index in range(2)]
    events += [(ALICE, f"93:CP{index}", 100.0 * (index + 5), 130 + index) for index in range(2)]
    _feed(runtime, events)
    result = _run(runtime, _transaction(90, ALICE, "94:NEW", 700.0, minute=135), 135, 90)
    distribution = result.behaviour.by_type(BehaviourType.DISTRIBUTION_BEHAVIOUR)
    assert distribution is not None
    assert distribution.counter_evidence, "a distribution claim against real in-degree must record it"


def test_unsupported_catalog_terms_are_withheld_with_their_missing_input():
    runtime = _runtime()
    result = _run(runtime, _transaction(0, ALICE, BOB, 100.0), 0, 0)
    withheld = {item["type"] for item in result.behaviour.withheld}
    assert {"SeasonalBusinessCycle", "MonthlySalaryCadence", "AnnualTaxCycle"}.issubset(withheld)
    for item in result.behaviour.withheld:
        assert item["missing_inputs"]


def test_lifecycle_reports_which_horizons_cannot_be_filled():
    runtime = _runtime()
    result = _run(runtime, _transaction(0, ALICE, BOB, 100.0), 0, 0)
    assert set(result.behaviour.lifecycle.horizons_unfillable) == {"days", "weeks"}


# -- temporal engine ---------------------------------------------------------

def test_horizon_reads_are_a_tumbling_bucket_plus_its_predecessor():
    engine = TemporalEngine()
    for minute in range(0, 10):
        engine.commit(ALICE, BOB, minute, 100.0, "US Dollar", self_posting=False, new_counterparty=minute == 0,
                      established=False, cash_instrument=False, cross_jurisdiction=False,
                      regime_stable=True, regime_broken=False)
    state = engine.state(ALICE)
    state.roll(10)
    assert state.out_in(Horizon.MINUTES) == 10
    state.roll(60)  # two buckets on, so the predecessor is dropped
    assert state.out_in(Horizon.MINUTES) == 0
    assert state.out_count == 10


def test_distinct_degree_is_derived_from_semantic_relationship_objects():
    """The layer keeps no pair index of its own."""
    engine = TemporalEngine()
    for index in range(5):
        engine.commit(ALICE, f"90:CP{index}", index, 100.0, "US Dollar", self_posting=False,
                      new_counterparty=True, established=False, cash_instrument=False,
                      cross_jurisdiction=False, regime_stable=True, regime_broken=False)
    engine.commit(ALICE, "90:CP0", 6, 100.0, "US Dollar", self_posting=False, new_counterparty=False,
                  established=True, cash_instrument=False, cross_jurisdiction=False,
                  regime_stable=True, regime_broken=False)
    state = engine.state(ALICE)
    assert state.distinct_out == 5
    assert state.out_count == 6
    assert state.established_out == 1


def test_prompt_value_preserving_forward_is_counted():
    engine = TemporalEngine()
    engine.commit(BOB, ALICE, 0, 1000.0, "US Dollar", self_posting=False, new_counterparty=True,
                  established=False, cash_instrument=False, cross_jurisdiction=False,
                  regime_stable=True, regime_broken=False)
    engine.commit(ALICE, CAROL, 5, 1010.0, "US Dollar", self_posting=False, new_counterparty=True,
                  established=False, cash_instrument=False, cross_jurisdiction=False,
                  regime_stable=True, regime_broken=False)
    assert engine.state(ALICE).forward_events == 1


def test_liquidity_share_is_measured_over_outbound_events():
    """A self-posting increments both directions; the denominator must not."""
    runtime = _runtime()
    _feed(runtime, [(ALICE, ALICE, 100.0, index) for index in range(6)])
    state = runtime.layer.engine.state(ALICE)
    assert state.self_count == 6 and state.out_count == 6
    assert state.self_count / state.out_count == 1.0
    result = _run(runtime, _transaction(50, ALICE, ALICE, 100.0, minute=7), 7, 50)
    assert BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR in result.behaviour.types()
    assert result.behaviour.role.role is RoleType.TREASURY_ACCOUNT


# -- behaviour inference -----------------------------------------------------

def test_distribution_behaviour_needs_a_sustained_shape():
    runtime = _runtime()
    # A wide fan-out inside a single hour bucket is a burst, not a sustained shape.
    _feed(runtime, [(ALICE, f"90:CP{index}", 100.0 * (index + 1), index) for index in range(FAN_MINIMUM_COUNTERPARTIES + 2)])
    inside = _run(runtime, _transaction(50, ALICE, "91:NEW", 100.0, minute=11), 11, 50)
    assert BehaviourType.FAN_OUT_DISTRIBUTION in inside.behaviour.types()
    assert BehaviourType.DISTRIBUTION_BEHAVIOUR not in inside.behaviour.types()
    # Once the shape has held across further hour buckets it becomes sustained.
    _feed(runtime, [(ALICE, f"92:CP{index}", 100.0 * (index + 3), 70 + index) for index in range(2)], start=60)
    _feed(runtime, [(ALICE, f"93:CP{index}", 100.0 * (index + 5), 130 + index) for index in range(2)], start=70)
    later = _run(runtime, _transaction(80, ALICE, "94:NEW", 700.0, minute=135), 135, 80)
    assert BehaviourType.DISTRIBUTION_BEHAVIOUR in later.behaviour.types()


def test_transit_behaviour_needs_external_flow_in_both_directions():
    runtime = _runtime()
    events = []
    for index in range(5):
        events.append((f"80:P{index}", ALICE, 1000.0, index * 2))
        events.append((ALICE, f"90:CP{index}", 1000.0, index * 2 + 1))
    _feed(runtime, events)
    result = _run(runtime, _transaction(90, ALICE, "91:NEW", 1000.0, minute=20), 20, 90)
    types = result.behaviour.types()
    assert BehaviourType.TRANSIT_BEHAVIOUR in types
    # Collection from several sources with prompt forwarding and no established
    # relationship is the composite reading, which outranks a bare conduit.
    assert BehaviourType.MONEY_MULE_BEHAVIOUR in types
    assert result.behaviour.role.role is RoleType.MONEY_MULE_CANDIDATE


def test_self_postings_are_not_counted_as_transit_flow():
    runtime = _runtime()
    _feed(runtime, [(ALICE, ALICE, 100.0, index) for index in range(8)])
    result = _run(runtime, _transaction(50, ALICE, ALICE, 100.0, minute=9), 9, 50)
    assert BehaviourType.TRANSIT_BEHAVIOUR not in result.behaviour.types()


def test_accumulation_and_transit_are_mutually_exclusive_readings():
    runtime = _runtime()
    # inflow only: value is being gathered, not moved
    events = [(f"80:P{index}", ALICE, 1000.0, index) for index in range(6)]
    events.append((ALICE, "90:CP0", 1.0, 10))
    _feed(runtime, events)
    result = _run(runtime, _transaction(90, ALICE, "91:NEW", 1.0, minute=12), 12, 90)
    types = result.behaviour.types()
    assert BehaviourType.MONEY_ACCUMULATION_BEHAVIOUR in types
    assert BehaviourType.TRANSIT_BEHAVIOUR not in types


# -- roles -------------------------------------------------------------------

def test_role_is_a_state_the_account_moves_through():
    runtime = _runtime()
    # Amounts are dispersed so the run is not read as a homogeneous payment run.
    _feed(runtime, [(ALICE, BOB, 100.0 * (index + 1) ** 2, index) for index in range(BEHAVIOUR_MINIMUM_OBSERVATIONS + 1)])
    ordinary = _run(runtime, _transaction(50, ALICE, BOB, 100.0, minute=10), 10, 50)
    assert ordinary.behaviour.role.role is RoleType.ACTIVE_COUNTERPARTY

    # Feed the fan-out one event at a time and capture the transition itself.
    transitions = []
    for offset in range(FAN_MINIMUM_COUNTERPARTIES + 1):
        transaction = _transaction(60 + offset, ALICE, f"90:CP{offset}", 100.0 * (offset + 1) ** 2, minute=11 + offset)
        result = _run(runtime, transaction, 11 + offset, 60 + offset)
        if result.behaviour.transition:
            transitions.append(result.behaviour.transition)
        runtime.commit(transaction, result, 11 + offset)
    changed = _run(runtime, _transaction(80, ALICE, "91:NEW", 100.0, minute=22), 22, 80)
    assert changed.behaviour.role.role is RoleType.DISTRIBUTOR
    promotion = next(item for item in transitions if item.to_role is RoleType.DISTRIBUTOR)
    assert promotion.from_role is RoleType.ACTIVE_COUNTERPARTY
    assert promotion.caused_by, "a transition must name the behaviours that caused it"
    assert "Distributor" in promotion.explanation


def test_transit_role_outranks_distribution_role():
    """Declared priority: a conduit is the more informative description."""
    from aml_runtime.behaviour.layer import _ROLE_PRIORITY

    order = [role for role, _ in _ROLE_PRIORITY]
    assert order.index(RoleType.TRANSIT_ACCOUNT) < order.index(RoleType.DISTRIBUTOR)
    assert order.index(RoleType.MONEY_MULE_CANDIDATE) < order.index(RoleType.TRANSIT_ACCOUNT)


def test_salary_receiver_is_inferred_from_the_funder_role():
    runtime = _runtime()
    # BOB becomes a distributor
    _feed(runtime, [(BOB, f"90:CP{index}", 100.0 * (index + 1), index) for index in range(FAN_MINIMUM_COUNTERPARTIES + 2)])
    _feed(runtime, [(BOB, ALICE, 100.0, 12)], start=60)
    assert runtime.layer.engine.state(BOB).role in (RoleType.DISTRIBUTOR.value, RoleType.PAYROLL_OPERATOR.value)
    _feed(runtime, [(BOB, ALICE, 100.0, 13)], start=70)
    result = _run(runtime, _transaction(80, ALICE, CAROL, 50.0, minute=14), 14, 80)
    assert result.behaviour.role.role is RoleType.SALARY_RECEIVER


# -- scenarios ---------------------------------------------------------------

def test_scenarios_match_as_an_ordered_subsequence():
    assert _is_subsequence((Stage.RECEIVE, Stage.SPLIT, Stage.FORWARD),
                           (Stage.SETTLE, Stage.RECEIVE, Stage.PAYMENTS, Stage.SPLIT, Stage.PAYMENTS, Stage.FORWARD))
    assert not _is_subsequence((Stage.RECEIVE, Stage.SPLIT, Stage.FORWARD),
                               (Stage.RECEIVE, Stage.FORWARD, Stage.SPLIT))


def test_scenario_confidence_grows_with_pattern_length():
    lengths = {scenario: len(pattern) for scenario, pattern, _ in SCENARIO_PATTERNS}
    assert lengths[ScenarioType.COLLECT_THEN_FORWARD] > lengths[ScenarioType.LAYERING_ATTEMPT]
    assert lengths[ScenarioType.LAYERING_ATTEMPT] > lengths[ScenarioType.TREASURY_CYCLING]


def test_treasury_cycling_is_detected_on_internal_settlement():
    runtime = _runtime()
    _feed(runtime, [(ALICE, ALICE, 100.0, index) for index in range(4)])
    result = _run(runtime, _transaction(50, ALICE, ALICE, 100.0, minute=5), 5, 50)
    assert ScenarioType.TREASURY_CYCLING in {item.type for item in result.behaviour.scenarios}


def test_hold_stage_is_declared_but_never_emitted():
    """It needs a balance model this source does not carry."""
    assert Stage.HOLD in Stage
    for _scenario, pattern, _meaning in SCENARIO_PATTERNS:
        assert Stage.HOLD not in pattern


# -- decisions ---------------------------------------------------------------

def test_the_frozen_policy_still_selects_every_decision():
    runtime = _runtime()
    result = _run(runtime, _transaction(0, ALICE, BOB, 100.0), 0, 0)
    assert {item.policy_id for item in result.policies} >= {"SEM-P01", "SEM-P10", "SEM-P20", "SEM-P00"}
    assert result.decision.decision in set(Decision)


def test_decisions_are_explainable_in_behaviour_and_scenario_objects():
    runtime = _runtime()
    _feed(runtime, [(ALICE, f"90:CP{index}", 100.0, index) for index in range(12)])
    result = _run(runtime, _transaction(50, ALICE, "91:NEW", 100.0, minute=12), 12, 50)
    rationale = result.behaviour_rationale
    assert "Role:" in rationale and "Behaviour:" in rationale and "Scenario:" in rationale
    assert "AML-R0" not in rationale


def test_ml_can_only_lift_an_abstention_through_the_declared_band():
    runtime = _runtime()
    base = _run(runtime, _transaction(0, ALICE, "40:DDD", 100.0), 0, 0)
    assert base.decision.decision is Decision.ABSTAIN
    assert BehaviourDecisionRuntime.routes_to_ml(base)
    low = Evidence("E-low", "ML/T", (), 0.50, "p", base.transaction.timestamp, "ML-T", "risk",
                   "ml_probability", 1.0, 0, {"probability": "0.50"})
    assert runtime.with_ml_evidence(base, low, 0.90).decision.decision is Decision.ABSTAIN
    high = Evidence("E-high", "ML/T", (), 0.99, "p", base.transaction.timestamp, "ML-T", "risk",
                    "ml_probability", 1.0, 0, {"probability": "0.99"})
    lifted = runtime.with_ml_evidence(base, high, 0.90)
    assert lifted.decision.decision is Decision.REVIEW
    assert lifted.decision.policy_ids == ("BEH-ML-02",)


# -- semantic feature space --------------------------------------------------

def test_feature_space_contains_no_transaction_column():
    """Every feature must be a named object from a declared vocabulary."""
    from aml_runtime.ml_benchmark import FEATURE_NAMES as RAW_TRANSACTION_FEATURES

    assert not set(SEMANTIC_FEATURE_NAMES) & set(RAW_TRANSACTION_FEATURES)
    allowed = ("sem_", "beh_", "scn_", "role_", "lifecycle_", "evidence_")
    for name in SEMANTIC_FEATURE_NAMES:
        assert name.startswith(allowed), name


def test_feature_space_covers_every_declared_vocabulary():
    assert sum(1 for name in SEMANTIC_FEATURE_NAMES if name.startswith("beh_")) == len(BehaviourType)
    assert sum(1 for name in SEMANTIC_FEATURE_NAMES if name.startswith("sem_")) == len(SemanticType)
    assert sum(1 for name in SEMANTIC_FEATURE_NAMES if name.startswith("scn_")) == len(ScenarioType)


def test_feature_vector_encodes_behaviour_confidences():
    runtime = _runtime()
    _feed(runtime, [(ALICE, f"90:CP{index}", 100.0, index) for index in range(12)])
    result = _run(runtime, _transaction(50, ALICE, "91:NEW", 100.0, minute=12), 12, 50)
    vector = semantic_feature_vector(
        {item.type: item.confidence for item in result.semantic.context.objects},
        result.behaviour, result.evidence, len(result.conflicts), 1, 0.5,
    )
    assert len(vector) == len(SEMANTIC_FEATURE_NAMES)
    index = SEMANTIC_FEATURE_NAMES.index(f"beh_{BehaviourType.FAN_OUT_DISTRIBUTION.value}")
    assert vector[index] > 0.0


# -- audit and replay --------------------------------------------------------

def test_audit_record_carries_behaviour_role_scenario_and_lifecycle():
    runtime = _runtime()
    _feed(runtime, [(ALICE, f"90:CP{index}", 100.0, index) for index in range(12)])
    record = _run(runtime, _transaction(50, ALICE, "91:NEW", 100.0, minute=12), 12, 50).audit_record()
    behaviour = record["behaviour"]
    assert behaviour["ontology_hash"] == BEHAVIOUR_ONTOLOGY_HASH
    assert behaviour["behaviours"] and behaviour["role"] and behaviour["lifecycle"]
    assert "withheld" in behaviour
    for item in behaviour["behaviours"]:
        assert item["interval"]["horizon"] in {horizon.value for horizon in Horizon}
        assert "counter_evidence" in item


def test_replay_pins_extend_the_semantic_pins():
    runtime = _runtime()
    result = _run(runtime, _transaction(0, ALICE, BOB, 100.0), 0, 0)
    pins = behaviour_replay_pins("stub-snapshot", result)
    assert {"ontology_hash", "context_state_hash", "semantic_object_set_hash"}.issubset(pins)
    assert {"behaviour_ontology_hash", "behaviour_layer_hash", "behaviour_projection_hash",
            "behaviour_object_set_hash", "role_state_hash"}.issubset(pins)
    assert all(pins.values())
