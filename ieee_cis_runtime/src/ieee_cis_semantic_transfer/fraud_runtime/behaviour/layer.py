"""Causal Behaviour Layer for the IEEE-CIS Semantic Runtime.

This is the fraud-domain port of ``aml_runtime.behaviour.layer``.  It consumes
semantic objects plus its own prior-only temporal state; it does not inspect a
label, query future rows, or turn undocumented fields into behaviour claims.
"""

from __future__ import annotations

from ..models import Transaction
from ..semantic.objects import SemanticContextResult
from ..semantic.ontology import SemanticType
from .objects import BehaviourObject, BehaviourReading, LifecycleObject, RoleObject, RoleTransition, ScenarioObject, TimeInterval, behaviour_id
from .ontology import (
    BEHAVIOUR_MINIMUM_OBSERVATIONS, BEHAVIOUR_ONTOLOGY_VERSION, COMPROMISE_DEVICE_MINIMUM_PRIOR,
    DEVICE_ROTATION_MINIMUM, DORMANCY_GAP_MULTIPLE, DORMANCY_MINIMUM_MINUTES, HORIZON_MINUTES,
    SCENARIO_PATTERNS, SCENARIO_PRIOR, SCENARIO_STAGE_MEMORY, STABILITY_OBSERVATION_MINIMUM, TEST_CHARGE_MINIMUM,
    TRUSTED_DEVICE_MINIMUM_PAIR_COUNT, VELOCITY_BURST_MINIMUM_EVENTS, VELOCITY_BURST_RATE_MULTIPLE,
    BehaviourType, Horizon, RoleType, ScenarioType, Stage, UNSUPPORTED_ON_SOURCE, behaviour_confidence,
)
from .temporal import CardState, TemporalEngine

BEHAVIOUR_LAYER_VERSION = f"{BEHAVIOUR_ONTOLOGY_VERSION}+fraud-behaviour-inference/1.0"

_WITHHELD = tuple(
    {"type": name, "missing_inputs": list(inputs),
     "reason": "The IEEE-CIS public release does not supply these inputs; asserting this behaviour would fabricate them."}
    for name, inputs in UNSUPPORTED_ON_SOURCE
)
_STAGE_CODES = {stage: index for index, stage in enumerate(Stage)}
_STAGE_BY_CODE = {index: stage for stage, index in _STAGE_CODES.items()}

_CONSUMED: dict[BehaviourType, frozenset[SemanticType]] = {
    BehaviourType.CARD_TESTING_BEHAVIOUR: frozenset({SemanticType.MINIMAL_TEST_AMOUNT}),
    BehaviourType.DEVICE_ROTATION_BEHAVIOUR: frozenset({SemanticType.FIRST_DEVICE_CONTACT}),
    BehaviourType.VELOCITY_BURST_BEHAVIOUR: frozenset({SemanticType.TEMPO_REGIME_SHIFT}),
    BehaviourType.COMPROMISED_CARD_BEHAVIOUR: frozenset({SemanticType.FIRST_DEVICE_CONTACT, SemanticType.UNEXPECTED_SPENDING_AMOUNT, SemanticType.TEMPO_REGIME_SHIFT, SemanticType.UNEXPECTED_BILLING_REGION}),
    BehaviourType.UNEXPECTED_SPENDING_BEHAVIOUR: frozenset({SemanticType.UNEXPECTED_SPENDING_AMOUNT, SemanticType.TEMPO_REGIME_SHIFT, SemanticType.UNEXPECTED_BILLING_REGION}),
    BehaviourType.DORMANT_CARD_REACTIVATION: frozenset({SemanticType.TEMPO_REGIME}),
    BehaviourType.TRUSTED_DEVICE_BEHAVIOUR: frozenset({SemanticType.ESTABLISHED_DEVICE_RELATIONSHIP}),
    BehaviourType.NORMAL_SPENDING_BEHAVIOUR: frozenset({SemanticType.ROUTINE_SPENDING_AMOUNT, SemanticType.EXPECTED_HIGH_VALUE_SPEND}),
    BehaviourType.EXPECTED_VELOCITY_BEHAVIOUR: frozenset({SemanticType.TEMPO_REGIME}),
    BehaviourType.INSUFFICIENT_CARD_HISTORY: frozenset({SemanticType.NO_ESTABLISHED_CARD_HISTORY}),
}


class SemanticFlags:
    """A deliberate reduction of semantic objects to behaviour inputs."""

    __slots__ = ("types", "test_amount", "new_device", "established_device", "broken", "stable")

    def __init__(self, reading: SemanticContextResult) -> None:
        self.types = reading.types()
        self.test_amount = SemanticType.MINIMAL_TEST_AMOUNT in self.types
        self.new_device = SemanticType.FIRST_DEVICE_CONTACT in self.types
        self.established_device = SemanticType.ESTABLISHED_DEVICE_RELATIONSHIP in self.types
        self.broken = bool(self.types & {SemanticType.UNEXPECTED_SPENDING_AMOUNT, SemanticType.TEMPO_REGIME_SHIFT, SemanticType.UNEXPECTED_BILLING_REGION})
        self.stable = bool(self.types & {SemanticType.ROUTINE_SPENDING_AMOUNT, SemanticType.EXPECTED_HIGH_VALUE_SPEND}) and not self.broken


class BehaviourLayer:
    """Infer behaviours, roles, lifecycle and scenarios over a causal fold."""

    VERSION = BEHAVIOUR_LAYER_VERSION

    def __init__(self, engine: TemporalEngine | None = None) -> None:
        self.engine = engine or TemporalEngine()
        self.behaviours_emitted = self.transitions_emitted = self.scenarios_emitted = 0

    def observe(self, transaction: Transaction, reading: SemanticContextResult, minute: int, event_index: int) -> BehaviourReading:
        card = transaction.card_id
        state = self.engine.peek(card) or CardState()
        flags = SemanticFlags(reading)
        created_at = f"{transaction.timestamp}#{event_index}"
        idle = minute - state.last_minute if state.last_minute >= 0 else 0
        entities = tuple(item.id for item in reading.entities)
        relationship = f"rel:{card}->{transaction.device_id}" if transaction.device_id else ""
        objects: list[BehaviourObject] = []

        def emit(type_: BehaviourType, observations: int, present: int, declared: int, observed: tuple[str, ...], counter: tuple[str, ...], explanation: str, start: int | None = None) -> None:
            confidence, confidence_explanation = behaviour_confidence(type_, observations, present, declared)
            horizon = type_spec(type_).horizon
            objects.append(BehaviourObject(
                id=behaviour_id("BO", type_.value, card, transaction.id, str(observations)), type=type_, subject_id=card,
                confidence=confidence, confidence_explanation=confidence_explanation,
                interval=TimeInterval(start if start is not None else max(state.first_minute, minute - HORIZON_MINUTES[horizon]), minute, horizon),
                supporting_semantic_objects=tuple(item.id for item in reading.objects if item.type in _CONSUMED[type_]),
                supporting_entities=entities, supporting_relationships=((relationship,) if relationship else ()),
                supporting_observations=observed, counter_evidence=counter, causal_explanation=explanation,
                origin=f"BI-{type_.value}", version=BEHAVIOUR_LAYER_VERSION, created_at=created_at,
            ))

        if state.purchase_count < BEHAVIOUR_MINIMUM_OBSERVATIONS:
            emit(BehaviourType.INSUFFICIENT_CARD_HISTORY, 1, 1, 1,
                 (f"{state.purchase_count} prior purchases (< {BEHAVIOUR_MINIMUM_OBSERVATIONS})",), (),
                 "No card-level behavioural hypothesis is supportable before the declared history minimum.",
                 state.first_minute if state.first_minute >= 0 else minute)
        else:
            self._infer(state, flags, transaction, minute, idle, emit)

        role, transition = self._role(state, objects, card, transaction, minute)
        stage = self._stage(flags, state, minute, idle)
        scenarios = self._scenarios(state, stage, objects, card, minute)
        age = max(0, minute - state.first_minute) if state.first_minute >= 0 else 0
        lifecycle = LifecycleObject(
            id=behaviour_id("LC", "Lifecycle", card, transaction.id), subject_id=card, age_minutes=age,
            idle_minutes=idle, observed_events=state.purchase_count, distinct_devices=state.distinct_devices_seen,
            buckets_active=state.hours_active, first_seen_minute=state.first_minute,
            horizons_filled=tuple(h.value for h, width in HORIZON_MINUTES.items() if age >= width), horizons_unfillable=(),
        )
        self.behaviours_emitted += len(objects)
        self.transitions_emitted += int(transition is not None)
        self.scenarios_emitted += len(scenarios)
        return BehaviourReading(transaction.id, card, tuple(sorted(objects, key=lambda item: (item.type.value, item.id))), role, transition,
                                tuple(sorted(scenarios, key=lambda item: item.type.value)), lifecycle, stage, _WITHHELD)

    def _infer(self, state: CardState, flags: SemanticFlags, transaction: Transaction, minute: int, idle: int, emit) -> None:
        current_tests = state.test_charges_in_minutes() + int(flags.test_amount)
        current_new_devices = state.new_devices_in_minutes() + int(flags.new_device)
        current_velocity = state.out_in_minutes() + 1
        rate = state.mean_bucket_rate

        if current_tests >= TEST_CHARGE_MINIMUM:
            emit(BehaviourType.CARD_TESTING_BEHAVIOUR, current_tests, 2, 2,
                 (f"{current_tests} minimal test-sized charges in the trailing 30-minute window",), (),
                 "Repeated minimal charges form a temporally concentrated card-testing pattern.")
        if current_new_devices >= DEVICE_ROTATION_MINIMUM:
            emit(BehaviourType.DEVICE_ROTATION_BEHAVIOUR, current_new_devices, 2, 2,
                 (f"{current_new_devices} first-time devices in the trailing 30-minute window",), (),
                 "The card moved across several previously unseen devices in a short window.")
        if current_velocity >= VELOCITY_BURST_MINIMUM_EVENTS and rate > 0 and current_velocity > VELOCITY_BURST_RATE_MULTIPLE * rate:
            emit(BehaviourType.VELOCITY_BURST_BEHAVIOUR, state.purchase_count, 2, 2,
                 (f"{current_velocity} purchases in trailing 30 minutes", f"own prior rate {rate:.3f} purchases per bucket"), (),
                 "Purchase tempo is materially above the card's own observed rate.")
        if flags.new_device and flags.broken and state.established_device_max_count >= COMPROMISE_DEVICE_MINIMUM_PRIOR:
            emit(BehaviourType.COMPROMISED_CARD_BEHAVIOUR, state.established_device_max_count, 3, 3,
                 (f"established device was observed at least {state.established_device_max_count} times", "new device and regime break co-occur"), (),
                 "A previously established device relationship was displaced when the card's own regime broke.")
        if state.stable_events >= STABILITY_OBSERVATION_MINIMUM and flags.broken:
            emit(BehaviourType.UNEXPECTED_SPENDING_BEHAVIOUR, state.stable_events, 2, 2,
                 (f"{state.stable_events} prior regime-conforming events", "current event broke the amount, tempo, or region regime"), (),
                 "A stable card-level spending regime has just broken.")
        if state.mean_gap_minutes > 0 and idle >= DORMANCY_MINIMUM_MINUTES and idle >= DORMANCY_GAP_MULTIPLE * state.mean_gap_minutes:
            emit(BehaviourType.DORMANT_CARD_REACTIVATION, state.gap_count, 2, 2,
                 (f"idle {idle} minutes", f"own mean inter-purchase gap {state.mean_gap_minutes:.1f} minutes"), (),
                 "Purchase activity resumed after an idle interval large relative to this card's own rhythm.")
        pair_count = state.devices.get(transaction.device_id, 0) if transaction.device_id else 0
        if flags.established_device and pair_count >= TRUSTED_DEVICE_MINIMUM_PAIR_COUNT and state.stable_events >= STABILITY_OBSERVATION_MINIMUM:
            emit(BehaviourType.TRUSTED_DEVICE_BEHAVIOUR, pair_count, 3, 3,
                 (f"this card-device pair has {pair_count} prior purchases", f"{state.stable_events} prior regime-conforming events"), (),
                 "A sustained card-device relationship coincides with a stable spending regime.")
        if state.stable_events >= STABILITY_OBSERVATION_MINIMUM and not flags.broken:
            emit(BehaviourType.NORMAL_SPENDING_BEHAVIOUR, state.stable_events, 2, 2,
                 (f"{state.stable_events} prior amount-and-tempo regime-conforming purchases",), (),
                 "The card has sustained an unbroken observed spending regime.")
        if current_velocity >= VELOCITY_BURST_MINIMUM_EVENTS and rate >= current_velocity / VELOCITY_BURST_RATE_MULTIPLE:
            emit(BehaviourType.EXPECTED_VELOCITY_BEHAVIOUR, state.purchase_count, 2, 2,
                 (f"{current_velocity} purchases in trailing 30 minutes", f"own established rate {rate:.3f} purchases per bucket"), (),
                 "The observed velocity is within the card's own established tempo.")

    def _role(self, state: CardState, objects: list[BehaviourObject], card: str, transaction: Transaction, minute: int) -> tuple[RoleObject, RoleTransition | None]:
        present = {item.type for item in objects}
        priority = (
            (RoleType.COMPROMISED_CARD_CANDIDATE, {BehaviourType.COMPROMISED_CARD_BEHAVIOUR}),
            (RoleType.ROTATING_DEVICE_CARD, {BehaviourType.DEVICE_ROTATION_BEHAVIOUR}),
            (RoleType.HIGH_VELOCITY_CARD, {BehaviourType.VELOCITY_BURST_BEHAVIOUR}),
            (RoleType.TESTED_CARD, {BehaviourType.CARD_TESTING_BEHAVIOUR}),
            (RoleType.TRUSTED_CARD, {BehaviourType.TRUSTED_DEVICE_BEHAVIOUR, BehaviourType.NORMAL_SPENDING_BEHAVIOUR}),
        )
        role = RoleType.ACTIVE_CARD if state.purchase_count >= BEHAVIOUR_MINIMUM_OBSERVATIONS else RoleType.UNKNOWN
        support: tuple[BehaviourObject, ...] = ()
        for candidate, required in priority:
            found = tuple(item for item in objects if item.type in required)
            if found:
                role, support = candidate, found
                break
        if role is RoleType.ACTIVE_CARD and state.mean_gap_minutes > 0 and minute - state.last_minute >= DORMANCY_MINIMUM_MINUTES:
            role = RoleType.DORMANT_CARD
        previous = RoleType(state.role) if state.role else RoleType.UNKNOWN
        transition = None
        if role is not previous:
            transition = RoleTransition(behaviour_id("RT", f"{previous.value}->{role.value}", card, transaction.id), card, previous, role, minute, transaction.id,
                                        tuple(item.id for item in support), f"{previous.value} -> {role.value} from current behavioural evidence.")
        role_object = RoleObject(behaviour_id("RO", role.value, card, transaction.id), card, role,
                                 round(max((item.confidence for item in support), default=(0.5 if role is not RoleType.UNKNOWN else 0.0)), 6),
                                 state.role_since if role is previous and state.role_since >= 0 else minute,
                                 max(0, minute - state.role_since) if role is previous and state.role_since >= 0 else 0,
                                 state.transitions + int(transition is not None), tuple(item.id for item in support), previous,
                                 f"Role {role.value} follows from behavioural evidence or lifecycle state.", BEHAVIOUR_LAYER_VERSION)
        return role_object, transition

    @staticmethod
    def _stage(flags: SemanticFlags, state: CardState, minute: int, idle: int) -> Stage:
        if flags.test_amount:
            return Stage.TEST
        if flags.new_device:
            return Stage.NEW_DEVICE
        if flags.broken and state.out_in_minutes() + 1 >= VELOCITY_BURST_MINIMUM_EVENTS:
            return Stage.BURST
        if SemanticType.UNEXPECTED_BILLING_REGION in flags.types:
            return Stage.REGION_CHANGE
        if state.mean_gap_minutes > 0 and idle >= DORMANCY_MINIMUM_MINUTES and idle >= DORMANCY_GAP_MULTIPLE * state.mean_gap_minutes:
            return Stage.DORMANT
        return Stage.PURCHASE

    def _scenarios(self, state: CardState, stage: Stage, objects: list[BehaviourObject], card: str, minute: int) -> list[ScenarioObject]:
        history = list(state.stages or ())
        observed = tuple(_STAGE_BY_CODE[code] for code, _ in history[-(SCENARIO_STAGE_MEMORY - 1):]) + (stage,)
        start = history[-(SCENARIO_STAGE_MEMORY - 1)][1] if len(history) >= SCENARIO_STAGE_MEMORY - 1 else (history[0][1] if history else minute)
        results: list[ScenarioObject] = []
        for type_, pattern, meaning in SCENARIO_PATTERNS:
            if not _is_subsequence(pattern, observed):
                continue
            mitigation = tuple(f"{item.type.value} argues for an ordinary reading" for item in objects if item.direction == "mitigation")
            results.append(ScenarioObject(behaviour_id("SC", type_.value, card, str(minute)), type_, card,
                                          round(SCENARIO_PRIOR * len(pattern) / (len(pattern) + 1), 6), pattern, observed,
                                          TimeInterval(start, minute, Horizon.HOURS), tuple(item.id for item in objects), mitigation,
                                          meaning, BEHAVIOUR_LAYER_VERSION))
        return results

    def commit(self, transaction: Transaction, reading: SemanticContextResult, behaviour: BehaviourReading, minute: int) -> None:
        flags = SemanticFlags(reading)
        self.engine.commit(transaction.card_id, transaction.device_id, minute, transaction.amount, transaction.distance1,
                           new_device=flags.new_device, is_test_amount=flags.test_amount,
                           regime_stable=flags.stable, regime_broken=flags.broken)
        state = self.engine.state(transaction.card_id)
        if behaviour.role.role.value != state.role:
            state.role_prev, state.role, state.role_since = state.role, behaviour.role.role.value, minute
            state.transitions += 1
        history = list(state.stages or ())
        history.append((_STAGE_CODES[behaviour.stage], minute))
        state.stages = tuple(history[-SCENARIO_STAGE_MEMORY:])


def _is_subsequence(pattern: tuple[Stage, ...], observed: tuple[Stage, ...]) -> bool:
    position = 0
    for stage in observed:
        if position < len(pattern) and stage is pattern[position]:
            position += 1
    return position == len(pattern)


def type_spec(type_: BehaviourType):
    from .ontology import BEHAVIOUR_SPECIFICATIONS
    return BEHAVIOUR_SPECIFICATIONS[type_]
