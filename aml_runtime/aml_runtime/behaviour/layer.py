"""The Semantic Behaviour Layer.

Input: the stream of ``SemanticContextResult`` objects produced by the Semantic
Context Layer, unchanged.  Output: behaviour objects, a dynamic role, role
transitions, scenario objects and a lifecycle object.

Two disciplines are enforced structurally:

*Causality.* ``observe`` is pure with respect to state; ``commit`` folds the
event in afterwards.  A behaviour claim about event *i* reads only events
``0..i-1``.

*Honesty.* An account below the declared observation minimum yields
``InsufficientBehaviouralHistory`` and nothing else, and horizons the source
cannot fill are reported as unfillable rather than treated as empty-and-clean.
"""

from __future__ import annotations

from ..models import Transaction
from ..semantic.entities import ResolvedAccount
from ..semantic.objects import SemanticContextResult
from ..semantic.ontology import EntityForm, SemanticType
from .objects import (
    BehaviourObject,
    BehaviourReading,
    LifecycleObject,
    RoleObject,
    RoleTransition,
    ScenarioObject,
    TimeInterval,
    behaviour_id,
)
from .ontology import (
    ACCUMULATION_RETENTION,
    BEHAVIOUR_MINIMUM_OBSERVATIONS,
    BEHAVIOUR_ONTOLOGY_VERSION,
    BURST_MINIMUM_EVENTS,
    BURST_RATE_MULTIPLE,
    CASH_CONCENTRATION_SHARE,
    DORMANCY_GAP_MULTIPLE,
    DORMANCY_MINIMUM_MINUTES,
    FAN_ASYMMETRY_RATIO,
    FAN_MINIMUM_COUNTERPARTIES,
    FLOW_MINIMUM_EVENTS,
    FORWARD_EVENT_MINIMUM,
    HORIZON_MINUTES,
    HUB_MAXIMUM_ASYMMETRY,
    HUB_MINIMUM_DEGREE,
    LIQUIDITY_SELF_POSTING_SHARE,
    PAYROLL_AMOUNT_DISPERSION,
    PAYROLL_MINIMUM_PAYMENTS,
    PROMPT_FORWARD_MINUTES,
    RELATIONSHIP_GROWTH_RATIO,
    SCENARIO_PATTERNS,
    SCENARIO_PRIOR,
    SCENARIO_STAGE_MEMORY,
    STABILITY_BUCKET_MINIMUM,
    SUPPLIER_MINIMUM_ESTABLISHED,
    SUSTAINED_BUCKET_MINIMUM,
    TRANSIT_RETENTION_TOLERANCE,
    UNFILLABLE_HORIZONS_ON_SOURCE,
    UNSUPPORTED_ON_SOURCE,
    BehaviourType,
    Horizon,
    RoleType,
    ScenarioType,
    Stage,
    behaviour_confidence,
)
from .temporal import AccountBehaviourState, TemporalEngine

BEHAVIOUR_LAYER_VERSION = f"{BEHAVIOUR_ONTOLOGY_VERSION}+aml-behaviour-inference/1.0"

_INCORPORATED = frozenset({EntityForm.CORPORATION, EntityForm.PARTNERSHIP, EntityForm.SOLE_PROPRIETORSHIP})

_STAGE_CODES = {stage: index for index, stage in enumerate(Stage)}
_STAGE_BY_CODE = {index: stage for stage, index in _STAGE_CODES.items()}

_WITHHELD = tuple(
    {"type": name, "missing_inputs": list(inputs),
     "reason": "The observed window cannot supply this input; asserting the behaviour would fabricate a history."}
    for name, inputs in UNSUPPORTED_ON_SOURCE
)

#: Semantic object types each behaviour family aggregates.  Recorded on the
#: object so an auditor can walk behaviour -> semantic object -> raw observation.
_CONSUMED: dict[BehaviourType, frozenset[SemanticType]] = {
    BehaviourType.DISTRIBUTION_BEHAVIOUR: frozenset({SemanticType.FIRST_CONTACT, SemanticType.NON_INFORMATIVE_NOVELTY, SemanticType.DISTRIBUTION_NODE}),
    BehaviourType.FAN_OUT_DISTRIBUTION: frozenset({SemanticType.FIRST_CONTACT, SemanticType.NON_INFORMATIVE_NOVELTY}),
    BehaviourType.COLLECTION_BEHAVIOUR: frozenset({SemanticType.COLLECTION_NODE, SemanticType.NON_INFORMATIVE_NOVELTY}),
    BehaviourType.FAN_IN_COLLECTION: frozenset({SemanticType.COLLECTION_NODE}),
    BehaviourType.SETTLEMENT_HUB_BEHAVIOUR: frozenset({SemanticType.DISTRIBUTION_NODE, SemanticType.COLLECTION_NODE, SemanticType.ESTABLISHED_RELATIONSHIP}),
    BehaviourType.MONEY_ACCUMULATION_BEHAVIOUR: frozenset({SemanticType.VALUE_REGIME}),
    BehaviourType.TRANSIT_BEHAVIOUR: frozenset({SemanticType.PASS_THROUGH_ACCOUNT, SemanticType.VALUE_REGIME}),
    BehaviourType.PASS_THROUGH_BEHAVIOUR: frozenset({SemanticType.PASS_THROUGH_ACCOUNT, SemanticType.LAYERING_CHAIN_SEGMENT}),
    BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR: frozenset({SemanticType.INTERNAL_BOOK_ENTRY, SemanticType.INTRA_CUSTOMER_TRANSFER, SemanticType.BOOKKEEPING_ACCOUNT}),
    BehaviourType.CASH_CONCENTRATION_BEHAVIOUR: frozenset({SemanticType.CASH_INSTRUMENT_SETTLEMENT, SemanticType.COLLECTION_NODE}),
    BehaviourType.BURST_ACTIVITY_BEHAVIOUR: frozenset({SemanticType.TEMPO_REGIME, SemanticType.BEHAVIOUR_REGIME_SHIFT}),
    BehaviourType.HIGH_VELOCITY_LAYERING: frozenset({SemanticType.LAYERING_CHAIN_SEGMENT, SemanticType.PASS_THROUGH_ACCOUNT}),
    BehaviourType.RAPID_LAYERING_BEHAVIOUR: frozenset({SemanticType.LAYERING_CHAIN_SEGMENT, SemanticType.PASS_THROUGH_ACCOUNT}),
    BehaviourType.DORMANT_ACCOUNT_ACTIVATION: frozenset({SemanticType.TEMPO_REGIME}),
    BehaviourType.CIRCULAR_MONEY_MOVEMENT: frozenset({SemanticType.ESTABLISHED_RELATIONSHIP, SemanticType.RECENTLY_CREATED_RELATIONSHIP, SemanticType.FIRST_CONTACT}),
    BehaviourType.RELATIONSHIP_GROWTH_BEHAVIOUR: frozenset({SemanticType.FIRST_CONTACT, SemanticType.COUNTERPARTY_REGIME}),
    BehaviourType.RELATIONSHIP_COLLAPSE_BEHAVIOUR: frozenset({SemanticType.ESTABLISHED_RELATIONSHIP, SemanticType.COUNTERPARTY_REGIME}),
    BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR: frozenset({SemanticType.ROUTINE_VALUE_TRANSFER, SemanticType.VALUE_REGIME, SemanticType.NORMAL_OPERATIONAL_BURST}),
    BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR: frozenset({SemanticType.DISTRIBUTION_NODE, SemanticType.NORMAL_OPERATIONAL_BURST, SemanticType.VALUE_REGIME}),
    BehaviourType.SUPPLIER_SETTLEMENT_BEHAVIOUR: frozenset({SemanticType.ESTABLISHED_RELATIONSHIP, SemanticType.ROUTINE_VALUE_TRANSFER}),
    BehaviourType.EXPECTED_BUSINESS_CYCLE: frozenset({SemanticType.ROUTINE_VALUE_TRANSFER, SemanticType.EXPECTED_HIGH_VALUE_TRANSFER, SemanticType.VALUE_REGIME, SemanticType.TEMPO_REGIME}),
    BehaviourType.UNEXPECTED_BUSINESS_CYCLE: frozenset({SemanticType.UNEXPECTED_LARGE_TRANSFER, SemanticType.BEHAVIOUR_REGIME_SHIFT}),
    BehaviourType.MONEY_MULE_BEHAVIOUR: frozenset({SemanticType.NON_INFORMATIVE_NOVELTY, SemanticType.NO_ESTABLISHED_BASELINE, SemanticType.PASS_THROUGH_ACCOUNT}),
    BehaviourType.SHELL_COMPANY_BEHAVIOUR: frozenset({SemanticType.CROSS_JURISDICTION_TRANSFER, SemanticType.PASS_THROUGH_ACCOUNT, SemanticType.NO_ESTABLISHED_BASELINE}),
    BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY: frozenset({SemanticType.NO_ESTABLISHED_BASELINE}),
}


class SemanticFlags:
    """The semantic reading, reduced to the flags behaviour aggregates."""

    __slots__ = ("self_posting", "intra_customer", "new_counterparty", "established",
                 "cash", "cross_jurisdiction", "stable", "broken", "layering", "types")

    def __init__(self, reading: SemanticContextResult) -> None:
        types = reading.types()
        self.types = types
        self.self_posting = SemanticType.INTERNAL_BOOK_ENTRY in types
        self.intra_customer = SemanticType.INTRA_CUSTOMER_TRANSFER in types
        self.new_counterparty = bool(types & {SemanticType.FIRST_CONTACT, SemanticType.NON_INFORMATIVE_NOVELTY})
        self.established = SemanticType.ESTABLISHED_RELATIONSHIP in types
        self.cash = SemanticType.CASH_INSTRUMENT_SETTLEMENT in types
        self.cross_jurisdiction = SemanticType.CROSS_JURISDICTION_TRANSFER in types
        self.broken = bool(types & {SemanticType.UNEXPECTED_LARGE_TRANSFER, SemanticType.BEHAVIOUR_REGIME_SHIFT})
        self.stable = bool(types & {SemanticType.ROUTINE_VALUE_TRANSFER, SemanticType.EXPECTED_HIGH_VALUE_TRANSFER}) and not self.broken
        self.layering = SemanticType.LAYERING_CHAIN_SEGMENT in types


class BehaviourLayer:
    """Behaviour inference over the temporal engine."""

    VERSION = BEHAVIOUR_LAYER_VERSION

    def __init__(self, engine: TemporalEngine | None = None) -> None:
        self.engine = engine or TemporalEngine()
        self.behaviours_emitted = 0
        self.transitions_emitted = 0
        self.scenarios_emitted = 0

    # -- inference -------------------------------------------------------
    def observe(
        self,
        transaction: Transaction,
        reading: SemanticContextResult,
        originator: ResolvedAccount,
        minute: int,
        event_index: int,
    ) -> BehaviourReading:
        account = transaction.originator_account_id
        state = self.engine.peek(account) or AccountBehaviourState()
        flags = SemanticFlags(reading)
        created_at = f"{transaction.timestamp}#{event_index}"
        entities = tuple(item.id for item in originator.entities())
        relationship = f"rel:{account}->{transaction.beneficiary_account_id}"
        idle = minute - state.last_minute if state.last_minute >= 0 else 0

        def semantic_ids(type_: BehaviourType) -> tuple[str, ...]:
            wanted = _CONSUMED[type_]
            return tuple(item.id for item in reading.objects if item.type in wanted)

        objects: list[BehaviourObject] = []

        def emit(
            type_: BehaviourType,
            observations: int,
            present: int,
            declared: int,
            observed: tuple[str, ...],
            counter: tuple[str, ...],
            explanation: str,
            start_minute: int | None = None,
        ) -> None:
            confidence, confidence_explanation = behaviour_confidence(type_, observations, present, declared)
            specification_horizon = _horizon_of(type_)
            width = HORIZON_MINUTES[specification_horizon]
            objects.append(BehaviourObject(
                id=behaviour_id("BO", type_.value, account, transaction.id, str(observations)),
                type=type_, subject_id=account, confidence=confidence,
                confidence_explanation=confidence_explanation,
                interval=TimeInterval(start_minute if start_minute is not None else max(state.first_minute, minute - width), minute, specification_horizon),
                supporting_semantic_objects=semantic_ids(type_),
                supporting_entities=entities,
                supporting_relationships=(relationship,),
                supporting_observations=observed,
                counter_evidence=counter,
                causal_explanation=explanation,
                origin=f"BI-{type_.value}",
                version=BEHAVIOUR_LAYER_VERSION,
                created_at=created_at,
            ))

        if state.events < BEHAVIOUR_MINIMUM_OBSERVATIONS:
            emit(BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY, 1, 1, 1,
                 (f"{state.events} prior observations (< {BEHAVIOUR_MINIMUM_OBSERVATIONS})",),
                 (f"{state.out_count} outbound and {state.in_count} inbound events exist but do not reach the minimum",),
                 "No behavioural claim is supportable for this account yet.",
                 state.first_minute if state.first_minute >= 0 else minute)
        else:
            self._infer(state, flags, originator, minute, idle, transaction, emit)

        role_object, transition = self._role(state, objects, minute, idle, transaction, created_at)
        stage = self._stage(state, flags, minute)
        scenarios = self._scenarios(state, stage, objects, account, minute)
        lifecycle = LifecycleObject(
            id=behaviour_id("LC", "Lifecycle", account, transaction.id),
            subject_id=account,
            age_minutes=max(0, minute - state.first_minute) if state.first_minute >= 0 else 0,
            idle_minutes=idle,
            observed_events=state.events,
            distinct_counterparties=state.distinct_out + state.distinct_in,
            buckets_active=state.hours_active,
            first_seen_minute=state.first_minute,
            horizons_filled=state.horizons_filled(),
            horizons_unfillable=tuple(item.value for item in UNFILLABLE_HORIZONS_ON_SOURCE),
        )
        self.behaviours_emitted += len(objects)
        self.scenarios_emitted += len(scenarios)
        self.transitions_emitted += 1 if transition else 0
        return BehaviourReading(
            transaction_id=transaction.id, subject_id=account,
            behaviours=tuple(sorted(objects, key=lambda item: (item.type.value, item.id))),
            role=role_object, transition=transition,
            scenarios=tuple(sorted(scenarios, key=lambda item: item.type.value)),
            lifecycle=lifecycle, stage=stage, withheld=_WITHHELD,
        )

    def _infer(self, state: AccountBehaviourState, flags: SemanticFlags, originator: ResolvedAccount, minute: int, idle: int, transaction: Transaction, emit) -> None:
        """Every declared behavioural predicate, in catalog order."""
        out_degree, in_degree = state.distinct_out, state.distinct_in
        sustained = state.hours_active >= SUSTAINED_BUCKET_MINIMUM
        burst_out = state.new_out_in(Horizon.MINUTES)
        burst_in = state.new_in_in(Horizon.MINUTES)
        forwards_minutes = state.forwards_in(Horizon.MINUTES)
        transit_gap = abs(state.in_value - state.out_value) / state.in_value if state.in_value > 0 else 1.0

        # -- shape ---------------------------------------------------------
        if out_degree >= FAN_MINIMUM_COUNTERPARTIES and out_degree >= FAN_ASYMMETRY_RATIO * max(1, in_degree) and sustained:
            emit(BehaviourType.DISTRIBUTION_BEHAVIOUR, out_degree, 3, 3,
                 (f"{out_degree} distinct beneficiaries against {in_degree} distinct payers",
                  f"shape held across {state.hours_active} hour buckets"),
                 ((f"{in_degree} inbound counterparties are not negligible",) if in_degree else
                  ("no inbound counterparty at all, so the shape may be an artefact of a short window",)),
                 "One-to-many payment shape persisting across more than one time bucket.")
        if burst_out >= FAN_MINIMUM_COUNTERPARTIES:
            emit(BehaviourType.FAN_OUT_DISTRIBUTION, burst_out, 2, 2,
                 (f"{burst_out} new beneficiaries inside the trailing {HORIZON_MINUTES[Horizon.MINUTES]}-minute window",),
                 ((f"outbound amounts are homogeneous (dispersion {state.outbound_dispersion:.3f}), which also fits a payment run",)
                  if state.outbound_dispersion <= PAYROLL_AMOUNT_DISPERSION else ()),
                 "A wide fan-out concentrated inside one short bucket.")
        if in_degree >= FAN_MINIMUM_COUNTERPARTIES and in_degree >= FAN_ASYMMETRY_RATIO * max(1, out_degree) and sustained:
            emit(BehaviourType.COLLECTION_BEHAVIOUR, in_degree, 3, 3,
                 (f"{in_degree} distinct payers against {out_degree} distinct beneficiaries",
                  f"shape held across {state.hours_active} hour buckets"),
                 ((f"{out_degree} outbound counterparties exist, so value is not only gathering",) if out_degree else ()),
                 "Many-to-one receipt shape persisting across more than one time bucket.")
        if burst_in >= FAN_MINIMUM_COUNTERPARTIES:
            emit(BehaviourType.FAN_IN_COLLECTION, burst_in, 2, 2,
                 (f"{burst_in} new payers inside the trailing {HORIZON_MINUTES[Horizon.MINUTES]}-minute window",),
                 (), "Value arriving from many distinct sources at once.")
        if out_degree >= HUB_MINIMUM_DEGREE and in_degree >= HUB_MINIMUM_DEGREE and max(out_degree, in_degree) <= HUB_MAXIMUM_ASYMMETRY * max(1, min(out_degree, in_degree)):
            emit(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR, min(out_degree, in_degree), 3, 3,
                 (f"{out_degree} outbound and {in_degree} inbound counterparties, asymmetry within {HUB_MAXIMUM_ASYMMETRY:.0f}x",),
                 ((f"{state.forward_events} prompt forwards also fit a transit reading",) if state.forward_events >= FORWARD_EVENT_MINIMUM else ()),
                 "Symmetric two-way degree at scale: infrastructure rather than a directional flow.")

        # -- flow ----------------------------------------------------------
        if state.in_count >= FLOW_MINIMUM_EVENTS and state.retention >= ACCUMULATION_RETENTION:
            emit(BehaviourType.MONEY_ACCUMULATION_BEHAVIOUR, state.in_count, 2, 2,
                 (f"retained {state.retention:.2f} of {state.in_value:.2f} received over {state.in_count} inflows",),
                 ((f"{state.out_count} outflows exist, so accumulation is partial",) if state.out_count else ()),
                 "Inflow materially exceeds outflow: value is being gathered rather than moved.")
        # Transit is a claim about value *arriving and leaving*.  A self-posting
        # does neither, and it increments both counters, so an account that only
        # posts to itself would otherwise satisfy the predicate trivially.  The
        # flow claim is therefore made over external events only.
        external_in = state.in_count - state.self_count
        external_out = state.out_count - state.self_count
        if external_in >= FLOW_MINIMUM_EVENTS and external_out >= FLOW_MINIMUM_EVENTS and transit_gap <= TRANSIT_RETENTION_TOLERANCE:
            emit(BehaviourType.TRANSIT_BEHAVIOUR, min(external_in, external_out), 2, 2,
                 (f"inflow {state.in_value:.2f} against outflow {state.out_value:.2f} (gap {transit_gap:.3f})",),
                 ((f"{state.self_count} self-postings inflate both sides",) if state.self_count else ()),
                 "Value arrives and leaves at comparable scale: the account is a conduit.")
        if forwards_minutes >= FORWARD_EVENT_MINIMUM:
            mean_delay = state.forward_delay_sum / state.forward_events if state.forward_events else 0.0
            emit(BehaviourType.PASS_THROUGH_BEHAVIOUR, forwards_minutes, 2, 2,
                 (f"{forwards_minutes} value-preserving forwards inside the trailing short window",
                  f"mean forward delay {mean_delay:.1f} minutes"),
                 ((f"{state.self_count} of the account's events are self-postings",) if state.self_count else ()),
                 "Received value is being forwarded onward within minutes, repeatedly.")
        # The share is taken over *outbound* events: a self-posting increments
        # both the outbound and the inbound counter, so measuring it against
        # total events would cap the ratio at 0.5 and make the claim
        # unsatisfiable for an account that does nothing else.
        if state.self_count >= FLOW_MINIMUM_EVENTS and state.self_count / max(1, state.out_count) >= LIQUIDITY_SELF_POSTING_SHARE:
            emit(BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR, state.self_count, 2, 2,
                 (f"{state.self_count} of {state.out_count} outbound events are internal postings",),
                 ((f"{state.distinct_out} external beneficiaries exist",) if state.distinct_out else ()),
                 "Two-way flow dominated by internal movement: a treasury or accounting location.")
        if in_degree >= FAN_MINIMUM_COUNTERPARTIES and state.cash_inflow_share >= CASH_CONCENTRATION_SHARE:
            emit(BehaviourType.CASH_CONCENTRATION_BEHAVIOUR, state.cash_in_count, 2, 2,
                 (f"{state.cash_in_count} of {state.in_count} inflows settled by cash instrument ({state.cash_inflow_share:.2f})",
                  f"{in_degree} distinct payers"),
                 (), "Fan-in whose inflows are materially settled by instruments carrying no counterparty trail.")

        # -- tempo -----------------------------------------------------------
        current_burst = state.out_in(Horizon.MINUTES) + 1
        rate = state.mean_bucket_rate
        if current_burst >= BURST_MINIMUM_EVENTS and rate > 0 and current_burst > BURST_RATE_MULTIPLE * rate:
            emit(BehaviourType.BURST_ACTIVITY_BEHAVIOUR, state.out_count, 2, 2,
                 (f"{current_burst} outbound events in the trailing short window",
                  f"own mean {rate:.3f} events per bucket"),
                 ((f"{state.distinct_out} counterparties make a payment run plausible",) if state.distinct_out >= FAN_MINIMUM_COUNTERPARTIES else ()),
                 "Outbound tempo far above the account's own established rate.")
        if forwards_minutes >= FORWARD_EVENT_MINIMUM and burst_out >= 2:
            emit(BehaviourType.HIGH_VELOCITY_LAYERING, forwards_minutes, 2, 2,
                 (f"{forwards_minutes} preserving forwards and {burst_out} new beneficiaries in the same short window",),
                 (), "Forwarding and dispersal are happening together inside one short bucket.")
        if state.forward_events >= 2 * FORWARD_EVENT_MINIMUM and state.forward_delay_sum / state.forward_events <= PROMPT_FORWARD_MINUTES / 2:
            emit(BehaviourType.RAPID_LAYERING_BEHAVIOUR, state.forward_events, 2, 2,
                 (f"{state.forward_events} preserving forwards, mean delay {state.forward_delay_sum / state.forward_events:.1f} minutes",),
                 ((f"the account also holds {state.retention:.2f} of what it receives",) if state.retention > TRANSIT_RETENTION_TOLERANCE else ()),
                 "Sustained forward-with-preservation across the hour horizon.")
        if state.mean_gap_minutes > 0 and idle >= DORMANCY_GAP_MULTIPLE * state.mean_gap_minutes and idle >= DORMANCY_MINIMUM_MINUTES:
            emit(BehaviourType.DORMANT_ACCOUNT_ACTIVATION, state.gap_count, 2, 2,
                 (f"idle {idle} minutes against own mean gap {state.mean_gap_minutes:.1f} minutes",),
                 (), "Activity resumed after an idle period long relative to this account's own tempo.")

        # -- motif -----------------------------------------------------------
        beneficiary = transaction.beneficiary_account_id
        if beneficiary and state.in_minute >= 0 and minute - state.in_minute <= HORIZON_MINUTES[Horizon.HOURS]:
            if beneficiary == state.in_src:
                emit(BehaviourType.CIRCULAR_MONEY_MOVEMENT, max(1, state.forward_events), 2, 2,
                     (f"value is returning to {beneficiary}, which funded this account {minute - state.in_minute} minutes ago",),
                     ((f"{beneficiary} is an established counterparty, so reciprocity may be ordinary",) if flags.established else ()),
                     "Value returned to the party that recently funded this account (one hop).")
            elif beneficiary == state.in_origin and state.in_origin:
                emit(BehaviourType.CIRCULAR_MONEY_MOVEMENT, max(1, state.forward_events), 2, 2,
                     (f"value is returning to {beneficiary}, which funded this account's payer",),
                     ((f"{beneficiary} is an established counterparty, so reciprocity may be ordinary",) if flags.established else ()),
                     "Value returned to the origin two hops upstream.")

        # -- relationship evolution -------------------------------------------
        growth = state.new_out_in(Horizon.HOURS)
        if out_degree >= FAN_MINIMUM_COUNTERPARTIES and growth >= RELATIONSHIP_GROWTH_RATIO * out_degree:
            emit(BehaviourType.RELATIONSHIP_GROWTH_BEHAVIOUR, growth, 2, 2,
                 (f"{growth} of {out_degree} counterparties were first used inside the trailing hour",),
                 ((f"{state.established_out} payments went to established counterparties",) if state.established_out else ()),
                 "The counterparty set is expanding faster than the account's own history supports.")
        if state.peak_distinct_out >= FAN_MINIMUM_COUNTERPARTIES and state.h_out >= FLOW_MINIMUM_EVENTS and state.h_new_out == 0:
            emit(BehaviourType.RELATIONSHIP_COLLAPSE_BEHAVIOUR, state.peak_distinct_out, 2, 2,
                 (f"peak counterparty set {state.peak_distinct_out} but no new counterparty in {state.h_out} payments this hour",),
                 (), "A previously broad counterparty set has contracted.")

        # -- regime (the mitigating pole) --------------------------------------
        homogeneous = state.h_out >= PAYROLL_MINIMUM_PAYMENTS and state.outbound_dispersion <= PAYROLL_AMOUNT_DISPERSION
        if homogeneous:
            emit(BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR, state.h_out, 2, 2,
                 (f"{state.h_out} payments this hour with amount dispersion {state.outbound_dispersion:.3f}",),
                 ((f"{state.new_out_in(Horizon.HOURS)} of them went to first-time counterparties",) if state.new_out_in(Horizon.HOURS) else ()),
                 "Repeated fan-out of homogeneous amounts: a payment run, not a dispersal.")
            if sustained and out_degree >= HUB_MINIMUM_DEGREE:
                emit(BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR, out_degree, 3, 3,
                     (f"payment-run shape across {state.hours_active} hour buckets to {out_degree} counterparties",),
                     (), "Sustained payment-run shape: an operator, not a one-off event.")
        if state.established_out >= SUPPLIER_MINIMUM_ESTABLISHED and state.out_count >= BEHAVIOUR_MINIMUM_OBSERVATIONS and state.established_out / state.out_count >= 0.50:
            emit(BehaviourType.SUPPLIER_SETTLEMENT_BEHAVIOUR, state.established_out, 2, 2,
                 (f"{state.established_out} of {state.out_count} payments went to established counterparties",),
                 ((f"{state.distinct_out - state.established_out} counterparties remain non-established",) if state.distinct_out > state.established_out else ()),
                 "Repeated payment into an established, stable counterparty set.")
        if state.stable_events >= STABILITY_BUCKET_MINIMUM * 2 and state.break_events == 0:
            emit(BehaviourType.EXPECTED_BUSINESS_CYCLE, state.stable_events, 4, 4,
                 (f"{state.stable_events} consecutive observations inside the party's own value and tempo regimes",),
                 (), "Value and tempo have both stayed inside the party's own regimes.")
        if state.stable_events >= STABILITY_BUCKET_MINIMUM * 2 and flags.broken:
            emit(BehaviourType.UNEXPECTED_BUSINESS_CYCLE, state.stable_events, 2, 2,
                 (f"a regime that held for {state.stable_events} observations has just broken",),
                 (), "A previously stable value or tempo regime has broken.")

        # -- composite hypotheses -------------------------------------------
        if in_degree >= 3 and state.forward_events >= FORWARD_EVENT_MINIMUM and state.retention <= TRANSIT_RETENTION_TOLERANCE and state.established_out == 0:
            emit(BehaviourType.MONEY_MULE_BEHAVIOUR, state.forward_events, 4, 4,
                 (f"{in_degree} distinct payers, {state.forward_events} prompt forwards, retention {state.retention:.2f}",
                  "no established counterparty relationship exists"),
                 ((f"the account has been observed for {minute - state.first_minute} minutes, which is short",) if minute - state.first_minute < HORIZON_MINUTES[Horizon.HOURS] else ()),
                 "Collection from several sources, prompt forwarding onward, and no established baseline.")
        if (
            originator.form in _INCORPORATED
            and state.out_count >= FLOW_MINIMUM_EVENTS
            and transit_gap <= TRANSIT_RETENTION_TOLERANCE
            and state.established_out == 0
            and state.cross_juris_out / max(1, state.out_count) >= 0.50
        ):
            emit(BehaviourType.SHELL_COMPANY_BEHAVIOUR, state.out_count, 4, 4,
                 (f"{originator.form.value} whose outflow matches inflow (gap {transit_gap:.3f})",
                  f"{state.cross_juris_out} of {state.out_count} payments cross a jurisdiction boundary",
                  "no established counterparty relationship exists"),
                 (), "An incorporated party whose account only transits value across jurisdictions with no operating regime.")

    # -- roles -----------------------------------------------------------
    def _role(self, state: AccountBehaviourState, objects: list[BehaviourObject], minute: int, idle: int, transaction: Transaction, created_at: str) -> tuple[RoleObject, RoleTransition | None]:
        present = {item.type for item in objects}
        role = RoleType.UNKNOWN
        support: tuple[BehaviourObject, ...] = ()
        for candidate, required in _ROLE_PRIORITY:
            matched = present & required
            if matched:
                role = candidate
                support = tuple(item for item in objects if item.type in matched)
                break
        else:
            funder = self.engine.peek(state.in_src) if state.in_src else None
            if funder is not None and funder.role in (RoleType.PAYROLL_OPERATOR.value, RoleType.DISTRIBUTOR.value) and state.distinct_in <= 2:
                role = RoleType.SALARY_RECEIVER
            elif idle >= DORMANCY_MINIMUM_MINUTES and state.mean_gap_minutes > 0 and idle >= DORMANCY_GAP_MULTIPLE * state.mean_gap_minutes:
                role = RoleType.DORMANT
            elif state.events >= BEHAVIOUR_MINIMUM_OBSERVATIONS:
                role = RoleType.ACTIVE_COUNTERPARTY

        previous = RoleType(state.role) if state.role else RoleType.UNKNOWN
        confidence = round(max((item.confidence for item in support), default=0.0), 6) if support else (
            0.5 if role is not RoleType.UNKNOWN else 0.0)
        transition: RoleTransition | None = None
        if role is not previous:
            transition = RoleTransition(
                id=behaviour_id("RT", f"{previous.value}->{role.value}", state.role if state.role else "Unknown", transaction.id),
                subject_id=transaction.originator_account_id,
                from_role=previous, to_role=role, at_minute=minute, at_event=transaction.id,
                caused_by=tuple(item.id for item in support),
                explanation=(
                    f"{previous.value} -> {role.value}: "
                    + (", ".join(sorted(item.type.value for item in support)) if support
                       else "no behavioural support; the role follows from lifecycle state alone")
                ),
            )
        role_object = RoleObject(
            id=behaviour_id("RO", role.value, transaction.originator_account_id, transaction.id),
            subject_id=transaction.originator_account_id, role=role, confidence=confidence,
            since_minute=state.role_since if state.role_since >= 0 and role is previous else minute,
            tenure_minutes=max(0, minute - state.role_since) if state.role_since >= 0 and role is previous else 0,
            transition_count=state.transitions + (1 if transition else 0),
            supporting_behaviours=tuple(item.id for item in support),
            previous_role=previous,
            explanation=(f"Role {role.value} follows from "
                         + (", ".join(sorted(item.type.value for item in support)) if support else "lifecycle state")),
            version=BEHAVIOUR_LAYER_VERSION,
        )
        return role_object, transition

    # -- scenarios --------------------------------------------------------
    @staticmethod
    def _stage(state: AccountBehaviourState, flags: SemanticFlags, minute: int) -> Stage:
        if flags.self_posting or flags.intra_customer:
            return Stage.SETTLE
        if state.forwards_in(Horizon.MINUTES) >= 1 or flags.layering:
            return Stage.FORWARD
        if state.new_out_in(Horizon.MINUTES) >= 3:
            return Stage.SPLIT
        if state.h_out >= PAYROLL_MINIMUM_PAYMENTS and state.outbound_dispersion <= PAYROLL_AMOUNT_DISPERSION:
            return Stage.DISTRIBUTE
        return Stage.PAYMENTS

    def _scenarios(self, state: AccountBehaviourState, stage: Stage, objects: list[BehaviourObject], account: str, minute: int) -> list[ScenarioObject]:
        history = list(state.stages or ())
        observed = tuple(_STAGE_BY_CODE[code] for code, _ in history[-(SCENARIO_STAGE_MEMORY - 1):]) + (stage,)
        start_minute = history[-(SCENARIO_STAGE_MEMORY - 1)][1] if len(history) >= SCENARIO_STAGE_MEMORY - 1 else (history[0][1] if history else minute)
        results: list[ScenarioObject] = []
        for scenario, pattern, meaning in SCENARIO_PATTERNS:
            if not _is_subsequence(pattern, observed):
                continue
            confidence = round(SCENARIO_PRIOR * len(pattern) / (len(pattern) + 1), 6)
            counter: tuple[str, ...] = ()
            if scenario in _RISK_SCENARIOS and any(item.direction == "mitigation" for item in objects):
                counter = tuple(f"{item.type.value} argues for an ordinary reading" for item in objects if item.direction == "mitigation")
            results.append(ScenarioObject(
                id=behaviour_id("SC", scenario.value, account, str(minute)),
                type=scenario, subject_id=account, confidence=confidence,
                matched_stages=pattern, observed_stages=observed,
                interval=TimeInterval(start_minute, minute, Horizon.HOURS),
                supporting_behaviours=tuple(item.id for item in objects),
                counter_evidence=counter, causal_explanation=meaning,
                version=BEHAVIOUR_LAYER_VERSION,
            ))
        return results

    # -- causal commit ----------------------------------------------------
    def commit(self, transaction: Transaction, reading: SemanticContextResult, behaviour: BehaviourReading, minute: int) -> None:
        flags = SemanticFlags(reading)
        account = transaction.originator_account_id
        self.engine.commit(
            account, transaction.beneficiary_account_id, minute, transaction.amount, transaction.currency,
            self_posting=flags.self_posting, new_counterparty=flags.new_counterparty,
            established=flags.established, cash_instrument=flags.cash,
            cross_jurisdiction=flags.cross_jurisdiction, regime_stable=flags.stable, regime_broken=flags.broken,
        )
        state = self.engine.state(account)
        if behaviour.role.role.value != state.role:
            state.role_prev = state.role
            state.role = behaviour.role.role.value
            state.role_since = minute
            state.transitions += 1
        history = list(state.stages or ())
        history.append((_STAGE_CODES[behaviour.stage], minute))
        state.stages = tuple(history[-SCENARIO_STAGE_MEMORY:])
        if not flags.self_posting:
            target = self.engine.state(transaction.beneficiary_account_id)
            received = list(target.stages or ())
            received.append((_STAGE_CODES[Stage.RECEIVE], minute))
            target.stages = tuple(received[-SCENARIO_STAGE_MEMORY:])


def _is_subsequence(pattern: tuple[Stage, ...], observed: tuple[Stage, ...]) -> bool:
    position = 0
    for stage in observed:
        if position < len(pattern) and stage is pattern[position]:
            position += 1
    return position == len(pattern)


def _horizon_of(type_: BehaviourType) -> Horizon:
    from .ontology import BEHAVIOUR_SPECIFICATIONS

    return BEHAVIOUR_SPECIFICATIONS[type_].horizon


#: Role priority.  The first matching family wins; ordering is declared, so a
#: transit reading is never hidden behind a distribution reading.
_ROLE_PRIORITY: tuple[tuple[RoleType, frozenset[BehaviourType]], ...] = (
    (RoleType.MONEY_MULE_CANDIDATE, frozenset({BehaviourType.MONEY_MULE_BEHAVIOUR})),
    (RoleType.SHELL_COMPANY_CANDIDATE, frozenset({BehaviourType.SHELL_COMPANY_BEHAVIOUR})),
    (RoleType.TRANSIT_ACCOUNT, frozenset({BehaviourType.PASS_THROUGH_BEHAVIOUR, BehaviourType.TRANSIT_BEHAVIOUR,
                                          BehaviourType.RAPID_LAYERING_BEHAVIOUR, BehaviourType.HIGH_VELOCITY_LAYERING})),
    (RoleType.PAYROLL_OPERATOR, frozenset({BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR})),
    (RoleType.SETTLEMENT_HUB, frozenset({BehaviourType.SETTLEMENT_HUB_BEHAVIOUR})),
    (RoleType.TREASURY_ACCOUNT, frozenset({BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR})),
    (RoleType.COLLECTOR, frozenset({BehaviourType.COLLECTION_BEHAVIOUR, BehaviourType.FAN_IN_COLLECTION,
                                    BehaviourType.CASH_CONCENTRATION_BEHAVIOUR})),
    (RoleType.DISTRIBUTOR, frozenset({BehaviourType.DISTRIBUTION_BEHAVIOUR, BehaviourType.FAN_OUT_DISTRIBUTION,
                                      BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR})),
    (RoleType.ACCUMULATOR, frozenset({BehaviourType.MONEY_ACCUMULATION_BEHAVIOUR})),
)

_RISK_SCENARIOS = frozenset({ScenarioType.LAYERING_ATTEMPT, ScenarioType.COLLECT_THEN_FORWARD,
                             ScenarioType.DORMANCY_AFTER_OUTFLOW})
