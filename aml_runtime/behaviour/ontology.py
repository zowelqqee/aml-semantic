"""The closed vocabulary of the Semantic Behaviour Layer.

This module adds a second, higher-level vocabulary on top of the Semantic
Context Layer.  It does not modify or reinterpret that layer: every behaviour
claim below is an aggregate over *semantic objects observed across time*, never
over transaction columns.

As in `semantic/ontology.py`, every constant is declared from the meaning of the
claim it governs.  None is fitted or selected against an evaluation label, and
changing any of them changes ``BEHAVIOUR_ONTOLOGY_HASH``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

BEHAVIOUR_ONTOLOGY_VERSION = "aml-behaviour-ontology/1.0"


class Horizon(str, Enum):
    """The temporal scales the engine reasons over."""

    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


#: Width of each horizon, in minutes.  A horizon is evaluated as a tumbling
#: bucket plus its predecessor, so the effective trailing window is between one
#: and two widths.  This is exact, deterministic and cheap; it is declared here
#: rather than hidden in the implementation.
HORIZON_MINUTES: dict[Horizon, int] = {
    Horizon.MINUTES: 15,
    Horizon.HOURS: 60,
    Horizon.DAYS: 1_440,
    Horizon.WEEKS: 10_080,
}


class BehaviourType(str, Enum):
    """Behavioural hypotheses.  Not labels: each carries counter-evidence."""

    # -- shape over time -------------------------------------------------
    DISTRIBUTION_BEHAVIOUR = "DistributionBehaviour"
    FAN_OUT_DISTRIBUTION = "FanOutDistribution"
    COLLECTION_BEHAVIOUR = "CollectionBehaviour"
    FAN_IN_COLLECTION = "FanInCollection"
    SETTLEMENT_HUB_BEHAVIOUR = "SettlementHubBehaviour"

    # -- flow over time --------------------------------------------------
    MONEY_ACCUMULATION_BEHAVIOUR = "MoneyAccumulationBehaviour"
    TRANSIT_BEHAVIOUR = "TransitBehaviour"
    PASS_THROUGH_BEHAVIOUR = "PassThroughBehaviour"
    LIQUIDITY_BALANCING_BEHAVIOUR = "LiquidityBalancingBehaviour"
    CASH_CONCENTRATION_BEHAVIOUR = "CashConcentrationBehaviour"

    # -- tempo over time -------------------------------------------------
    BURST_ACTIVITY_BEHAVIOUR = "BurstActivityBehaviour"
    HIGH_VELOCITY_LAYERING = "HighVelocityLayering"
    RAPID_LAYERING_BEHAVIOUR = "RapidLayeringBehaviour"
    DORMANT_ACCOUNT_ACTIVATION = "DormantAccountActivation"

    # -- network motif over time -----------------------------------------
    CIRCULAR_MONEY_MOVEMENT = "CircularMoneyMovement"

    # -- relationship evolution ------------------------------------------
    RELATIONSHIP_GROWTH_BEHAVIOUR = "RelationshipGrowthBehaviour"
    RELATIONSHIP_COLLAPSE_BEHAVIOUR = "RelationshipCollapseBehaviour"

    # -- regime over time (the mitigating pole) ---------------------------
    ROUTINE_PAYROLL_BEHAVIOUR = "RoutinePayrollBehaviour"
    PAYROLL_OPERATOR_BEHAVIOUR = "PayrollOperatorBehaviour"
    SUPPLIER_SETTLEMENT_BEHAVIOUR = "SupplierSettlementBehaviour"
    EXPECTED_BUSINESS_CYCLE = "ExpectedBusinessCycle"
    UNEXPECTED_BUSINESS_CYCLE = "UnexpectedBusinessCycle"

    # -- composite hypotheses ---------------------------------------------
    MONEY_MULE_BEHAVIOUR = "MoneyMuleBehaviour"
    SHELL_COMPANY_BEHAVIOUR = "ShellCompanyBehaviour"

    # -- honesty ------------------------------------------------------------
    INSUFFICIENT_BEHAVIOURAL_HISTORY = "InsufficientBehaviouralHistory"


class RoleType(str, Enum):
    """Roles are dynamic states of an account, not classifications of it."""

    UNKNOWN = "Unknown"
    ACTIVE_COUNTERPARTY = "ActiveCounterparty"
    SALARY_RECEIVER = "SalaryReceiver"
    DISTRIBUTOR = "Distributor"
    COLLECTOR = "Collector"
    SETTLEMENT_HUB = "SettlementHub"
    PAYROLL_OPERATOR = "PayrollOperator"
    TREASURY_ACCOUNT = "TreasuryAccount"
    ACCUMULATOR = "Accumulator"
    TRANSIT_ACCOUNT = "TransitAccount"
    MONEY_MULE_CANDIDATE = "MoneyMuleCandidate"
    SHELL_COMPANY_CANDIDATE = "ShellCompanyCandidate"
    DORMANT = "Dormant"


class Stage(str, Enum):
    """Atomic steps a scenario is built from."""

    RECEIVE = "Receive"
    HOLD = "Hold"
    SPLIT = "Split"
    FORWARD = "Forward"
    PAYMENTS = "Payments"
    DISTRIBUTE = "Distribute"
    SETTLE = "Settle"
    DORMANT = "Dormant"


class ScenarioType(str, Enum):
    """Ordered stage patterns.  A scenario is a hypothesis about a story."""

    LAYERING_ATTEMPT = "LayeringAttempt"
    COLLECT_THEN_FORWARD = "CollectThenForward"
    DISTRIBUTION_RUN = "DistributionRun"
    NORMAL_CONSUMER_BEHAVIOUR = "NormalConsumerBehaviour"
    TREASURY_CYCLING = "TreasuryCycling"
    DORMANCY_AFTER_OUTFLOW = "DormancyAfterOutflow"


#: Declared stage patterns, matched as an ordered subsequence of the account's
#: recent stage history.  Order matters; gaps are permitted.
SCENARIO_PATTERNS: tuple[tuple[ScenarioType, tuple[Stage, ...], str], ...] = (
    (ScenarioType.LAYERING_ATTEMPT, (Stage.RECEIVE, Stage.SPLIT, Stage.FORWARD),
     "Value arrived, was broken up, and was forwarded onward."),
    (ScenarioType.COLLECT_THEN_FORWARD, (Stage.RECEIVE, Stage.RECEIVE, Stage.RECEIVE, Stage.FORWARD),
     "Value arrived from several sources and was forwarded onward."),
    (ScenarioType.DORMANCY_AFTER_OUTFLOW, (Stage.FORWARD, Stage.DORMANT),
     "Value was forwarded and the account then went quiet."),
    (ScenarioType.DISTRIBUTION_RUN, (Stage.RECEIVE, Stage.DISTRIBUTE),
     "Value arrived and was paid out across many counterparties at once."),
    (ScenarioType.NORMAL_CONSUMER_BEHAVIOUR, (Stage.RECEIVE, Stage.PAYMENTS, Stage.PAYMENTS),
     "Value arrived and was spent in ordinary routine payments."),
    (ScenarioType.TREASURY_CYCLING, (Stage.SETTLE, Stage.SETTLE),
     "Activity is internal settlement between accounts of one party."),
)

#: How many recent stages a scenario may be matched against.
SCENARIO_STAGE_MEMORY = 6

#: Strength of a fully matched scenario before pattern-length support is applied.
SCENARIO_PRIOR = 0.90


@dataclass(frozen=True)
class BehaviourSpecification:
    """What a behaviour claims, over which horizon, and how strong it can be."""

    type: BehaviourType
    horizon: Horizon
    direction: str
    topic: str
    prior: float
    half_support: int
    meaning: str


def _behaviour(type_: BehaviourType, horizon: Horizon, direction: str, topic: str, prior: float, half_support: int, meaning: str):
    return type_, BehaviourSpecification(type_, horizon, direction, topic, prior, half_support, meaning)


#: ``topic`` deliberately reuses ``network_motif`` for motif-class behaviours so
#: that the frozen semantic policy treats them as high-concern without any
#: change to that policy.  All other topics are new names that simply take part
#: in the existing corroboration count.
BEHAVIOUR_SPECIFICATIONS: dict[BehaviourType, BehaviourSpecification] = dict((
    _behaviour(BehaviourType.DISTRIBUTION_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_shape", 0.85, 20,
               "Sustained one-to-many payment shape across more than one time bucket."),
    _behaviour(BehaviourType.FAN_OUT_DISTRIBUTION, Horizon.MINUTES, "risk", "behaviour_shape", 0.80, 10,
               "Many distinct beneficiaries paid inside a single short bucket."),
    _behaviour(BehaviourType.COLLECTION_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_shape", 0.85, 20,
               "Sustained many-to-one receipt shape across more than one time bucket."),
    _behaviour(BehaviourType.FAN_IN_COLLECTION, Horizon.MINUTES, "risk", "behaviour_shape", 0.80, 10,
               "Value received from many distinct sources inside a single short bucket."),
    _behaviour(BehaviourType.SETTLEMENT_HUB_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_shape", 0.85, 20,
               "Symmetric two-way degree at scale: an infrastructural settlement point."),
    _behaviour(BehaviourType.MONEY_ACCUMULATION_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_flow", 0.80, 10,
               "Inflow materially exceeds outflow over the horizon: value is being gathered."),
    _behaviour(BehaviourType.TRANSIT_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_flow", 0.85, 10,
               "Value arrives and leaves at comparable scale within the horizon."),
    _behaviour(BehaviourType.PASS_THROUGH_BEHAVIOUR, Horizon.MINUTES, "risk", "behaviour_flow", 0.88, 6,
               "Received value is forwarded onward within minutes, repeatedly."),
    _behaviour(BehaviourType.LIQUIDITY_BALANCING_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_flow", 0.90, 10,
               "Two-way flow dominated by self-postings and intra-party movement."),
    _behaviour(BehaviourType.CASH_CONCENTRATION_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_flow", 0.88, 8,
               "Fan-in whose inflows are materially settled by cash instruments."),
    _behaviour(BehaviourType.BURST_ACTIVITY_BEHAVIOUR, Horizon.MINUTES, "risk", "behaviour_tempo", 0.80, 8,
               "Outbound tempo in this bucket is far above the account's own mean bucket rate."),
    _behaviour(BehaviourType.HIGH_VELOCITY_LAYERING, Horizon.MINUTES, "risk", "network_motif", 0.92, 4,
               "Repeated forward-with-preservation events inside a single short bucket."),
    _behaviour(BehaviourType.RAPID_LAYERING_BEHAVIOUR, Horizon.HOURS, "risk", "network_motif", 0.90, 6,
               "Repeated forward-with-preservation events sustained across the horizon."),
    _behaviour(BehaviourType.DORMANT_ACCOUNT_ACTIVATION, Horizon.HOURS, "risk", "behaviour_lifecycle", 0.82, 4,
               "Activity resumed after an idle period long relative to the account's own tempo."),
    _behaviour(BehaviourType.CIRCULAR_MONEY_MOVEMENT, Horizon.HOURS, "risk", "network_motif", 0.93, 4,
               "Value returned to a party that recently funded this account."),
    _behaviour(BehaviourType.RELATIONSHIP_GROWTH_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_relationship", 0.75, 10,
               "The counterparty set is expanding faster than the account's own history supports."),
    _behaviour(BehaviourType.RELATIONSHIP_COLLAPSE_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_relationship", 0.75, 10,
               "A previously broad counterparty set has contracted onto a single destination."),
    _behaviour(BehaviourType.ROUTINE_PAYROLL_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_regime", 0.88, 10,
               "Repeated fan-out of homogeneous amounts: a payment run, not a dispersal."),
    _behaviour(BehaviourType.PAYROLL_OPERATOR_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_regime", 0.90, 20,
               "Sustained payment-run shape across buckets: an operator, not an event."),
    _behaviour(BehaviourType.SUPPLIER_SETTLEMENT_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_regime", 0.85, 8,
               "Repeated payments into an established, stable counterparty set."),
    _behaviour(BehaviourType.EXPECTED_BUSINESS_CYCLE, Horizon.HOURS, "mitigation", "behaviour_stability", 0.86, 10,
               "Value and tempo have both stayed inside the party's own regimes across buckets."),
    _behaviour(BehaviourType.UNEXPECTED_BUSINESS_CYCLE, Horizon.HOURS, "risk", "behaviour_stability", 0.86, 10,
               "A previously stable value or tempo regime has broken."),
    _behaviour(BehaviourType.MONEY_MULE_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_composite", 0.90, 6,
               "Collection from several sources, prompt forwarding onward, and no established baseline."),
    _behaviour(BehaviourType.SHELL_COMPANY_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_composite", 0.88, 6,
               "An incorporated party whose account only transits value across jurisdictions with no operating regime."),
    _behaviour(BehaviourType.INSUFFICIENT_BEHAVIOURAL_HISTORY, Horizon.MINUTES, "none", "behaviour_coverage", 0.99, 1,
               "Too few observations exist for any behavioural claim about this account."),
))


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

#: Observations required before any behavioural claim about an account.
BEHAVIOUR_MINIMUM_OBSERVATIONS = 6

#: Distinct counterparties that make a fan shape a shape rather than a coincidence.
FAN_MINIMUM_COUNTERPARTIES = 8

#: ...and the degree asymmetry a directional shape requires.
FAN_ASYMMETRY_RATIO = 4.0

#: Buckets a shape must persist over before it is called sustained behaviour.
SUSTAINED_BUCKET_MINIMUM = 2

#: Symmetric degree at which an account looks infrastructural rather than directional.
HUB_MINIMUM_DEGREE = 20

#: ...and the maximum asymmetry a hub may show.
HUB_MAXIMUM_ASYMMETRY = 2.0

#: Retained share of inflow above which value is being accumulated, not moved.
ACCUMULATION_RETENTION = 0.70

#: Inflow/outflow agreement below which value is transiting.
TRANSIT_RETENTION_TOLERANCE = 0.25

#: Observations in each direction before a flow claim is made.
FLOW_MINIMUM_EVENTS = 4

#: Minutes within which a forwarded amount counts as prompt.
PROMPT_FORWARD_MINUTES = 30

#: Value agreement between an inflow and its forward.
FORWARD_VALUE_TOLERANCE = 0.15

#: Prompt forwards needed before forwarding is called a behaviour.
FORWARD_EVENT_MINIMUM = 3

#: Share of an account's *outbound* events that must be self-directed before
#: its flow is called internal treasury movement.  The denominator is outbound
#: rather than total events: a self-posting increments both directions, so a
#: total-event denominator would cap the ratio at 0.5 and make the claim
#: unsatisfiable for an account that does nothing else.
LIQUIDITY_SELF_POSTING_SHARE = 0.80

#: Share of inflow settled in cash before concentration is asserted.
CASH_CONCENTRATION_SHARE = 0.30

#: Tempo multiple over the account's own mean bucket rate that makes a burst.
BURST_RATE_MULTIPLE = 4.0

#: ...and the floor below which a burst is not a burst.
BURST_MINIMUM_EVENTS = 4

#: Idle multiple of the account's own mean gap that counts as dormancy.
DORMANCY_GAP_MULTIPLE = 8.0

#: ...with an absolute floor so short-lived accounts cannot trigger it.
DORMANCY_MINIMUM_MINUTES = 60

#: Coefficient of variation below which a set of outbound amounts is homogeneous.
PAYROLL_AMOUNT_DISPERSION = 0.25

#: Payments in a bucket before homogeneity means a payment run.
PAYROLL_MINIMUM_PAYMENTS = 5

#: Established counterparties needed before repeat payment is supplier settlement.
SUPPLIER_MINIMUM_ESTABLISHED = 3

#: Buckets of unbroken regime before behaviour is called stable.
STABILITY_BUCKET_MINIMUM = 3

#: New counterparties per bucket, relative to the account's own set, that count
#: as expansion rather than ordinary use.
RELATIONSHIP_GROWTH_RATIO = 0.50

#: Provenance depth carried on inflows for cycle detection.  Two hops is what a
#: single scalar per account can support exactly; deeper cycles are not claimed.
PROVENANCE_DEPTH = 2

#: Horizons that this source cannot fill.  The frozen 500,000-event prefix spans
#: 5.6 hours, so day and week buckets never accumulate.  Declared, not faked.
UNFILLABLE_HORIZONS_ON_SOURCE = (Horizon.DAYS, Horizon.WEEKS)

#: Behaviours the catalog names but that this window cannot causally support.
UNSUPPORTED_ON_SOURCE = (
    ("SeasonalBusinessCycle", ("multi_season_history",)),
    ("MonthlySalaryCadence", ("multi_month_history",)),
    ("LongTermDormancyReactivation", ("multi_week_history",)),
    ("AnnualTaxCycle", ("multi_year_history",)),
)


def behaviour_confidence(type_: BehaviourType, observations: int, present_inputs: int, declared_inputs: int) -> tuple[float, str]:
    """prior x support x coverage, identical in form to the semantic layer."""
    specification = BEHAVIOUR_SPECIFICATIONS[type_]
    support = observations / (observations + specification.half_support) if observations > 0 else 0.0
    coverage = 1.0 if declared_inputs <= 0 else present_inputs / declared_inputs
    value = round(specification.prior * support * coverage, 6)
    explanation = (
        f"prior {specification.prior:.2f} x support {support:.4f} "
        f"(n={observations}, k={specification.half_support}) x coverage {coverage:.2f} = {value:.6f}"
    )
    return value, explanation


def _payload() -> dict[str, object]:
    return {
        "version": BEHAVIOUR_ONTOLOGY_VERSION,
        "horizons": {key.value: value for key, value in HORIZON_MINUTES.items()},
        "behaviours": {
            key.value: {"horizon": value.horizon.value, "direction": value.direction, "topic": value.topic,
                        "prior": value.prior, "half_support": value.half_support, "meaning": value.meaning}
            for key, value in sorted(BEHAVIOUR_SPECIFICATIONS.items(), key=lambda item: item[0].value)
        },
        "roles": [item.value for item in RoleType],
        "stages": [item.value for item in Stage],
        "scenarios": [[item[0].value, [stage.value for stage in item[1]], item[2]] for item in SCENARIO_PATTERNS],
        "constants": {
            "BEHAVIOUR_MINIMUM_OBSERVATIONS": BEHAVIOUR_MINIMUM_OBSERVATIONS,
            "FAN_MINIMUM_COUNTERPARTIES": FAN_MINIMUM_COUNTERPARTIES,
            "FAN_ASYMMETRY_RATIO": FAN_ASYMMETRY_RATIO,
            "SUSTAINED_BUCKET_MINIMUM": SUSTAINED_BUCKET_MINIMUM,
            "HUB_MINIMUM_DEGREE": HUB_MINIMUM_DEGREE,
            "HUB_MAXIMUM_ASYMMETRY": HUB_MAXIMUM_ASYMMETRY,
            "ACCUMULATION_RETENTION": ACCUMULATION_RETENTION,
            "TRANSIT_RETENTION_TOLERANCE": TRANSIT_RETENTION_TOLERANCE,
            "FLOW_MINIMUM_EVENTS": FLOW_MINIMUM_EVENTS,
            "PROMPT_FORWARD_MINUTES": PROMPT_FORWARD_MINUTES,
            "FORWARD_VALUE_TOLERANCE": FORWARD_VALUE_TOLERANCE,
            "FORWARD_EVENT_MINIMUM": FORWARD_EVENT_MINIMUM,
            "LIQUIDITY_SELF_POSTING_SHARE": LIQUIDITY_SELF_POSTING_SHARE,
            "CASH_CONCENTRATION_SHARE": CASH_CONCENTRATION_SHARE,
            "BURST_RATE_MULTIPLE": BURST_RATE_MULTIPLE,
            "BURST_MINIMUM_EVENTS": BURST_MINIMUM_EVENTS,
            "DORMANCY_GAP_MULTIPLE": DORMANCY_GAP_MULTIPLE,
            "DORMANCY_MINIMUM_MINUTES": DORMANCY_MINIMUM_MINUTES,
            "PAYROLL_AMOUNT_DISPERSION": PAYROLL_AMOUNT_DISPERSION,
            "PAYROLL_MINIMUM_PAYMENTS": PAYROLL_MINIMUM_PAYMENTS,
            "SUPPLIER_MINIMUM_ESTABLISHED": SUPPLIER_MINIMUM_ESTABLISHED,
            "STABILITY_BUCKET_MINIMUM": STABILITY_BUCKET_MINIMUM,
            "RELATIONSHIP_GROWTH_RATIO": RELATIONSHIP_GROWTH_RATIO,
            "PROVENANCE_DEPTH": PROVENANCE_DEPTH,
            "SCENARIO_STAGE_MEMORY": SCENARIO_STAGE_MEMORY,
            "SCENARIO_PRIOR": SCENARIO_PRIOR,
        },
        "unfillable_horizons": [item.value for item in UNFILLABLE_HORIZONS_ON_SOURCE],
        "unsupported_on_source": [list(item) for item in UNSUPPORTED_ON_SOURCE],
    }


BEHAVIOUR_ONTOLOGY_HASH = hashlib.sha256(
    json.dumps(_payload(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
).hexdigest()
