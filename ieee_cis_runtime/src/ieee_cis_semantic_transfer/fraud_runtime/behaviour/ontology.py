"""The closed vocabulary of the fraud Behaviour Layer.

Same mechanism as ``aml_runtime/behaviour/ontology.py`` — behaviour is an
aggregate over *semantic objects observed across time*, never over
transaction columns directly, confidence is ``prior x support x coverage``,
and every declared type states its own horizon, direction and topic.

Three of the brief's example terms are declared here as unsupported and are
never emitted, for the same reason ``aml_runtime`` declined to fabricate
``SalaryDistribution`` on a source with no payroll calendar: the public
IEEE-CIS release has no merchant identifier (so neither *Merchant Abuse* nor
*Trusted Merchant* can be built without inventing one), and no
authentication-attempt log (so *Credential Stuffing*, which is fundamentally
about failed logins, cannot be distinguished from ordinary device use here —
the closest available proxy, the IP-proxy-type flag ``id_23``, covers 3.6% of
the already-sparse identity join and is too thin to support a claim).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

BEHAVIOUR_ONTOLOGY_VERSION = "fraud-behaviour-ontology/1.0"


class Horizon(str, Enum):
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


#: Same tumbling-bucket-plus-predecessor discipline as aml_runtime. Unlike the
#: AML port, IEEE-CIS spans 182 days end to end, so DAYS and WEEKS are not
#: structurally unfillable here — they are declared but this behaviour layer's
#: predicates only read MINUTES/HOURS, matching what aml_runtime's own
#: predicates actually used regardless of which horizons were fillable.
HORIZON_MINUTES: dict[Horizon, int] = {
    Horizon.MINUTES: 15,
    Horizon.HOURS: 60,
    Horizon.DAYS: 1_440,
    Horizon.WEEKS: 10_080,
}


class BehaviourType(str, Enum):
    CARD_TESTING_BEHAVIOUR = "CardTestingBehaviour"
    DEVICE_ROTATION_BEHAVIOUR = "DeviceRotationBehaviour"
    VELOCITY_BURST_BEHAVIOUR = "VelocityBurstBehaviour"
    COMPROMISED_CARD_BEHAVIOUR = "CompromisedCardBehaviour"
    UNEXPECTED_SPENDING_BEHAVIOUR = "UnexpectedSpendingBehaviour"
    DORMANT_CARD_REACTIVATION = "DormantCardReactivation"
    TRUSTED_DEVICE_BEHAVIOUR = "TrustedDeviceBehaviour"
    NORMAL_SPENDING_BEHAVIOUR = "NormalSpendingBehaviour"
    EXPECTED_VELOCITY_BEHAVIOUR = "ExpectedVelocityBehaviour"
    INSUFFICIENT_CARD_HISTORY = "InsufficientCardHistory"


class RoleType(str, Enum):
    UNKNOWN = "Unknown"
    ACTIVE_CARD = "ActiveCard"
    TRUSTED_CARD = "TrustedCard"
    TESTED_CARD = "TestedCard"
    HIGH_VELOCITY_CARD = "HighVelocityCard"
    ROTATING_DEVICE_CARD = "RotatingDeviceCard"
    COMPROMISED_CARD_CANDIDATE = "CompromisedCardCandidate"
    DORMANT_CARD = "DormantCard"


class Stage(str, Enum):
    PURCHASE = "Purchase"
    TEST = "Test"
    NEW_DEVICE = "NewDevice"
    BURST = "Burst"
    REGION_CHANGE = "RegionChange"
    DORMANT = "Dormant"


class ScenarioType(str, Enum):
    CARD_TESTING_RUN = "CardTestingRun"
    DEVICE_TAKEOVER_PATTERN = "DeviceTakeoverPattern"
    ROUTINE_CONSUMER_PATTERN = "RoutineConsumerPattern"
    DORMANCY_AFTER_BURST = "DormancyAfterBurst"
    DEVICE_ROTATION_RUN = "DeviceRotationRun"


SCENARIO_PATTERNS: tuple[tuple[ScenarioType, tuple[Stage, ...], str], ...] = (
    (ScenarioType.CARD_TESTING_RUN, (Stage.TEST, Stage.TEST, Stage.PURCHASE),
     "Several minimal test charges followed by a real purchase."),
    (ScenarioType.DEVICE_TAKEOVER_PATTERN, (Stage.PURCHASE, Stage.PURCHASE, Stage.NEW_DEVICE, Stage.BURST),
     "An established pattern is followed by a new device and a sudden burst."),
    (ScenarioType.ROUTINE_CONSUMER_PATTERN, (Stage.PURCHASE, Stage.PURCHASE, Stage.PURCHASE),
     "Ordinary, repeated, regime-conforming purchases."),
    (ScenarioType.DORMANCY_AFTER_BURST, (Stage.BURST, Stage.DORMANT),
     "A burst of activity is followed by the card going quiet."),
    (ScenarioType.DEVICE_ROTATION_RUN, (Stage.NEW_DEVICE, Stage.NEW_DEVICE, Stage.NEW_DEVICE),
     "The card cycles through a run of previously-unseen devices."),
)

SCENARIO_STAGE_MEMORY = 6
SCENARIO_PRIOR = 0.90


@dataclass(frozen=True)
class BehaviourSpecification:
    type: BehaviourType
    horizon: Horizon
    direction: str
    topic: str
    prior: float
    half_support: int
    meaning: str


def _behaviour(type_: BehaviourType, horizon: Horizon, direction: str, topic: str, prior: float, half_support: int, meaning: str):
    return type_, BehaviourSpecification(type_, horizon, direction, topic, prior, half_support, meaning)


#: ``topic`` reuses the semantic layer's ``device_motif`` string for
#: motif-class behaviours, exactly as aml_runtime reused ``network_motif`` —
#: so the frozen ``SemanticPolicyEngine`` treats them as high-concern without
#: any change to that policy.
BEHAVIOUR_SPECIFICATIONS: dict[BehaviourType, BehaviourSpecification] = dict((
    _behaviour(BehaviourType.CARD_TESTING_BEHAVIOUR, Horizon.MINUTES, "risk", "behaviour_tempo", 0.90, 4,
               "Repeated minimal-amount charges inside a short window — the live-card-test pattern."),
    _behaviour(BehaviourType.DEVICE_ROTATION_BEHAVIOUR, Horizon.MINUTES, "risk", "device_motif", 0.85, 3,
               "This card cycled through several distinct devices inside a short window."),
    _behaviour(BehaviourType.VELOCITY_BURST_BEHAVIOUR, Horizon.MINUTES, "risk", "behaviour_tempo", 0.80, 4,
               "Purchase tempo in this bucket is far above the card's own established rate."),
    _behaviour(BehaviourType.COMPROMISED_CARD_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_composite", 0.90, 4,
               "An established device relationship was displaced by a new device together with a regime break."),
    _behaviour(BehaviourType.UNEXPECTED_SPENDING_BEHAVIOUR, Horizon.HOURS, "risk", "behaviour_stability", 0.86, 6,
               "A previously stable amount regime has broken, sustained across observations."),
    _behaviour(BehaviourType.DORMANT_CARD_REACTIVATION, Horizon.HOURS, "risk", "behaviour_lifecycle", 0.82, 3,
               "Activity resumed after an idle period long relative to this card's own tempo."),
    _behaviour(BehaviourType.TRUSTED_DEVICE_BEHAVIOUR, Horizon.HOURS, "mitigation", "device_regime", 0.88, 8,
               "A long, stable, sustained history between this card and this device."),
    _behaviour(BehaviourType.NORMAL_SPENDING_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_stability", 0.86, 6,
               "Amount and tempo have both stayed inside the card's own regimes across observations."),
    _behaviour(BehaviourType.EXPECTED_VELOCITY_BEHAVIOUR, Horizon.HOURS, "mitigation", "behaviour_tempo", 0.85, 8,
               "Sustained high tempo that is nonetheless this card's own established rate."),
    _behaviour(BehaviourType.INSUFFICIENT_CARD_HISTORY, Horizon.MINUTES, "none", "behaviour_coverage", 0.99, 1,
               "Too few observations exist for any behavioural claim about this card."),
))

# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------

BEHAVIOUR_MINIMUM_OBSERVATIONS = 5
TEST_CHARGE_MINIMUM = 3
DEVICE_ROTATION_MINIMUM = 3
VELOCITY_BURST_MINIMUM_EVENTS = 4
VELOCITY_BURST_RATE_MULTIPLE = 4.0
COMPROMISE_DEVICE_MINIMUM_PRIOR = 3
STABILITY_OBSERVATION_MINIMUM = 6
DORMANCY_GAP_MULTIPLE = 8.0
DORMANCY_MINIMUM_MINUTES = 120
TRUSTED_DEVICE_MINIMUM_PAIR_COUNT = 8

#: This ontology names but this source cannot causally support.
UNSUPPORTED_ON_SOURCE = (
    ("MerchantAbuseBehaviour", ("merchant_identity",)),
    ("TrustedMerchantBehaviour", ("merchant_identity",)),
    ("CredentialStuffingBehaviour", ("authentication_log",)),
    ("DeviceSharingBehaviour", ("persistent_physical_device_identifier",)),
    ("SharedDeviceCluster", ("persistent_physical_device_identifier",)),
    ("ImpossibleTravelBehaviour", ("geocoded_origin", "geocoded_destination", "travel_time_model")),
    ("SyntheticIdentityBehaviour", ("verified_person_identity", "identity-linkage-ground-truth")),
)


def behaviour_confidence(type_: BehaviourType, observations: int, present_inputs: int, declared_inputs: int) -> tuple[float, str]:
    specification = BEHAVIOUR_SPECIFICATIONS[type_]
    support = observations / (observations + specification.half_support) if observations > 0 else 0.0
    coverage = 1.0 if declared_inputs <= 0 else present_inputs / declared_inputs
    value = round(specification.prior * support * coverage, 6)
    explanation = (f"prior {specification.prior:.2f} x support {support:.4f} "
                   f"(n={observations}, k={specification.half_support}) x coverage {coverage:.2f} = {value:.6f}")
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
            "TEST_CHARGE_MINIMUM": TEST_CHARGE_MINIMUM,
            "DEVICE_ROTATION_MINIMUM": DEVICE_ROTATION_MINIMUM,
            "VELOCITY_BURST_MINIMUM_EVENTS": VELOCITY_BURST_MINIMUM_EVENTS,
            "VELOCITY_BURST_RATE_MULTIPLE": VELOCITY_BURST_RATE_MULTIPLE,
            "COMPROMISE_DEVICE_MINIMUM_PRIOR": COMPROMISE_DEVICE_MINIMUM_PRIOR,
            "STABILITY_OBSERVATION_MINIMUM": STABILITY_OBSERVATION_MINIMUM,
            "DORMANCY_GAP_MULTIPLE": DORMANCY_GAP_MULTIPLE,
            "DORMANCY_MINIMUM_MINUTES": DORMANCY_MINIMUM_MINUTES,
            "TRUSTED_DEVICE_MINIMUM_PAIR_COUNT": TRUSTED_DEVICE_MINIMUM_PAIR_COUNT,
            "SCENARIO_STAGE_MEMORY": SCENARIO_STAGE_MEMORY,
            "SCENARIO_PRIOR": SCENARIO_PRIOR,
        },
        "unsupported_on_source": [list(item) for item in UNSUPPORTED_ON_SOURCE],
    }


BEHAVIOUR_ONTOLOGY_HASH = hashlib.sha256(
    json.dumps(_payload(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
).hexdigest()
