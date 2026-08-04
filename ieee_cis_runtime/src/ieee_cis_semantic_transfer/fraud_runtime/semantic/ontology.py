"""The closed vocabulary of the fraud Semantic Context Layer.

Structurally identical to ``aml_runtime/semantic/ontology.py``: a closed set of
typed claims, each with a declared prior, a declared half-support point, and a
declared list of inputs it reads. Confidence is ``prior x support x coverage``,
computed the same way, never fitted.

The vocabulary itself is new. IEEE-CIS card-purchase transactions have a
structurally different shape from AMLSim payments: a card does not *receive*
money the way an account does, so there is no accumulation/collection/transit
analogue at the level of a single card. What a card purchase *does* have is a
resolvable party (Card), a device it was made from, a spending channel
(ProductCD), and a billing region — and, for 24.4% of rows, a rich device
fingerprint. The ontology below is built from exactly that shape.

Declared not built, and stated as such: everything the public IEEE-CIS release
does not carry. There is no merchant identifier in this dataset (P/R
_emaildomain are the *purchaser's* and *recipient's* mail providers, not a
selling platform), no authentication-attempt log, and no calibrated
high-risk-region list (``addr1`` is an anonymised integer, not an ISO country
code). Terms that would require those inputs are declared
``UNSUPPORTED_ON_SOURCE`` and are never emitted — the same discipline
``aml_runtime`` applied to SAR/KYC/sanctions feeds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

ONTOLOGY_VERSION = "fraud-semantic-ontology/1.0"


class ObjectClass(str, Enum):
    ENTITY = "entity"
    PROFILE = "profile"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    COVERAGE = "coverage"


class SemanticType(str, Enum):
    """Every assertion the fraud layer is permitted to make."""

    # -- profile layer: reference frames for a card's own regime -----------
    NO_ESTABLISHED_CARD_HISTORY = "NoEstablishedCardHistory"
    AMOUNT_REGIME = "AmountRegime"
    TEMPO_REGIME = "TempoRegime"
    DEVICE_REGIME = "DeviceRegime"
    CHANNEL_REGIME = "ChannelRegime"

    # -- relationship layer: the card-device pairing, not card-card ---------
    FIRST_DEVICE_CONTACT = "FirstDeviceContact"
    RECENTLY_LINKED_DEVICE = "RecentlyLinkedDevice"
    ESTABLISHED_DEVICE_RELATIONSHIP = "EstablishedDeviceRelationship"
    NON_INFORMATIVE_DEVICE_NOVELTY = "NonInformativeDeviceNovelty"

    # -- event layer: what this purchase actually is -------------------------
    ROUTINE_SPENDING_AMOUNT = "RoutineSpendingAmount"
    EXPECTED_HIGH_VALUE_SPEND = "ExpectedHighValueSpend"
    UNEXPECTED_SPENDING_AMOUNT = "UnexpectedSpendingAmount"
    UNSCALED_SPENDING_AMOUNT = "UnscaledSpendingAmount"
    MINIMAL_TEST_AMOUNT = "MinimalTestAmount"
    UNEXPECTED_BILLING_REGION = "UnexpectedBillingRegion"
    TEMPO_REGIME_SHIFT = "TempoRegimeShift"
    UNVERIFIED_DEVICE_CONTEXT = "UnverifiedDeviceContext"
    SHARED_CLIENT_SIGNATURE_EXPOSURE = "SharedClientSignatureExposure"

    # -- honesty --------------------------------------------------------------
    COVERAGE_GAP = "CoverageGap"


@dataclass(frozen=True)
class TypeSpecification:
    type: SemanticType
    object_class: ObjectClass
    prior: float
    half_support: int
    inputs: tuple[str, ...]
    meaning: str


def _spec(type_: SemanticType, object_class: ObjectClass, prior: float, half_support: int, inputs: tuple[str, ...], meaning: str) -> tuple[SemanticType, TypeSpecification]:
    return type_, TypeSpecification(type_, object_class, prior, half_support, inputs, meaning)


SPECIFICATIONS: dict[SemanticType, TypeSpecification] = dict((
    _spec(SemanticType.NO_ESTABLISHED_CARD_HISTORY, ObjectClass.PROFILE, 0.99, 1, ("prior_purchase_count",),
          "Too little history exists for any claim about how this card normally behaves."),
    _spec(SemanticType.AMOUNT_REGIME, ObjectClass.PROFILE, 0.95, 8, ("prior_amounts",),
          "The card's own observed spending scale in USD."),
    _spec(SemanticType.TEMPO_REGIME, ObjectClass.PROFILE, 0.90, 8, ("prior_purchase_times",),
          "The card's own observed purchase tempo."),
    _spec(SemanticType.DEVICE_REGIME, ObjectClass.PROFILE, 0.90, 5, ("prior_devices",),
          "The set of devices this card has normally purchased from."),
    _spec(SemanticType.CHANNEL_REGIME, ObjectClass.PROFILE, 0.85, 5, ("prior_channels",),
          "The set of spending channels (ProductCD) this card normally uses."),

    _spec(SemanticType.FIRST_DEVICE_CONTACT, ObjectClass.RELATIONSHIP, 0.80, 5, ("device_history", "device_regime"),
          "A device never used before by a card whose device set is known."),
    _spec(SemanticType.RECENTLY_LINKED_DEVICE, ObjectClass.RELATIONSHIP, 0.75, 5, ("device_history", "device_regime"),
          "A card-device pairing formed within the observed window."),
    _spec(SemanticType.ESTABLISHED_DEVICE_RELATIONSHIP, ObjectClass.RELATIONSHIP, 0.85, 5, ("device_history",),
          "A card-device pairing the card uses repeatedly."),
    _spec(SemanticType.NON_INFORMATIVE_DEVICE_NOVELTY, ObjectClass.RELATIONSHIP, 0.90, 1, ("device_history", "prior_purchase_count"),
          "Device novelty observed against no device baseline; the default state of a new card, not a signal."),

    _spec(SemanticType.ROUTINE_SPENDING_AMOUNT, ObjectClass.EVENT, 0.85, 8, ("amount", "amount_regime"),
          "The amount sits inside the card's own established spending regime."),
    _spec(SemanticType.EXPECTED_HIGH_VALUE_SPEND, ObjectClass.EVENT, 0.88, 8, ("amount", "amount_regime"),
          "The amount is high in absolute terms but ordinary for this card."),
    _spec(SemanticType.UNEXPECTED_SPENDING_AMOUNT, ObjectClass.EVENT, 0.92, 8, ("amount", "amount_regime"),
          "The amount is materially outside the card's own established regime."),
    _spec(SemanticType.UNSCALED_SPENDING_AMOUNT, ObjectClass.EVENT, 0.99, 1, ("amount",),
          "An amount observed with no reference frame; explicit ignorance, not a finding."),
    _spec(SemanticType.MINIMAL_TEST_AMOUNT, ObjectClass.EVENT, 0.80, 3, ("amount", "amount_regime"),
          "A very small charge relative to typical purchase amounts — the atom of card testing."),
    _spec(SemanticType.UNEXPECTED_BILLING_REGION, ObjectClass.EVENT, 0.85, 5, ("billing_region", "region_history"),
          "The billing region differs from every region previously seen on this card."),
    _spec(SemanticType.TEMPO_REGIME_SHIFT, ObjectClass.EVENT, 0.88, 8, ("purchase_times", "tempo_regime"),
          "Purchase tempo is materially outside the card's own established tempo."),
    _spec(SemanticType.UNVERIFIED_DEVICE_CONTEXT, ObjectClass.EVENT, 0.90, 1, ("identity_join",),
          "This transaction did not join to the device/identity table; no device signal exists for it."),
    _spec(SemanticType.SHARED_CLIENT_SIGNATURE_EXPOSURE, ObjectClass.EVENT, 0.90, 3, ("client_signature_card_fanout",),
          "The observed DeviceType/DeviceInfo signature has recently occurred with other distinct card identities; this is signature reuse, not verified hardware sharing."),

    _spec(SemanticType.COVERAGE_GAP, ObjectClass.COVERAGE, 1.00, 1, ("source_schema",),
          "A declared input the ontology requires is absent from this source."),
))


# ---------------------------------------------------------------------------
# Declared constants — from the meaning of the claim, not fitted.
# ---------------------------------------------------------------------------

#: Prior purchases required before any claim about a card's normal behaviour.
BASELINE_MINIMUM_EVENTS = 5

#: A value is "materially outside" a regime when it exceeds the card's own
#: previous maximum by this factor. Identical multiple to aml_runtime's
#: VALUE_REGIME_BREAK_MULTIPLE, for the same reason: the smallest multiple
#: that cannot be reached by ordinary doubling of a previous peak.
AMOUNT_REGIME_BREAK_MULTIPLE = 4.0

#: Values at or above the card's own upper decile are "high" but not outside.
AMOUNT_REGIME_HIGH_QUANTILE = 0.90

#: The most recent amounts that define a card's regime.
AMOUNT_REGIME_RESERVOIR = 64

#: A charge at or below this fraction of the card's own regime median is a
#: candidate test amount — the classic card-testing pattern is a $0.50-$2.00
#: authorization to confirm a stolen card is live before a real purchase.
TEST_AMOUNT_MEDIAN_FRACTION = 0.05

#: ...with an absolute floor, since a fraction of a very large median could
#: still be an ordinary small purchase.
TEST_AMOUNT_ABSOLUTE_CEILING = 3.00

#: Tempo is measured in whole-hour buckets: TransactionDT is second-granular
#: but the claim is about rate, not exact spacing.
TEMPO_BUCKET_MINUTES = 60

#: A tempo claim requires at least this many events inside one bucket.
TEMPO_MINIMUM_BURST = 3

#: ...and that burst must exceed the card's own mean bucket rate by this factor.
TEMPO_SHIFT_MULTIPLE = 4.0

#: Pair (card, device) observations before a relationship counts as established.
ESTABLISHED_DEVICE_MINIMUM = 3

#: Distinct card identities on one client signature in the trailing window.
#: This is tracked as a neutral semantic context feature only; DeviceInfo does
#: not establish a persistent physical-device identifier.
SHARED_DEVICE_MINIMUM_CARDS = 3

#: Every semantic type this ontology declares is emittable from the public
#: IEEE-CIS release; nothing here required fabricating a missing input. The
#: SOURCE_COVERAGE_GAPS below record what the *source* is missing outright.
SOURCE_COVERAGE_GAPS = (
    ("merchant_identity", "No merchant or seller identifier exists in the public release; P_emaildomain/R_emaildomain are the purchaser's and recipient's mail providers, not a selling platform."),
    ("authentication_log", "No login or authentication-attempt event exists; only completed transactions are recorded."),
    ("calibrated_region_risk", "addr1/addr2 are anonymised integers with no published mapping to real jurisdictions; an absolute high-risk-region list cannot be constructed."),
    ("physical_distance_units", "dist1/dist2 carry no documented unit or reference point; only regime-relative claims about them are supportable."),
    ("device_identity_below_min_coverage", "75.6% of transactions do not join to the identity/device table at all; device-dependent claims are withheld on those rows."),
    ("physical_device_identity", "DeviceType and DeviceInfo describe a client signature/model or browser, not a documented unique hardware identifier; device-sharing claims are therefore not emitted."),
)

#: Semantic types the fraud domain plausibly wants but this source cannot
#: causally support. Declared, never emitted.
UNSUPPORTED_ON_SOURCE = (
    ("CrossJurisdictionSpend", ("originator_jurisdiction", "beneficiary_jurisdiction")),
    ("VirtualAssetExposure", ("venue_registry",)),
    ("CurrencyConversionSpend", ("multi_currency_ledger",)),
)


def support_factor(observations: int, half_support: int) -> float:
    if observations <= 0:
        return 0.0
    return observations / (observations + half_support)


def coverage_factor(present: int, declared: int) -> float:
    return 1.0 if declared <= 0 else present / declared


def confidence_for(type_: SemanticType, observations: int, present_inputs: int) -> tuple[float, str]:
    specification = SPECIFICATIONS[type_]
    support = support_factor(observations, specification.half_support)
    coverage = coverage_factor(present_inputs, len(specification.inputs))
    value = round(specification.prior * support * coverage, 6)
    explanation = (
        f"prior {specification.prior:.2f} x support {support:.4f} "
        f"(n={observations}, k={specification.half_support}) x coverage {coverage:.2f} "
        f"= {value:.6f}"
    )
    return value, explanation


def _payload() -> dict[str, object]:
    return {
        "version": ONTOLOGY_VERSION,
        "types": {
            key.value: {"class": value.object_class.value, "prior": value.prior,
                        "half_support": value.half_support, "inputs": list(value.inputs), "meaning": value.meaning}
            for key, value in sorted(SPECIFICATIONS.items(), key=lambda item: item[0].value)
        },
        "constants": {
            "BASELINE_MINIMUM_EVENTS": BASELINE_MINIMUM_EVENTS,
            "AMOUNT_REGIME_BREAK_MULTIPLE": AMOUNT_REGIME_BREAK_MULTIPLE,
            "AMOUNT_REGIME_HIGH_QUANTILE": AMOUNT_REGIME_HIGH_QUANTILE,
            "AMOUNT_REGIME_RESERVOIR": AMOUNT_REGIME_RESERVOIR,
            "TEST_AMOUNT_MEDIAN_FRACTION": TEST_AMOUNT_MEDIAN_FRACTION,
            "TEST_AMOUNT_ABSOLUTE_CEILING": TEST_AMOUNT_ABSOLUTE_CEILING,
            "TEMPO_BUCKET_MINUTES": TEMPO_BUCKET_MINUTES,
            "TEMPO_MINIMUM_BURST": TEMPO_MINIMUM_BURST,
            "TEMPO_SHIFT_MULTIPLE": TEMPO_SHIFT_MULTIPLE,
            "ESTABLISHED_DEVICE_MINIMUM": ESTABLISHED_DEVICE_MINIMUM,
            "SHARED_DEVICE_MINIMUM_CARDS": SHARED_DEVICE_MINIMUM_CARDS,
        },
        "coverage_gaps": [list(item) for item in SOURCE_COVERAGE_GAPS],
        "unsupported_on_source": [list(item) for item in UNSUPPORTED_ON_SOURCE],
    }


ONTOLOGY_HASH = hashlib.sha256(
    json.dumps(_payload(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
).hexdigest()
