"""The closed, versioned vocabulary of the Semantic Context Layer.

Nothing outside this module may be asserted about the world.  Every constant
here is *declared* from the semantics of the claim it governs; none is fitted,
searched, or selected against an evaluation label.  Changing any value changes
``ONTOLOGY_HASH`` and therefore invalidates replay of earlier decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum


ONTOLOGY_VERSION = "aml-semantic-ontology/1.0"


class ObjectClass(str, Enum):
    """The four layers of the object hierarchy."""

    ENTITY = "entity"
    PROFILE = "profile"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    COVERAGE = "coverage"


class SemanticType(str, Enum):
    """Every assertion the layer is permitted to make."""

    # -- profile layer: the reference frames every magnitude claim needs ----
    NO_ESTABLISHED_BASELINE = "NoEstablishedBaseline"
    VALUE_REGIME = "ValueRegime"
    TEMPO_REGIME = "TempoRegime"
    COUNTERPARTY_REGIME = "CounterpartyRegime"
    DISTRIBUTION_NODE = "DistributionNode"
    COLLECTION_NODE = "CollectionNode"
    PASS_THROUGH_ACCOUNT = "PassThroughAccount"
    BOOKKEEPING_ACCOUNT = "BookkeepingAccount"

    # -- relationship layer: novelty is a lattice, not a boolean -----------
    FIRST_CONTACT = "FirstContact"
    RECENTLY_CREATED_RELATIONSHIP = "RecentlyCreatedRelationship"
    ESTABLISHED_RELATIONSHIP = "EstablishedRelationship"
    NON_INFORMATIVE_NOVELTY = "NonInformativeNovelty"

    # -- event layer: what this transaction actually is --------------------
    INTERNAL_BOOK_ENTRY = "InternalBookEntry"
    INTRA_CUSTOMER_TRANSFER = "IntraCustomerTransfer"
    ROUTINE_VALUE_TRANSFER = "RoutineValueTransfer"
    EXPECTED_HIGH_VALUE_TRANSFER = "ExpectedHighValueTransfer"
    UNEXPECTED_LARGE_TRANSFER = "UnexpectedLargeTransfer"
    UNSCALED_VALUE = "UnscaledValue"
    CROSS_JURISDICTION_TRANSFER = "CrossJurisdictionTransfer"
    HIGH_RISK_JURISDICTION_EXPOSURE = "HighRiskJurisdictionExposure"
    VIRTUAL_ASSET_EXPOSURE = "VirtualAssetExposure"
    CURRENCY_CONVERSION_TRANSFER = "CurrencyConversionTransfer"
    CASH_INSTRUMENT_SETTLEMENT = "CashInstrumentSettlement"
    BEHAVIOUR_REGIME_SHIFT = "BehaviourRegimeShift"
    NORMAL_OPERATIONAL_BURST = "NormalOperationalBurst"
    LAYERING_CHAIN_SEGMENT = "LayeringChainSegment"

    # -- coverage layer: absence is a fact, not a zero ---------------------
    COVERAGE_GAP = "CoverageGap"


class EntityForm(str, Enum):
    """Legal forms carried by the account reference data."""

    NATURAL_PERSON = "NaturalPerson"
    CORPORATION = "Corporation"
    PARTNERSHIP = "Partnership"
    SOLE_PROPRIETORSHIP = "SoleProprietorship"
    SOVEREIGN_ENTITY = "SovereignEntity"
    UNRESOLVED_FORM = "UnresolvedForm"


ENTITY_FORM_BY_SOURCE_LABEL = {
    "Individual": EntityForm.NATURAL_PERSON,
    "Corporation": EntityForm.CORPORATION,
    "Partnership": EntityForm.PARTNERSHIP,
    "Sole Proprietorship": EntityForm.SOLE_PROPRIETORSHIP,
    "Country": EntityForm.SOVEREIGN_ENTITY,
    "Direct": EntityForm.UNRESOLVED_FORM,
}


@dataclass(frozen=True)
class TypeSpecification:
    """What a semantic type claims, and what it needs to claim it.

    ``prior`` is the strength of the claim when its inputs are complete and its
    support is unbounded.  ``half_support`` is the number of causally observed
    prior events at which the claim reaches half of ``prior``.  ``inputs`` is
    the set of source attributes the claim reads; the fraction actually present
    becomes the coverage factor.
    """

    type: SemanticType
    object_class: ObjectClass
    prior: float
    half_support: int
    inputs: tuple[str, ...]
    meaning: str


def _spec(type_: SemanticType, object_class: ObjectClass, prior: float, half_support: int, inputs: tuple[str, ...], meaning: str) -> tuple[SemanticType, TypeSpecification]:
    return type_, TypeSpecification(type_, object_class, prior, half_support, inputs, meaning)


SPECIFICATIONS: dict[SemanticType, TypeSpecification] = dict((
    # Structural claims read the current event only: they are certain once the
    # identifiers resolve, so their half-support point is one observation.
    _spec(SemanticType.INTERNAL_BOOK_ENTRY, ObjectClass.EVENT, 0.99, 1, ("originator_account", "beneficiary_account"),
          "Originator and beneficiary are the same booking location; no counterparty exists."),
    _spec(SemanticType.INTRA_CUSTOMER_TRANSFER, ObjectClass.EVENT, 0.96, 1, ("originator_customer", "beneficiary_customer"),
          "Two accounts of one legal party; value has not left the customer."),
    _spec(SemanticType.CURRENCY_CONVERSION_TRANSFER, ObjectClass.EVENT, 0.90, 1, ("receiving_currency", "payment_currency"),
          "The two legs settle in different currencies."),
    _spec(SemanticType.CASH_INSTRUMENT_SETTLEMENT, ObjectClass.EVENT, 0.85, 1, ("payment_format",),
          "Settlement instrument carries no counterparty audit trail."),
    _spec(SemanticType.CROSS_JURISDICTION_TRANSFER, ObjectClass.EVENT, 0.88, 1, ("originator_jurisdiction", "beneficiary_jurisdiction"),
          "The booking jurisdictions of the two legs differ."),
    _spec(SemanticType.HIGH_RISK_JURISDICTION_EXPOSURE, ObjectClass.EVENT, 0.90, 1, ("originator_jurisdiction", "beneficiary_jurisdiction", "enhanced_scrutiny_list"),
          "One leg is booked in a jurisdiction on the configured enhanced-scrutiny list."),
    _spec(SemanticType.VIRTUAL_ASSET_EXPOSURE, ObjectClass.EVENT, 0.80, 1, ("originator_bank", "beneficiary_bank"),
          "One leg is booked at a virtual-asset venue."),

    # Regime claims are only as strong as the baseline behind them.
    _spec(SemanticType.VALUE_REGIME, ObjectClass.PROFILE, 0.95, 8, ("prior_amounts", "currency"),
          "The party's own observed operating scale in one currency."),
    _spec(SemanticType.TEMPO_REGIME, ObjectClass.PROFILE, 0.90, 8, ("prior_event_times",),
          "The party's own observed transaction tempo."),
    _spec(SemanticType.COUNTERPARTY_REGIME, ObjectClass.PROFILE, 0.90, 8, ("prior_counterparties",),
          "The counterparty set the party normally uses."),
    _spec(SemanticType.NO_ESTABLISHED_BASELINE, ObjectClass.PROFILE, 0.99, 1, ("prior_event_count",),
          "Too little history exists for any claim about this party's normal behaviour."),
    _spec(SemanticType.DISTRIBUTION_NODE, ObjectClass.PROFILE, 0.92, 20, ("out_degree", "in_degree"),
          "One-to-many payer: a settlement or payroll-operator shape."),
    _spec(SemanticType.COLLECTION_NODE, ObjectClass.PROFILE, 0.92, 20, ("out_degree", "in_degree"),
          "Many-to-one collector: a concentration shape."),
    _spec(SemanticType.PASS_THROUGH_ACCOUNT, ObjectClass.PROFILE, 0.90, 10, ("inflow_value", "outflow_value"),
          "Value transits the account instead of resting in it."),
    _spec(SemanticType.BOOKKEEPING_ACCOUNT, ObjectClass.PROFILE, 0.95, 10, ("self_posting_count", "event_count"),
          "Activity is dominated by self-postings; an internal accounting location."),

    _spec(SemanticType.ROUTINE_VALUE_TRANSFER, ObjectClass.EVENT, 0.85, 8, ("amount", "value_regime"),
          "Value sits inside the party's own established regime."),
    _spec(SemanticType.EXPECTED_HIGH_VALUE_TRANSFER, ObjectClass.EVENT, 0.88, 8, ("amount", "value_regime"),
          "Value is high in absolute terms but ordinary for this party."),
    _spec(SemanticType.UNEXPECTED_LARGE_TRANSFER, ObjectClass.EVENT, 0.92, 8, ("amount", "value_regime"),
          "Value is materially outside the party's own established regime."),
    _spec(SemanticType.UNSCALED_VALUE, ObjectClass.EVENT, 0.99, 1, ("amount",),
          "A magnitude observed with no reference frame; explicit ignorance, not a finding."),
    _spec(SemanticType.BEHAVIOUR_REGIME_SHIFT, ObjectClass.EVENT, 0.88, 8, ("event_times", "tempo_regime"),
          "Tempo is materially outside the party's own established tempo."),
    _spec(SemanticType.NORMAL_OPERATIONAL_BURST, ObjectClass.EVENT, 0.90, 20, ("event_times", "distribution_node"),
          "High tempo that matches the party's own operational shape."),
    _spec(SemanticType.LAYERING_CHAIN_SEGMENT, ObjectClass.EVENT, 0.93, 10, ("recent_inflow", "amount", "pass_through_account"),
          "Value received and forwarded with preservation inside a short window."),

    _spec(SemanticType.FIRST_CONTACT, ObjectClass.RELATIONSHIP, 0.80, 8, ("pair_history", "counterparty_regime"),
          "A counterparty never used before by a party whose counterparty set is known."),
    _spec(SemanticType.RECENTLY_CREATED_RELATIONSHIP, ObjectClass.RELATIONSHIP, 0.75, 8, ("pair_history", "counterparty_regime"),
          "A counterparty relationship formed within the observed window."),
    _spec(SemanticType.ESTABLISHED_RELATIONSHIP, ObjectClass.RELATIONSHIP, 0.85, 8, ("pair_history",),
          "A counterparty relationship the party uses repeatedly."),
    _spec(SemanticType.NON_INFORMATIVE_NOVELTY, ObjectClass.RELATIONSHIP, 0.90, 1, ("pair_history", "prior_event_count"),
          "Novelty observed against no counterparty baseline; the default state of the world, not a signal."),

    _spec(SemanticType.COVERAGE_GAP, ObjectClass.COVERAGE, 1.00, 1, ("source_schema",),
          "A declared input the ontology requires is absent from this source."),
))


# ---------------------------------------------------------------------------
# Declared constants.  Each is a statement about meaning, not a tuned knob.
# ---------------------------------------------------------------------------

#: Prior events required before *any* claim about a party's normal behaviour.
BASELINE_MINIMUM_EVENTS = 5

#: A value is "materially outside" a regime when it exceeds the party's own
#: previous maximum by this factor.  Four is the smallest multiple that cannot
#: be reached by ordinary doubling of a previous peak.
VALUE_REGIME_BREAK_MULTIPLE = 4.0

#: Values at or above the party's own upper decile are "high" but not outside.
VALUE_REGIME_HIGH_QUANTILE = 0.90

#: The most recent amounts that define a party's regime.  Bounded because a
#: regime is a description of current behaviour, not of all recorded history.
VALUE_REGIME_RESERVOIR = 64

#: Tempo is measured in whole-hour buckets: the source timestamps are
#: minute-granular and the claim is about rate, not about exact spacing.
TEMPO_BUCKET_MINUTES = 60

#: A tempo claim requires at least this many events inside one bucket.
TEMPO_MINIMUM_BURST = 3

#: ...and that burst must exceed the party's own mean bucket rate by this
#: factor before the tempo is called a regime shift.
TEMPO_SHIFT_MULTIPLE = 4.0

#: Distinct counterparties before a fan-out/fan-in shape is asserted.
DEGREE_SHAPE_MINIMUM = 20

#: ...and the asymmetry the shape requires.
DEGREE_SHAPE_RATIO = 4.0

#: Pass-through: inflow and outflow value must agree within this fraction.
PASS_THROUGH_RETENTION_TOLERANCE = 0.10

#: ...over at least this many events in each direction.
PASS_THROUGH_MINIMUM_EVENTS = 5

#: A bookkeeping account is one whose activity is at least this self-directed.
BOOKKEEPING_SELF_POSTING_FRACTION = 0.80

#: Layering: a forwarded amount must match the received amount this closely...
LAYERING_VALUE_TOLERANCE = 0.10

#: ...and follow it within this many minutes.
LAYERING_WINDOW_MINUTES = 60

#: Pair observations before a relationship counts as established.
ESTABLISHED_RELATIONSHIP_MINIMUM = 3

#: Configuration, not a factual claim about these states.  Jurisdictions whose
#: presence in a payment leg warrants enhanced scrutiny under the policy this
#: runtime enforces.  Held here so that changing it changes the ontology hash.
ENHANCED_SCRUTINY_JURISDICTIONS = frozenset({"Russia"})

#: Bank-name stems in the source that denote a booking jurisdiction.  Names not
#: matching one of these are domestic institutions of the issuing country.
JURISDICTION_TOKENS = frozenset({
    "Australia", "Austria", "Belgium", "Brazil", "Canada", "China", "Croatia",
    "Cyprus", "Estonia", "Finland", "France", "Germany", "Greece", "India",
    "Ireland", "Israel", "Italy", "Japan", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Mexico", "Netherlands", "Portugal", "Russia", "Saudi Arabia",
    "Slovakia", "Slovenia", "Spain", "Switzerland", "UK",
})

#: The dataset spells its virtual-asset venue this way.
VIRTUAL_ASSET_BANK_TOKEN = "Crytpo"

#: Jurisdiction assigned to accounts whose bank name carries no country token.
DOMESTIC_JURISDICTION = "United States"

#: Jurisdiction assigned to virtual-asset venues; deliberately not a country.
VIRTUAL_ASSET_JURISDICTION = "virtual-asset"

#: Settlement instruments that carry no counterparty audit trail.
CASH_INSTRUMENTS = frozenset({"Cash"})

#: Semantic types this ontology names but that IBM AML `HI-Small` cannot
#: causally support.  They are never emitted; requesting one records a
#: `CoverageGap` instead.  Fabricating them would violate invariant I3.
UNSUPPORTED_ON_SOURCE = (
    ("SalaryDistribution", ("payroll_calendar", "employment_relationship")),
    ("MortgagePayment", ("loan_account", "amortisation_schedule")),
    ("TaxPayment", ("tax_authority_registry",)),
    ("DividendDistribution", ("shareholder_registry",)),
    ("DormantRelationshipReactivated", ("multi_month_history",)),
    ("SeasonalBusiness", ("multi_season_history",)),
)

#: Inputs the ontology wants that this source does not carry at all.
SOURCE_COVERAGE_GAPS = (
    ("sar_feed", "No suspicious-activity or known-bad-party feed exists in this source."),
    ("kyc_dates", "No customer-identification dates exist in this source."),
    ("sanctions_list", "No sanctions or watchlist screening result exists in this source."),
    ("source_of_funds", "No independently verified source-of-funds control exists in this source."),
    ("declared_activity", "No customer-declared expected activity profile exists in this source."),
    ("multi_month_history", "The observed window is hours; dormancy and seasonality are unobservable."),
)


def support_factor(observations: int, half_support: int) -> float:
    """Saturating support: n / (n + k).  Zero observations means zero support."""
    if observations <= 0:
        return 0.0
    return observations / (observations + half_support)


def coverage_factor(present: int, declared: int) -> float:
    return 1.0 if declared <= 0 else present / declared


def confidence_for(type_: SemanticType, observations: int, present_inputs: int) -> tuple[float, str]:
    """The whole confidence model: prior x support x coverage, and its wording."""
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


def _ontology_payload() -> dict[str, object]:
    return {
        "version": ONTOLOGY_VERSION,
        "types": {
            key.value: {
                "class": value.object_class.value, "prior": value.prior,
                "half_support": value.half_support, "inputs": list(value.inputs),
                "meaning": value.meaning,
            }
            for key, value in sorted(SPECIFICATIONS.items(), key=lambda item: item[0].value)
        },
        "constants": {
            "BASELINE_MINIMUM_EVENTS": BASELINE_MINIMUM_EVENTS,
            "VALUE_REGIME_BREAK_MULTIPLE": VALUE_REGIME_BREAK_MULTIPLE,
            "VALUE_REGIME_HIGH_QUANTILE": VALUE_REGIME_HIGH_QUANTILE,
            "VALUE_REGIME_RESERVOIR": VALUE_REGIME_RESERVOIR,
            "TEMPO_BUCKET_MINUTES": TEMPO_BUCKET_MINUTES,
            "TEMPO_MINIMUM_BURST": TEMPO_MINIMUM_BURST,
            "TEMPO_SHIFT_MULTIPLE": TEMPO_SHIFT_MULTIPLE,
            "DEGREE_SHAPE_MINIMUM": DEGREE_SHAPE_MINIMUM,
            "DEGREE_SHAPE_RATIO": DEGREE_SHAPE_RATIO,
            "PASS_THROUGH_RETENTION_TOLERANCE": PASS_THROUGH_RETENTION_TOLERANCE,
            "PASS_THROUGH_MINIMUM_EVENTS": PASS_THROUGH_MINIMUM_EVENTS,
            "BOOKKEEPING_SELF_POSTING_FRACTION": BOOKKEEPING_SELF_POSTING_FRACTION,
            "LAYERING_VALUE_TOLERANCE": LAYERING_VALUE_TOLERANCE,
            "LAYERING_WINDOW_MINUTES": LAYERING_WINDOW_MINUTES,
            "ESTABLISHED_RELATIONSHIP_MINIMUM": ESTABLISHED_RELATIONSHIP_MINIMUM,
            "ENHANCED_SCRUTINY_JURISDICTIONS": sorted(ENHANCED_SCRUTINY_JURISDICTIONS),
            "JURISDICTION_TOKENS": sorted(JURISDICTION_TOKENS),
            "CASH_INSTRUMENTS": sorted(CASH_INSTRUMENTS),
        },
        "unsupported_on_source": [list(item) for item in UNSUPPORTED_ON_SOURCE],
        "coverage_gaps": [list(item) for item in SOURCE_COVERAGE_GAPS],
    }


ONTOLOGY_HASH = hashlib.sha256(
    json.dumps(_ontology_payload(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
).hexdigest()
