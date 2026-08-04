"""The fraud Semantic Context Layer: a single causal fold over the stream.

Same discipline as ``aml_runtime.semantic.context``: ``observe`` reads only
state built from strictly earlier transactions; ``commit`` folds the current
one in afterwards. No object about transaction *i* may read transaction *i*
itself, anything later, or the label column — this module never receives it.

The state kept is genuinely different in shape from AML's, because the domain
is: a *card* spends (one-directional; there is no inbound side to a card the
way an account has inbound payments), and a *device* is the entity that can
exhibit fan-in from many distinct cards. So this layer tracks two registries —
``CardState`` and ``DeviceState`` — rather than one symmetric account registry.
"""

from __future__ import annotations

import hashlib
from collections import deque

from .entities import ResolvedCard, ResolvedDevice, UNRESOLVED_CARD
from .objects import SemanticContextResult, SemanticObject, WithheldObject, object_id
from .ontology import (
    AMOUNT_REGIME_BREAK_MULTIPLE,
    AMOUNT_REGIME_HIGH_QUANTILE,
    AMOUNT_REGIME_RESERVOIR,
    BASELINE_MINIMUM_EVENTS,
    ESTABLISHED_DEVICE_MINIMUM,
    ONTOLOGY_HASH,
    ONTOLOGY_VERSION,
    SHARED_DEVICE_MINIMUM_CARDS,
    SOURCE_COVERAGE_GAPS,
    SPECIFICATIONS,
    TEMPO_BUCKET_MINUTES,
    TEMPO_MINIMUM_BURST,
    TEMPO_SHIFT_MULTIPLE,
    TEST_AMOUNT_ABSOLUTE_CEILING,
    TEST_AMOUNT_MEDIAN_FRACTION,
    UNSUPPORTED_ON_SOURCE,
    SemanticType,
    confidence_for,
)

INFERENCE_RULES_VERSION = "fraud-semantic-inference/1.0"
CONTEXT_VERSION = f"{ONTOLOGY_VERSION}+{INFERENCE_RULES_VERSION}"

COVERAGE_GAP_NAMES = tuple(name for name, _ in SOURCE_COVERAGE_GAPS)
WITHHELD_UNSUPPORTED = tuple(
    WithheldObject(name, inputs, "Input absent from this source; asserting the type would fabricate a jurisdiction, venue, or ledger this release does not carry.")
    for name, inputs in UNSUPPORTED_ON_SOURCE
)

#: Trailing window, in minutes, a device's recent-card membership is tracked
#: over. Two hour-buckets, matching the tempo horizon everywhere else here.
DEVICE_WINDOW_MINUTES = TEMPO_BUCKET_MINUTES * 2


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[int(quantile * (len(sorted_values) - 1))]


class _CardState:
    __slots__ = (
        "purchase_count", "amounts", "first_minute", "last_minute",
        "bucket_id", "bucket_count", "devices", "last_device", "channels", "regions",
        "gap_sum", "gap_count",
    )

    def __init__(self) -> None:
        self.purchase_count = 0
        self.amounts: list[float] = []
        self.first_minute = -1
        self.last_minute = -1
        self.bucket_id = -1
        self.bucket_count = 0
        self.devices: dict[str, int] = {}
        self.last_device = ""
        self.channels: dict[str, int] = {}
        self.regions: dict[str, int] = {}
        self.gap_sum = 0
        self.gap_count = 0

    def burst_in_current_bucket(self, bucket: int) -> int:
        return self.bucket_count if self.bucket_id == bucket else 0

    def mean_bucket_rate(self) -> float:
        if self.purchase_count == 0 or self.first_minute < 0:
            return 0.0
        span = max(1, (self.last_minute - self.first_minute) // TEMPO_BUCKET_MINUTES + 1)
        return self.purchase_count / span

    @property
    def mean_gap_minutes(self) -> float:
        return self.gap_sum / self.gap_count if self.gap_count else 0.0

    @property
    def distinct_devices(self) -> int:
        return len(self.devices)

    @property
    def distinct_channels(self) -> int:
        return len(self.channels)


class _DeviceState:
    __slots__ = ("recent", "first_minute", "purchase_count")

    def __init__(self) -> None:
        self.recent: deque[tuple[int, str]] = deque()
        self.first_minute = -1
        self.purchase_count = 0

    def prune(self, minute: int) -> None:
        cutoff = minute - DEVICE_WINDOW_MINUTES
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()

    def distinct_cards(self, minute: int) -> int:
        self.prune(minute)
        return len({card for _minute, card in self.recent})


class SemanticContextLayer:
    """Turns the transaction stream into fraud semantic objects, causally."""

    def __init__(self) -> None:
        self._cards: dict[str, _CardState] = {}
        self._devices: dict[str, _DeviceState] = {}
        self.events_committed = 0

    def _card(self, card_id: str) -> _CardState:
        state = self._cards.get(card_id)
        if state is None:
            state = _CardState()
            self._cards[card_id] = state
        return state

    def _device(self, device_id: str) -> _DeviceState:
        state = self._devices.get(device_id)
        if state is None:
            state = _DeviceState()
            self._devices[device_id] = state
        return state

    # -- observation ------------------------------------------------------
    def observe(self, transaction, event_index: int) -> SemanticContextResult:
        card = ResolvedCard(transaction.card_id, transaction.metadata.get("network", ""),
                            transaction.metadata.get("card_type", ""), transaction.billing_region,
                            transaction.billing_country) if transaction.card_id else UNRESOLVED_CARD
        state = self._cards.get(transaction.card_id) or _CardState()
        minute = transaction.transaction_dt // 60
        created_at = f"{transaction.timestamp}#{event_index}"
        entity_ids = tuple(item.id for item in card.entities())

        objects: list[SemanticObject] = []

        def emit(type_: SemanticType, subject_id: str, observations: int, present_inputs: int,
                 facts: tuple[str, ...], causal: dict[str, object], relationships: tuple[str, ...] = (),
                 origin: str = "") -> None:
            confidence, explanation = confidence_for(type_, observations, present_inputs)
            objects.append(SemanticObject(
                id=object_id(type_, subject_id, transaction.id, str(observations)),
                type=type_, object_class=SPECIFICATIONS[type_].object_class, subject_id=subject_id,
                confidence=confidence, confidence_explanation=explanation,
                supporting_facts=facts, supporting_entities=entity_ids, supporting_relationships=relationships,
                causal_evidence={"window": "prior-only", **causal}, origin=origin or f"SI-{type_.value}",
                version=CONTEXT_VERSION, created_at=created_at,
            ))

        # 1. Device verification coverage -----------------------------------
        if not transaction.has_identity:
            emit(SemanticType.UNVERIFIED_DEVICE_CONTEXT, transaction.id, 1, 1,
                 ("no row in the identity/device table for this transaction",), {"has_identity": False})

        # 2. Card profile: the reference frame for everything else ----------
        prior_purchases = state.purchase_count
        if prior_purchases < BASELINE_MINIMUM_EVENTS:
            emit(SemanticType.NO_ESTABLISHED_CARD_HISTORY, transaction.card_id, 1, 1,
                 (f"{prior_purchases} prior purchases (< {BASELINE_MINIMUM_EVENTS})",),
                 {"prior_purchases": prior_purchases, "baseline_minimum": BASELINE_MINIMUM_EVENTS})
        else:
            emit(SemanticType.TEMPO_REGIME, transaction.card_id, prior_purchases, 1,
                 (f"mean {state.mean_bucket_rate():.3f} purchases per {TEMPO_BUCKET_MINUTES}-minute bucket",),
                 {"prior_purchases": prior_purchases, "mean_bucket_rate": round(state.mean_bucket_rate(), 6)})
            if state.distinct_devices >= 1:
                emit(SemanticType.DEVICE_REGIME, transaction.card_id, state.distinct_devices, 1,
                     (f"{state.distinct_devices} distinct prior devices",), {"distinct_devices": state.distinct_devices})
            if state.distinct_channels >= 1:
                emit(SemanticType.CHANNEL_REGIME, transaction.card_id, state.distinct_channels, 1,
                     (f"{state.distinct_channels} distinct prior channels",), {"distinct_channels": state.distinct_channels})

        # 3. Amount, situated in the card's own regime ------------------------
        amounts = state.amounts
        if len(amounts) < BASELINE_MINIMUM_EVENTS:
            emit(SemanticType.UNSCALED_SPENDING_AMOUNT, transaction.id, 1, 1,
                 (f"amount {transaction.amount:.2f} USD observed with {len(amounts)} prior events",),
                 {"amount": transaction.amount, "prior_events": len(amounts)})
        else:
            ordered = sorted(amounts)
            prior_max = ordered[-1]
            median = ordered[len(ordered) // 2]
            high_mark = _quantile(ordered, AMOUNT_REGIME_HIGH_QUANTILE)
            emit(SemanticType.AMOUNT_REGIME, transaction.card_id, len(ordered), 1,
                 (f"prior max {prior_max:.2f}, median {median:.2f}, p{int(AMOUNT_REGIME_HIGH_QUANTILE * 100)} {high_mark:.2f} USD",),
                 {"prior_events": len(ordered), "prior_max": prior_max, "median": median, "high_mark": high_mark})
            facts = (f"amount {transaction.amount:.2f} USD", f"own prior max {prior_max:.2f}", f"own median {median:.2f}")
            causal = {"amount": transaction.amount, "prior_max": prior_max, "median": median,
                      "prior_events": len(ordered), "break_multiple": AMOUNT_REGIME_BREAK_MULTIPLE}
            if median > 0 and transaction.amount <= min(TEST_AMOUNT_ABSOLUTE_CEILING, median * TEST_AMOUNT_MEDIAN_FRACTION):
                emit(SemanticType.MINIMAL_TEST_AMOUNT, transaction.id, len(ordered), 1, facts, causal)
            elif transaction.amount > prior_max * AMOUNT_REGIME_BREAK_MULTIPLE:
                emit(SemanticType.UNEXPECTED_SPENDING_AMOUNT, transaction.id, len(ordered), 1, facts, causal)
            elif transaction.amount >= high_mark:
                emit(SemanticType.EXPECTED_HIGH_VALUE_SPEND, transaction.id, len(ordered), 1, facts, causal)
            else:
                emit(SemanticType.ROUTINE_SPENDING_AMOUNT, transaction.id, len(ordered), 1, facts, causal)

        # 4. Tempo, situated in the card's own regime -------------------------
        bucket = minute // TEMPO_BUCKET_MINUTES
        burst = state.burst_in_current_bucket(bucket) + 1
        if burst >= TEMPO_MINIMUM_BURST and prior_purchases >= BASELINE_MINIMUM_EVENTS:
            rate = state.mean_bucket_rate()
            if burst > TEMPO_SHIFT_MULTIPLE * rate:
                emit(SemanticType.TEMPO_REGIME_SHIFT, transaction.card_id, prior_purchases, 1,
                     (f"{burst} purchases in the current {TEMPO_BUCKET_MINUTES}-minute bucket", f"own mean {rate:.3f}"),
                     {"burst": burst, "mean_bucket_rate": round(rate, 6), "shift_multiple": TEMPO_SHIFT_MULTIPLE})

        # 5. Billing region, relative to what this card has shown before ------
        if transaction.billing_region and state.regions and transaction.billing_region not in state.regions:
            emit(SemanticType.UNEXPECTED_BILLING_REGION, transaction.card_id, len(state.regions), 1,
                 (f"region {transaction.billing_region} not among {sorted(state.regions)} previously seen",),
                 {"prior_regions": sorted(state.regions), "new_region": transaction.billing_region})

        # 6. Card-device relationship ------------------------------------------
        relationship_id = f"rel:{transaction.card_id}->{transaction.device_id}" if transaction.device_id else ""
        if transaction.device_id:
            pair_count = state.devices.get(transaction.device_id, 0)
            facts = (f"{pair_count} prior purchases from this device on this card",)
            causal = {"pair_prior_count": pair_count, "prior_distinct_devices": state.distinct_devices}
            if pair_count >= ESTABLISHED_DEVICE_MINIMUM:
                emit(SemanticType.ESTABLISHED_DEVICE_RELATIONSHIP, relationship_id, pair_count, 1, facts, causal, (relationship_id,))
            elif pair_count >= 1:
                emit(SemanticType.RECENTLY_LINKED_DEVICE, relationship_id, pair_count, 1, facts, causal, (relationship_id,))
            elif state.distinct_devices >= BASELINE_MINIMUM_EVENTS:
                emit(SemanticType.FIRST_DEVICE_CONTACT, relationship_id, state.distinct_devices, 1, facts, causal, (relationship_id,))
            else:
                emit(SemanticType.NON_INFORMATIVE_DEVICE_NOVELTY, relationship_id, 1, 1,
                     facts + (f"this card has only {state.distinct_devices} known devices, below the {BASELINE_MINIMUM_EVENTS} needed for a device baseline",),
                     causal, (relationship_id,))

            device_state = self._devices.get(transaction.device_id)
            distinct_cards = device_state.distinct_cards(minute) if device_state else 0
            if distinct_cards >= SHARED_DEVICE_MINIMUM_CARDS:
                emit(SemanticType.SHARED_CLIENT_SIGNATURE_EXPOSURE, transaction.id, distinct_cards, 1,
                     (f"{distinct_cards} distinct card identities used this client signature within the trailing {DEVICE_WINDOW_MINUTES}-minute window",),
                     {"distinct_cards": distinct_cards, "window_minutes": DEVICE_WINDOW_MINUTES}, (relationship_id,))

        withheld = WITHHELD_UNSUPPORTED
        entities = card.entities() + ((ResolvedDevice(transaction.device_id, transaction.metadata.get("device_type", "")).entity(),)
                                       if transaction.device_id else ())
        return SemanticContextResult(
            transaction_id=transaction.id,
            objects=tuple(sorted(objects, key=lambda item: (item.type.value, item.id))),
            withheld=withheld, coverage_gaps=COVERAGE_GAP_NAMES, entities=entities,
            context_state_hash=self._state_hash(transaction, state, minute),
        )

    @staticmethod
    def _state_hash(transaction, state: _CardState, minute: int) -> str:
        payload = "|".join(str(part) for part in (
            ONTOLOGY_HASH[:16], transaction.card_id, state.purchase_count, len(state.amounts),
            state.distinct_devices, state.distinct_channels, len(state.regions), state.bucket_id, state.bucket_count,
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # -- causal commit -------------------------------------------------------
    def commit(self, transaction) -> None:
        minute = transaction.transaction_dt // 60
        bucket = minute // TEMPO_BUCKET_MINUTES
        card = self._card(transaction.card_id)
        if card.first_minute < 0:
            card.first_minute = minute
        if card.last_minute >= 0:
            card.gap_sum += minute - card.last_minute
            card.gap_count += 1
        card.last_minute = minute
        card.purchase_count += 1
        if card.bucket_id == bucket:
            card.bucket_count += 1
        else:
            card.bucket_id = bucket
            card.bucket_count = 1
        card.amounts.append(transaction.amount)
        if len(card.amounts) > AMOUNT_REGIME_RESERVOIR:
            del card.amounts[0]
        card.regions[transaction.billing_region] = card.regions.get(transaction.billing_region, 0) + 1
        card.channels[transaction.product_channel] = card.channels.get(transaction.product_channel, 0) + 1

        if transaction.device_id:
            card.devices[transaction.device_id] = card.devices.get(transaction.device_id, 0) + 1
            card.last_device = transaction.device_id
            device = self._device(transaction.device_id)
            if device.first_minute < 0:
                device.first_minute = minute
            device.purchase_count += 1
            device.recent.append((minute, transaction.card_id))
            device.prune(minute)
        self.events_committed += 1
