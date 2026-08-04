"""The fraud temporal engine: multi-horizon state, folded causally.

Same tumbling-bucket-plus-predecessor discipline as
``aml_runtime/behaviour/temporal.py``: each horizon is read as
``count[current bucket] + count[previous bucket]`` when they are adjacent,
giving a trailing window of one to two bucket widths at constant memory per
entity.

This engine holds no transaction field it does not need — everything it reads
about "was this a new device", "was the amount inside regime", "was the
billing region unexpected" arrives as a *semantic object* from the Semantic
Context Layer, exactly as in the AML port. It is deliberately a second,
self-contained fold rather than reaching into the semantic layer's private
state, matching how ``aml_runtime``'s two layers stay decoupled.

Two registries, not one, because the domain is genuinely bipartite: a card
spends (one direction only — it never receives), and a device is what can
exhibit fan-in from many distinct cards. AML's single symmetric account
registry has no clean analogue here; forcing one would be exactly the kind of
redesign this port was told not to do.
"""

from __future__ import annotations

from .ontology import HORIZON_MINUTES, Horizon

MINUTES_WIDTH = HORIZON_MINUTES[Horizon.MINUTES]
HOURS_WIDTH = HORIZON_MINUTES[Horizon.HOURS]


class CardState:
    """Constant-memory behavioural state for one card."""

    __slots__ = (
        "purchase_count", "first_minute", "last_minute", "last_purchase_minute",
        "gap_sum", "gap_count", "hours_active",
        "devices", "last_device", "distinct_devices_seen",
        "m_bucket", "m_out", "m_out_prev", "m_new_device", "m_new_device_prev", "m_test", "m_test_prev",
        "h_bucket", "h_out", "h_out_prev", "h_new_device", "stable_events", "break_events",
        "distances", "last_distance_minute",
        "role", "role_since", "role_prev", "transitions", "stages",
    )

    def __init__(self) -> None:
        self.purchase_count = 0
        self.first_minute = -1
        self.last_minute = -1
        self.last_purchase_minute = -1
        self.gap_sum = 0
        self.gap_count = 0
        self.hours_active = 0
        self.devices: dict[str, int] = {}
        self.last_device = ""
        self.distinct_devices_seen = 0
        self.m_bucket = -1
        self.m_out = 0
        self.m_out_prev = 0
        self.m_new_device = 0
        self.m_new_device_prev = 0
        self.m_test = 0
        self.m_test_prev = 0
        self.h_bucket = -1
        self.h_out = 0
        self.h_out_prev = 0
        self.h_new_device = 0
        self.stable_events = 0
        self.break_events = 0
        self.distances: list[float] = []
        self.last_distance_minute = -1
        self.role = ""
        self.role_since = -1
        self.role_prev = ""
        self.transitions = 0
        self.stages: tuple[tuple[int, int], ...] | None = None

    def roll(self, minute: int) -> None:
        bucket = minute // MINUTES_WIDTH
        if bucket != self.m_bucket:
            adjacent = bucket == self.m_bucket + 1
            self.m_out_prev = self.m_out if adjacent else 0
            self.m_new_device_prev = self.m_new_device if adjacent else 0
            self.m_test_prev = self.m_test if adjacent else 0
            self.m_bucket = bucket
            self.m_out = self.m_new_device = self.m_test = 0
        bucket = minute // HOURS_WIDTH
        if bucket != self.h_bucket:
            adjacent = bucket == self.h_bucket + 1
            self.h_out_prev = self.h_out if adjacent else 0
            if self.h_bucket >= 0:
                self.hours_active += 1
            self.h_bucket = bucket
            self.h_out = self.h_new_device = 0

    def out_in_minutes(self) -> int:
        return self.m_out + self.m_out_prev

    def new_devices_in_minutes(self) -> int:
        return self.m_new_device + self.m_new_device_prev

    def test_charges_in_minutes(self) -> int:
        return self.m_test + self.m_test_prev

    def out_in_hours(self) -> int:
        return self.h_out + self.h_out_prev

    @property
    def mean_bucket_rate(self) -> float:
        if self.purchase_count == 0 or self.first_minute < 0:
            return 0.0
        span = max(1, (self.last_minute - self.first_minute) // MINUTES_WIDTH + 1)
        return self.purchase_count / span

    @property
    def mean_gap_minutes(self) -> float:
        return self.gap_sum / self.gap_count if self.gap_count else 0.0

    @property
    def established_device_max_count(self) -> int:
        return max(self.devices.values(), default=0)


class DeviceState:
    """Constant-memory fan-out state for one device."""

    __slots__ = ("cards", "m_bucket", "m_cards", "m_cards_prev", "h_bucket", "h_cards", "hours_active")

    def __init__(self) -> None:
        self.cards: dict[str, int] = {}
        self.m_bucket = -1
        self.m_cards: set[str] = set()
        self.m_cards_prev: set[str] = set()
        self.h_bucket = -1
        self.h_cards: set[str] = set()
        self.hours_active = 0

    def roll(self, minute: int) -> None:
        bucket = minute // MINUTES_WIDTH
        if bucket != self.m_bucket:
            self.m_cards_prev = self.m_cards if bucket == self.m_bucket + 1 else set()
            self.m_bucket = bucket
            self.m_cards = set()
        bucket = minute // HOURS_WIDTH
        if bucket != self.h_bucket:
            if self.h_bucket >= 0:
                self.hours_active += 1
            self.h_bucket = bucket
            self.h_cards = set()

    def distinct_cards_in_minutes(self) -> int:
        return len(self.m_cards | self.m_cards_prev)

    def distinct_cards_in_hours(self) -> int:
        return len(self.h_cards)


class TemporalEngine:
    def __init__(self) -> None:
        self._cards: dict[str, CardState] = {}
        self._devices: dict[str, DeviceState] = {}
        self.events_committed = 0

    def state(self, card_id: str) -> CardState:
        state = self._cards.get(card_id)
        if state is None:
            state = CardState()
            self._cards[card_id] = state
        return state

    def peek(self, card_id: str) -> CardState | None:
        return self._cards.get(card_id)

    def device_state(self, device_id: str) -> DeviceState | None:
        return self._devices.get(device_id)

    def _device(self, device_id: str) -> DeviceState:
        state = self._devices.get(device_id)
        if state is None:
            state = DeviceState()
            self._devices[device_id] = state
        return state

    def commit(self, card_id: str, device_id: str, minute: int, amount: float, distance1: float | None, *,
               new_device: bool, is_test_amount: bool, regime_stable: bool, regime_broken: bool) -> None:
        card = self.state(card_id)
        card.roll(minute)
        if card.first_minute < 0:
            card.first_minute = minute
        if card.last_minute >= 0:
            card.gap_sum += minute - card.last_minute
            card.gap_count += 1
        card.last_minute = minute
        card.last_purchase_minute = minute
        card.purchase_count += 1
        card.m_out += 1
        card.h_out += 1
        if is_test_amount:
            card.m_test += 1
        if regime_stable:
            card.stable_events += 1
        if regime_broken:
            card.break_events += 1
        if distance1 is not None:
            card.distances.append(distance1)
            if len(card.distances) > 32:
                del card.distances[0]
            card.last_distance_minute = minute

        if device_id:
            if new_device:
                card.distinct_devices_seen += 1
                card.m_new_device += 1
                card.h_new_device += 1
            card.devices[device_id] = card.devices.get(device_id, 0) + 1
            card.last_device = device_id

            device = self._device(device_id)
            device.roll(minute)
            if device_id not in device.cards:
                device.cards[device_id] = 0
            device.cards[card_id] = device.cards.get(card_id, 0) + 1
            device.m_cards.add(card_id)
            device.h_cards.add(card_id)
        self.events_committed += 1
