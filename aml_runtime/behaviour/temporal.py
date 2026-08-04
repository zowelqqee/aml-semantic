"""The temporal semantic engine: multi-horizon state, folded causally.

Every horizon is kept as a *tumbling bucket plus its immediate predecessor*.
Reading `current + previous` gives a trailing window between one and two bucket
widths — exact, deterministic, and constant-memory per account, which matters
when the stream carries hundreds of thousands of accounts.

The engine holds no transaction fields it does not need.  Relationship novelty,
value regime, tempo regime, jurisdiction and instrument all arrive as *semantic
objects* from the Semantic Context Layer; this engine only accumulates them over
time.  That is the whole point of the layer: behaviour is an aggregate over
meaning, not over rows.
"""

from __future__ import annotations

import math

from .ontology import HORIZON_MINUTES, Horizon

MINUTES_WIDTH = HORIZON_MINUTES[Horizon.MINUTES]
HOURS_WIDTH = HORIZON_MINUTES[Horizon.HOURS]
DAYS_WIDTH = HORIZON_MINUTES[Horizon.DAYS]
WEEKS_WIDTH = HORIZON_MINUTES[Horizon.WEEKS]


class AccountBehaviourState:
    """Constant-memory behavioural state for one account."""

    __slots__ = (
        # lifetime
        "out_count", "in_count", "self_count", "out_value", "in_value",
        "distinct_out", "distinct_in", "established_out", "cash_in_count", "cash_in_value",
        "cross_juris_out", "forward_events", "forward_delay_sum", "forward_value",
        "first_minute", "last_minute", "last_out_minute", "last_in_minute",
        "gap_sum", "gap_count", "stable_events", "break_events", "hours_active",
        "peak_distinct_out",
        # minutes horizon
        "m_bucket", "m_out", "m_out_prev", "m_in", "m_in_prev",
        "m_new_out", "m_new_out_prev", "m_new_in", "m_new_in_prev", "m_fwd", "m_fwd_prev",
        # hours horizon
        "h_bucket", "h_out", "h_out_prev", "h_in", "h_in_prev",
        "h_new_out", "h_new_out_prev", "h_new_in", "h_fwd", "h_fwd_prev",
        "h_out_sum", "h_out_sumsq", "h_in_value", "h_out_value", "h_cash_in",
        # day and week horizons (declared; unfillable on a 5.6-hour window)
        "d_bucket", "d_out", "d_in", "w_bucket", "w_out", "w_in",
        # inflow provenance, bounded to two hops
        "in_src", "in_origin", "in_minute", "in_amount", "in_currency", "in_src_prev",
        # role and scenario
        "role", "role_since", "role_prev", "transitions", "stages",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)
        self.out_value = 0.0
        self.in_value = 0.0
        self.cash_in_value = 0.0
        self.forward_value = 0.0
        self.h_out_sum = 0.0
        self.h_out_sumsq = 0.0
        self.h_in_value = 0.0
        self.h_out_value = 0.0
        self.in_amount = 0.0
        self.first_minute = -1
        self.last_minute = -1
        self.last_out_minute = -1
        self.last_in_minute = -1
        self.m_bucket = -1
        self.h_bucket = -1
        self.d_bucket = -1
        self.w_bucket = -1
        self.in_minute = -1
        self.in_src = ""
        self.in_origin = ""
        self.in_src_prev = ""
        self.in_currency = ""
        self.role = ""
        self.role_prev = ""
        self.role_since = -1
        self.stages = None

    # -- horizon rolling ---------------------------------------------------
    def roll(self, minute: int) -> None:
        """Advance every horizon to ``minute`` before reading or writing it."""
        bucket = minute // MINUTES_WIDTH
        if bucket != self.m_bucket:
            self.m_out_prev = self.m_out if bucket == self.m_bucket + 1 else 0
            self.m_in_prev = self.m_in if bucket == self.m_bucket + 1 else 0
            self.m_new_out_prev = self.m_new_out if bucket == self.m_bucket + 1 else 0
            self.m_new_in_prev = self.m_new_in if bucket == self.m_bucket + 1 else 0
            self.m_fwd_prev = self.m_fwd if bucket == self.m_bucket + 1 else 0
            self.m_bucket, self.m_out, self.m_in = bucket, 0, 0
            self.m_new_out = self.m_new_in = self.m_fwd = 0

        bucket = minute // HOURS_WIDTH
        if bucket != self.h_bucket:
            adjacent = bucket == self.h_bucket + 1
            self.h_out_prev = self.h_out if adjacent else 0
            self.h_in_prev = self.h_in if adjacent else 0
            self.h_new_out_prev = self.h_new_out if adjacent else 0
            self.h_fwd_prev = self.h_fwd if adjacent else 0
            if self.h_bucket >= 0:
                self.hours_active += 1
            self.h_bucket = bucket
            self.h_out = self.h_in = self.h_new_out = self.h_new_in = self.h_fwd = self.h_cash_in = 0
            self.h_out_sum = self.h_out_sumsq = self.h_in_value = self.h_out_value = 0.0

        bucket = minute // DAYS_WIDTH
        if bucket != self.d_bucket:
            self.d_bucket, self.d_out, self.d_in = bucket, 0, 0
        bucket = minute // WEEKS_WIDTH
        if bucket != self.w_bucket:
            self.w_bucket, self.w_out, self.w_in = bucket, 0, 0

    # -- horizon reads ------------------------------------------------------
    def out_in(self, horizon: Horizon) -> int:
        if horizon is Horizon.MINUTES:
            return self.m_out + self.m_out_prev
        if horizon is Horizon.HOURS:
            return self.h_out + self.h_out_prev
        return self.d_out if horizon is Horizon.DAYS else self.w_out

    def in_in(self, horizon: Horizon) -> int:
        if horizon is Horizon.MINUTES:
            return self.m_in + self.m_in_prev
        if horizon is Horizon.HOURS:
            return self.h_in + self.h_in_prev
        return self.d_in if horizon is Horizon.DAYS else self.w_in

    def new_out_in(self, horizon: Horizon) -> int:
        if horizon is Horizon.MINUTES:
            return self.m_new_out + self.m_new_out_prev
        return self.h_new_out + self.h_new_out_prev

    def new_in_in(self, horizon: Horizon) -> int:
        return self.m_new_in + self.m_new_in_prev if horizon is Horizon.MINUTES else self.h_new_in

    def forwards_in(self, horizon: Horizon) -> int:
        if horizon is Horizon.MINUTES:
            return self.m_fwd + self.m_fwd_prev
        return self.h_fwd + self.h_fwd_prev

    @property
    def events(self) -> int:
        return self.out_count + self.in_count

    @property
    def mean_gap_minutes(self) -> float:
        return self.gap_sum / self.gap_count if self.gap_count else 0.0

    @property
    def mean_bucket_rate(self) -> float:
        """Outbound events per minutes-bucket over the account's own lifetime."""
        if self.out_count == 0 or self.first_minute < 0:
            return 0.0
        span = max(1, (self.last_minute - self.first_minute) // MINUTES_WIDTH + 1)
        return self.out_count / span

    @property
    def retention(self) -> float:
        """Share of inflow value not paid out again.  1.0 means nothing left."""
        if self.in_value <= 0.0:
            return 1.0
        return max(0.0, min(1.0, (self.in_value - self.out_value) / self.in_value))

    @property
    def outbound_dispersion(self) -> float:
        """Coefficient of variation of this bucket's outbound amounts."""
        if self.h_out < 2 or self.h_out_sum <= 0.0:
            return math.inf
        mean = self.h_out_sum / self.h_out
        variance = max(0.0, self.h_out_sumsq / self.h_out - mean * mean)
        return math.sqrt(variance) / mean if mean > 0 else math.inf

    @property
    def cash_inflow_share(self) -> float:
        return self.cash_in_count / self.in_count if self.in_count else 0.0

    def horizons_filled(self) -> tuple[str, ...]:
        span = max(0, self.last_minute - self.first_minute) if self.first_minute >= 0 else 0
        return tuple(
            horizon.value for horizon, width in HORIZON_MINUTES.items() if span >= width
        )


class TemporalEngine:
    """The account registry.  One fold, one direction, no lookahead."""

    def __init__(self) -> None:
        self._states: dict[str, AccountBehaviourState] = {}
        self.events_committed = 0

    def __len__(self) -> int:
        return len(self._states)

    def state(self, account_id: str) -> AccountBehaviourState:
        found = self._states.get(account_id)
        if found is None:
            found = AccountBehaviourState()
            self._states[account_id] = found
        return found

    def peek(self, account_id: str) -> AccountBehaviourState | None:
        return self._states.get(account_id)

    # -- causal commit ------------------------------------------------------
    def commit(
        self,
        originator_id: str,
        beneficiary_id: str,
        minute: int,
        amount: float,
        currency: str,
        *,
        self_posting: bool,
        new_counterparty: bool,
        established: bool,
        cash_instrument: bool,
        cross_jurisdiction: bool,
        regime_stable: bool,
        regime_broken: bool,
    ) -> None:
        """Fold one event into behavioural state, after it has been observed.

        Every keyword flag is a semantic object's presence, not a raw column.
        """
        source = self.state(originator_id)
        source.roll(minute)
        if source.first_minute < 0:
            source.first_minute = minute
        if source.last_out_minute >= 0:
            source.gap_sum += minute - source.last_out_minute
            source.gap_count += 1
        source.last_out_minute = minute
        source.last_minute = minute
        source.out_count += 1
        source.out_value += amount
        source.m_out += 1
        source.h_out += 1
        source.d_out += 1
        source.w_out += 1
        source.h_out_sum += amount
        source.h_out_sumsq += amount * amount
        source.h_out_value += amount
        if new_counterparty:
            source.distinct_out += 1
            source.m_new_out += 1
            source.h_new_out += 1
            source.peak_distinct_out = max(source.peak_distinct_out, source.distinct_out)
        if established:
            source.established_out += 1
        if cross_jurisdiction:
            source.cross_juris_out += 1
        if regime_stable:
            source.stable_events += 1
        if regime_broken:
            source.break_events += 1

        # A prompt, value-preserving forward is the atom every transit and
        # layering behaviour is built from.  A self-posting is never a forward:
        # the value has not moved to anyone, and matching a self-posting against
        # the identical self-posting before it would manufacture a whole chain.
        if (
            not self_posting
            and source.in_minute >= 0
            and source.in_currency == currency
            and source.in_amount > 0.0
            and 0 <= minute - source.in_minute <= _PROMPT_FORWARD_MINUTES
            and abs(amount - source.in_amount) <= _FORWARD_VALUE_TOLERANCE * source.in_amount
        ):
            source.forward_events += 1
            source.forward_delay_sum += minute - source.in_minute
            source.forward_value += amount
            source.m_fwd += 1
            source.h_fwd += 1

        if self_posting:
            # The posting is real ledger activity, so it counts; but it does not
            # update the inflow provenance used for forward and cycle detection,
            # because no counterparty funded it.
            source.self_count += 1
            source.in_count += 1
            source.in_value += amount
            source.m_in += 1
            source.h_in += 1
            self.events_committed += 1
            return

        target = self.state(beneficiary_id)
        target.roll(minute)
        if target.first_minute < 0:
            target.first_minute = minute
        target.last_in_minute = minute
        target.last_minute = minute
        target.in_count += 1
        target.in_value += amount
        target.m_in += 1
        target.h_in += 1
        target.d_in += 1
        target.w_in += 1
        target.h_in_value += amount
        if new_counterparty:
            target.distinct_in += 1
            target.m_new_in += 1
            target.h_new_in += 1
        if cash_instrument:
            target.cash_in_count += 1
            target.cash_in_value += amount
            target.h_cash_in += 1
        # Bounded provenance: the beneficiary remembers who paid it, and who
        # paid that party.  Two hops is what one scalar can carry exactly.
        target.in_src_prev = target.in_src
        target.in_src = originator_id
        target.in_origin = source.in_src
        target.in_minute = minute
        target.in_amount = amount
        target.in_currency = currency
        self.events_committed += 1


# Imported late to keep the ontology the single source of these values while
# avoiding a per-event attribute lookup in the hot commit path.
from .ontology import FORWARD_VALUE_TOLERANCE as _FORWARD_VALUE_TOLERANCE  # noqa: E402
from .ontology import PROMPT_FORWARD_MINUTES as _PROMPT_FORWARD_MINUTES  # noqa: E402
