"""The Semantic Context Layer: a single causal fold over the transaction stream.

Ordering is the whole guarantee.  ``observe`` reads only state accumulated from
strictly earlier events; ``commit`` folds the current event in afterwards.  No
inference rule can see the transaction it is describing, any later transaction,
or any label.  The layer is constructed without a reference to the label column
at all, so the guarantee is structural rather than conventional.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable

from ..models import Transaction
from .entities import EntityResolver, ResolvedAccount
from .objects import SemanticContextResult, SemanticEntity, SemanticObject, WithheldObject, object_id
from .ontology import (
    BASELINE_MINIMUM_EVENTS,
    BOOKKEEPING_SELF_POSTING_FRACTION,
    CASH_INSTRUMENTS,
    DEGREE_SHAPE_MINIMUM,
    DEGREE_SHAPE_RATIO,
    ESTABLISHED_RELATIONSHIP_MINIMUM,
    LAYERING_VALUE_TOLERANCE,
    LAYERING_WINDOW_MINUTES,
    ONTOLOGY_HASH,
    ONTOLOGY_VERSION,
    PASS_THROUGH_MINIMUM_EVENTS,
    PASS_THROUGH_RETENTION_TOLERANCE,
    SOURCE_COVERAGE_GAPS,
    SPECIFICATIONS,
    TEMPO_BUCKET_MINUTES,
    TEMPO_MINIMUM_BURST,
    TEMPO_SHIFT_MULTIPLE,
    UNSUPPORTED_ON_SOURCE,
    VALUE_REGIME_BREAK_MULTIPLE,
    VALUE_REGIME_HIGH_QUANTILE,
    VALUE_REGIME_RESERVOIR,
    ObjectClass,
    SemanticType,
    confidence_for,
)

INFERENCE_RULES_VERSION = "aml-semantic-inference/1.0"
CONTEXT_VERSION = f"{ONTOLOGY_VERSION}+{INFERENCE_RULES_VERSION}"

COVERAGE_GAP_NAMES = tuple(name for name, _ in SOURCE_COVERAGE_GAPS)
WITHHELD_UNSUPPORTED = tuple(
    WithheldObject(name, inputs, "Input absent from this source; asserting the type would fabricate meaning.")
    for name, inputs in UNSUPPORTED_ON_SOURCE
)


def _minute(timestamp: str) -> int:
    """Minute-resolution logical clock.  The source is minute-granular."""
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        parsed = datetime.strptime(timestamp, "%Y/%m/%d %H:%M")
    return int(parsed.timestamp()) // 60


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    position = int(quantile * (len(sorted_values) - 1))
    return sorted_values[position]


class _AccountState:
    """Compact per-account causal state.

    Deliberately scalar.  Holding per-account collections for a stream with
    hundreds of thousands of accounts costs more memory than the claims are
    worth, and every claim below needs only counts, sums, and one recent inflow.
    """

    __slots__ = (
        "out_count", "in_count", "self_count", "out_distinct", "in_distinct",
        "first_minute", "last_minute", "bucket_id", "bucket_count",
        "last_in_minute", "last_in_amount", "last_in_currency",
        "out_value", "in_value", "currency", "mixed_currency",
    )

    def __init__(self) -> None:
        self.out_count = 0
        self.in_count = 0
        self.self_count = 0
        self.out_distinct = 0
        self.in_distinct = 0
        self.first_minute = -1
        self.last_minute = -1
        self.bucket_id = -1
        self.bucket_count = 0
        self.last_in_minute = -1
        self.last_in_amount = 0.0
        self.last_in_currency = ""
        self.out_value = 0.0
        self.in_value = 0.0
        self.currency = ""
        self.mixed_currency = False

    @property
    def event_count(self) -> int:
        return self.out_count + self.in_count

    def burst_in(self, bucket: int) -> int:
        return self.bucket_count if self.bucket_id == bucket else 0

    def mean_bucket_rate(self) -> float:
        if self.out_count == 0 or self.first_minute < 0:
            return 0.0
        span = max(1, (self.last_minute - self.first_minute) // TEMPO_BUCKET_MINUTES + 1)
        return self.out_count / span


class SemanticContextLayer:
    """Turns a transaction stream into semantic objects, one causal step at a time."""

    def __init__(self, resolver: EntityResolver) -> None:
        self.resolver = resolver
        self._state: dict[str, _AccountState] = {}
        self._regime: dict[tuple[str, str], list[float]] = {}
        self._pair: dict[tuple[str, str], int] = {}
        self.events_committed = 0

    # -- state access ----------------------------------------------------
    def _get(self, account_id: str) -> _AccountState:
        state = self._state.get(account_id)
        if state is None:
            state = _AccountState()
            self._state[account_id] = state
        return state

    def _amounts(self, account_id: str, currency: str) -> list[float]:
        return self._regime.get((account_id, currency), [])

    # -- profile predicates ----------------------------------------------
    @staticmethod
    def _is_distribution_node(state: _AccountState) -> bool:
        return (
            state.out_distinct >= DEGREE_SHAPE_MINIMUM
            and state.out_distinct >= DEGREE_SHAPE_RATIO * max(1, state.in_distinct)
        )

    @staticmethod
    def _is_collection_node(state: _AccountState) -> bool:
        return (
            state.in_distinct >= DEGREE_SHAPE_MINIMUM
            and state.in_distinct >= DEGREE_SHAPE_RATIO * max(1, state.out_distinct)
        )

    @staticmethod
    def _is_pass_through(state: _AccountState) -> bool:
        if state.mixed_currency or state.in_value <= 0.0:
            return False
        if state.in_count < PASS_THROUGH_MINIMUM_EVENTS or state.out_count < PASS_THROUGH_MINIMUM_EVENTS:
            return False
        return abs(state.in_value - state.out_value) / state.in_value <= PASS_THROUGH_RETENTION_TOLERANCE

    @staticmethod
    def _is_bookkeeping(state: _AccountState) -> bool:
        total = state.event_count
        if total < PASS_THROUGH_MINIMUM_EVENTS:
            return False
        return state.self_count / total >= BOOKKEEPING_SELF_POSTING_FRACTION

    # -- observation ------------------------------------------------------
    def observe(self, transaction: Transaction, event_index: int) -> SemanticContextResult:
        """Read the meaning of ``transaction`` from strictly-prior state."""
        originator = self.resolver.resolve(transaction.originator_account_id)
        beneficiary = self.resolver.resolve(transaction.beneficiary_account_id)
        state = self._get(transaction.originator_account_id)
        minute = _minute(transaction.timestamp)
        created_at = f"{transaction.timestamp}#{event_index}"
        currency = transaction.currency
        amounts = self._amounts(transaction.originator_account_id, currency)
        pair_count = self._pair.get((transaction.originator_account_id, transaction.beneficiary_account_id), 0)
        self_posting = transaction.originator_account_id == transaction.beneficiary_account_id
        relationship_id = f"rel:{transaction.originator_account_id}->{transaction.beneficiary_account_id}"

        objects: list[SemanticObject] = []
        entity_ids = tuple(item.id for item in originator.entities()) + tuple(item.id for item in beneficiary.entities())

        def emit(
            type_: SemanticType,
            subject_id: str,
            observations: int,
            present_inputs: int,
            facts: tuple[str, ...],
            causal: dict[str, object],
            relationships: tuple[str, ...] = (),
            origin: str = "",
        ) -> None:
            confidence, explanation = confidence_for(type_, observations, present_inputs)
            objects.append(SemanticObject(
                id=object_id(type_, subject_id, transaction.id, str(observations)),
                type=type_,
                object_class=SPECIFICATIONS[type_].object_class,
                subject_id=subject_id,
                confidence=confidence,
                confidence_explanation=explanation,
                supporting_facts=facts,
                supporting_entities=entity_ids,
                supporting_relationships=relationships,
                causal_evidence={"window": "prior-only", **causal},
                origin=origin or f"SI-{type_.value}",
                version=CONTEXT_VERSION,
                created_at=created_at,
            ))

        # 1. Structural identity of the event ------------------------------
        if self_posting:
            emit(SemanticType.INTERNAL_BOOK_ENTRY, transaction.originator_account_id, 1, 2,
                 (f"originator == beneficiary == {transaction.originator_account_id}",),
                 {"self_posting": True, "payment_format": transaction.payment_type})
        elif originator.customer_id and beneficiary.customer_id and originator.customer_id == beneficiary.customer_id:
            emit(SemanticType.INTRA_CUSTOMER_TRANSFER, f"cust:{originator.customer_id}", 1, 2,
                 (f"both accounts belong to customer {originator.customer_id}",),
                 {"customer_id": originator.customer_id, "originator_form": originator.form.value})

        payment_currency = transaction.metadata.get("payment_currency", currency)
        if payment_currency and payment_currency != currency:
            emit(SemanticType.CURRENCY_CONVERSION_TRANSFER, transaction.id, 1, 2,
                 (f"receiving {currency} vs payment {payment_currency}",),
                 {"receiving_currency": currency, "payment_currency": payment_currency})
        if transaction.payment_type in CASH_INSTRUMENTS:
            emit(SemanticType.CASH_INSTRUMENT_SETTLEMENT, transaction.id, 1, 1,
                 (f"payment format {transaction.payment_type}",), {"payment_format": transaction.payment_type})

        resolved_legs = self.resolver.resolved(transaction.originator_account_id) and self.resolver.resolved(transaction.beneficiary_account_id)
        if resolved_legs and not self_posting and originator.jurisdiction != beneficiary.jurisdiction:
            emit(SemanticType.CROSS_JURISDICTION_TRANSFER, transaction.id, 1, 2,
                 (f"{originator.jurisdiction} -> {beneficiary.jurisdiction}",),
                 {"originator_jurisdiction": originator.jurisdiction, "beneficiary_jurisdiction": beneficiary.jurisdiction})
        if resolved_legs and (originator.enhanced_scrutiny or beneficiary.enhanced_scrutiny):
            exposed = originator.jurisdiction if originator.enhanced_scrutiny else beneficiary.jurisdiction
            emit(SemanticType.HIGH_RISK_JURISDICTION_EXPOSURE, transaction.id, 1, 3,
                 (f"leg booked in {exposed}",),
                 {"exposed_jurisdiction": exposed, "list": "ENHANCED_SCRUTINY_JURISDICTIONS"})
        if resolved_legs and (originator.virtual_asset_venue or beneficiary.virtual_asset_venue):
            emit(SemanticType.VIRTUAL_ASSET_EXPOSURE, transaction.id, 1, 2,
                 ("one leg is booked at a virtual-asset venue",),
                 {"originator_bank": originator.bank_id, "beneficiary_bank": beneficiary.bank_id})

        # 2. Profiles: the reference frames -------------------------------
        prior_events = state.out_count
        if prior_events < BASELINE_MINIMUM_EVENTS:
            emit(SemanticType.NO_ESTABLISHED_BASELINE, transaction.originator_account_id, 1, 1,
                 (f"{prior_events} prior outbound events (< {BASELINE_MINIMUM_EVENTS})",),
                 {"prior_outbound_events": prior_events, "baseline_minimum": BASELINE_MINIMUM_EVENTS})
        else:
            emit(SemanticType.TEMPO_REGIME, transaction.originator_account_id, prior_events, 1,
                 (f"mean {state.mean_bucket_rate():.3f} outbound events per {TEMPO_BUCKET_MINUTES}-minute bucket",),
                 {"prior_outbound_events": prior_events, "mean_bucket_rate": round(state.mean_bucket_rate(), 6)})
            if state.out_distinct >= BASELINE_MINIMUM_EVENTS:
                emit(SemanticType.COUNTERPARTY_REGIME, transaction.originator_account_id, state.out_distinct, 1,
                     (f"{state.out_distinct} distinct prior counterparties",),
                     {"prior_distinct_counterparties": state.out_distinct})

        distribution = self._is_distribution_node(state)
        if distribution:
            emit(SemanticType.DISTRIBUTION_NODE, transaction.originator_account_id, state.out_distinct, 2,
                 (f"out-degree {state.out_distinct} vs in-degree {state.in_distinct}",),
                 {"out_distinct": state.out_distinct, "in_distinct": state.in_distinct})
        if self._is_collection_node(state):
            emit(SemanticType.COLLECTION_NODE, transaction.originator_account_id, state.in_distinct, 2,
                 (f"in-degree {state.in_distinct} vs out-degree {state.out_distinct}",),
                 {"out_distinct": state.out_distinct, "in_distinct": state.in_distinct})
        pass_through = self._is_pass_through(state)
        if pass_through:
            emit(SemanticType.PASS_THROUGH_ACCOUNT, transaction.originator_account_id, min(state.in_count, state.out_count), 2,
                 (f"inflow {state.in_value:.2f} vs outflow {state.out_value:.2f} in {state.currency}",),
                 {"in_value": round(state.in_value, 2), "out_value": round(state.out_value, 2),
                  "in_count": state.in_count, "out_count": state.out_count})
        if self._is_bookkeeping(state):
            emit(SemanticType.BOOKKEEPING_ACCOUNT, transaction.originator_account_id, state.self_count, 2,
                 (f"{state.self_count} of {state.event_count} events are self-postings",),
                 {"self_count": state.self_count, "event_count": state.event_count})

        # 3. Value, situated in the originator's own regime ----------------
        if len(amounts) < BASELINE_MINIMUM_EVENTS:
            emit(SemanticType.UNSCALED_VALUE, transaction.id, 1, 1,
                 (f"amount {transaction.amount:.2f} {currency} observed with {len(amounts)} prior same-currency events",),
                 {"amount": transaction.amount, "currency": currency, "prior_same_currency_events": len(amounts)})
        else:
            ordered = sorted(amounts)
            prior_max = ordered[-1]
            high_mark = _quantile(ordered, VALUE_REGIME_HIGH_QUANTILE)
            emit(SemanticType.VALUE_REGIME, f"{transaction.originator_account_id}|{currency}", len(ordered), 2,
                 (f"prior max {prior_max:.2f}, p{int(VALUE_REGIME_HIGH_QUANTILE * 100)} {high_mark:.2f} {currency}",),
                 {"prior_events": len(ordered), "prior_max": prior_max, "high_mark": high_mark, "currency": currency})
            facts = (
                f"amount {transaction.amount:.2f} {currency}",
                f"own prior max {prior_max:.2f}",
                f"own p{int(VALUE_REGIME_HIGH_QUANTILE * 100)} {high_mark:.2f}",
            )
            causal = {"amount": transaction.amount, "prior_max": prior_max, "high_mark": high_mark,
                      "prior_events": len(ordered), "currency": currency,
                      "break_multiple": VALUE_REGIME_BREAK_MULTIPLE}
            if transaction.amount > prior_max * VALUE_REGIME_BREAK_MULTIPLE:
                emit(SemanticType.UNEXPECTED_LARGE_TRANSFER, transaction.id, len(ordered), 2, facts, causal)
            elif transaction.amount >= high_mark:
                emit(SemanticType.EXPECTED_HIGH_VALUE_TRANSFER, transaction.id, len(ordered), 2, facts, causal)
            else:
                emit(SemanticType.ROUTINE_VALUE_TRANSFER, transaction.id, len(ordered), 2, facts, causal)

        # 4. Tempo, situated in the originator's own regime ----------------
        bucket = minute // TEMPO_BUCKET_MINUTES
        burst = state.burst_in(bucket) + 1
        if burst >= TEMPO_MINIMUM_BURST and prior_events >= BASELINE_MINIMUM_EVENTS:
            rate = state.mean_bucket_rate()
            tempo_facts = (f"{burst} outbound events in the current {TEMPO_BUCKET_MINUTES}-minute bucket",
                           f"own mean {rate:.3f} events per bucket")
            tempo_causal = {"burst": burst, "mean_bucket_rate": round(rate, 6),
                            "shift_multiple": TEMPO_SHIFT_MULTIPLE, "prior_outbound_events": prior_events}
            if distribution:
                emit(SemanticType.NORMAL_OPERATIONAL_BURST, transaction.originator_account_id, state.out_distinct, 2,
                     tempo_facts, tempo_causal)
            elif burst > TEMPO_SHIFT_MULTIPLE * rate:
                emit(SemanticType.BEHAVIOUR_REGIME_SHIFT, transaction.originator_account_id, prior_events, 2,
                     tempo_facts, tempo_causal)

        # 5. Relationship state --------------------------------------------
        if not self_posting:
            relationship_facts = (f"{pair_count} prior payments on this pair",)
            relationship_causal = {"pair_prior_count": pair_count, "prior_distinct_counterparties": state.out_distinct}
            if pair_count >= ESTABLISHED_RELATIONSHIP_MINIMUM:
                emit(SemanticType.ESTABLISHED_RELATIONSHIP, relationship_id, pair_count, 1,
                     relationship_facts, relationship_causal, (relationship_id,))
            elif pair_count >= 1:
                emit(SemanticType.RECENTLY_CREATED_RELATIONSHIP, relationship_id, pair_count, 2,
                     relationship_facts, relationship_causal, (relationship_id,))
            elif state.out_distinct >= BASELINE_MINIMUM_EVENTS:
                emit(SemanticType.FIRST_CONTACT, relationship_id, state.out_distinct, 2,
                     relationship_facts, relationship_causal, (relationship_id,))
            else:
                emit(SemanticType.NON_INFORMATIVE_NOVELTY, relationship_id, 1, 2,
                     relationship_facts + (f"originator has {state.out_distinct} known counterpart{'y' if state.out_distinct == 1 else 'ies'}, below the {BASELINE_MINIMUM_EVENTS} needed for a counterparty baseline",),
                     relationship_causal, (relationship_id,))

        # 6. Network motif --------------------------------------------------
        if (
            pass_through
            and state.last_in_currency == currency
            and state.last_in_amount > 0.0
            and 0 <= minute - state.last_in_minute <= LAYERING_WINDOW_MINUTES
            and abs(transaction.amount - state.last_in_amount) <= LAYERING_VALUE_TOLERANCE * state.last_in_amount
        ):
            emit(SemanticType.LAYERING_CHAIN_SEGMENT, transaction.id, min(state.in_count, state.out_count), 3,
                 (f"received {state.last_in_amount:.2f} {currency} {minute - state.last_in_minute} minutes earlier",
                  f"forwarding {transaction.amount:.2f} {currency}"),
                 {"inflow_amount": state.last_in_amount, "inflow_minutes_ago": minute - state.last_in_minute,
                  "outflow_amount": transaction.amount, "tolerance": LAYERING_VALUE_TOLERANCE},
                 (relationship_id,))

        withheld = WITHHELD_UNSUPPORTED
        if not resolved_legs:
            withheld = withheld + (WithheldObject(
                "ResolvedCounterpartyIdentity", ("account_reference_record",),
                "One or both accounts are absent from the reference data; identity claims are withheld.",
            ),)

        entities = originator.entities() + beneficiary.entities()
        return SemanticContextResult(
            transaction_id=transaction.id,
            objects=tuple(sorted(objects, key=lambda item: (item.type.value, item.id))),
            withheld=withheld,
            coverage_gaps=COVERAGE_GAP_NAMES,
            entities=entities,
            context_state_hash=self._state_hash(transaction, state, pair_count, len(amounts)),
        )

    @staticmethod
    def _state_hash(transaction: Transaction, state: _AccountState, pair_count: int, regime_size: int) -> str:
        """Content address of exactly the prior state the observation read."""
        payload = "|".join(str(part) for part in (
            ONTOLOGY_HASH[:16], transaction.originator_account_id, state.out_count, state.in_count,
            state.self_count, state.out_distinct, state.in_distinct, state.bucket_id, state.bucket_count,
            round(state.out_value, 4), round(state.in_value, 4), state.last_in_minute,
            round(state.last_in_amount, 4), state.last_in_currency, pair_count, regime_size,
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # -- causal commit ----------------------------------------------------
    def commit(self, transaction: Transaction) -> None:
        """Fold the event into context.  Never called before ``observe``."""
        originator_id = transaction.originator_account_id
        beneficiary_id = transaction.beneficiary_account_id
        currency = transaction.currency
        amount = transaction.amount
        minute = _minute(transaction.timestamp)
        bucket = minute // TEMPO_BUCKET_MINUTES
        pair_key = (originator_id, beneficiary_id)
        first_time_pair = pair_key not in self._pair
        self._pair[pair_key] = self._pair.get(pair_key, 0) + 1

        source = self._get(originator_id)
        source.out_count += 1
        source.out_value += amount
        source.last_minute = minute
        if source.first_minute < 0:
            source.first_minute = minute
        if source.bucket_id == bucket:
            source.bucket_count += 1
        else:
            source.bucket_id = bucket
            source.bucket_count = 1
        if not source.currency:
            source.currency = currency
        elif source.currency != currency:
            source.mixed_currency = True

        regime_key = (originator_id, currency)
        reservoir = self._regime.get(regime_key)
        if reservoir is None:
            self._regime[regime_key] = [amount]
        else:
            reservoir.append(amount)
            if len(reservoir) > VALUE_REGIME_RESERVOIR:
                del reservoir[0]

        if originator_id == beneficiary_id:
            source.self_count += 1
            source.in_count += 1
            source.in_value += amount
            source.last_in_minute = minute
            source.last_in_amount = amount
            source.last_in_currency = currency
        else:
            if first_time_pair:
                source.out_distinct += 1
            target = self._get(beneficiary_id)
            target.in_count += 1
            target.in_value += amount
            target.last_minute = minute
            if target.first_minute < 0:
                target.first_minute = minute
            target.last_in_minute = minute
            target.last_in_amount = amount
            target.last_in_currency = currency
            if not target.currency:
                target.currency = currency
            elif target.currency != currency:
                target.mixed_currency = True
            if first_time_pair:
                target.in_distinct += 1
        self.events_committed += 1

    def prime(self, transactions: Iterable[Transaction]) -> None:
        for transaction in transactions:
            self.commit(transaction)
