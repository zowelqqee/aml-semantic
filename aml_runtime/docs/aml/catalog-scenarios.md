# Scenario Catalog

A scenario is an **ordered story** an account is in, matched over its recent
stage history. Source of truth: `SCENARIO_PATTERNS` in
`aml_runtime/behaviour/ontology.py`; stage derivation in
`BehaviourLayer._stage`.

Scenarios are hypotheses, like behaviour objects. Risk-class scenarios carry
`counter_evidence` naming every mitigating behaviour present at the same moment.

| | |
|---|---|
| **Purpose** | The declared stage vocabulary and the ordered stage patterns that constitute a scenario. |
| **Inputs** | `SCENARIO_PATTERNS` in `aml_runtime/behaviour/ontology.py`; stage derivation in `BehaviourLayer._stage`. |
| **Outputs** | 8 stages and 6 declared patterns, matched as ordered subsequences over the last six stages. |
| **Guarantees** | Matching is order-sensitive and gap-tolerant. Confidence is a declared function of pattern length. Risk-class scenarios carry counter-evidence naming any mitigating behaviour present. |
| **Limitations** | `Hold` is declared but never emitted — it needs a balance model this source does not carry. Scenarios are scoped to one account; cross-account stories need provenance deeper than the declared two hops. |


---

## Stages

A stage is derived per event from the semantic reading plus temporal state. Each
account keeps the last `SCENARIO_STAGE_MEMORY = 6` stages with their minutes.

| Stage | Emitted when | Derived from |
|---|---|---|
| `Receive` | pushed onto the **beneficiary's** history at commit | any non-self-posting event |
| `Settle` | the event is a self-posting or an intra-customer transfer | `InternalBookEntry`, `IntraCustomerTransfer` |
| `Forward` | ≥ 1 value-preserving forward in the trailing short window, or a semantic `LayeringChainSegment` | temporal engine + semantic object |
| `Split` | ≥ 3 first-time beneficiaries in the trailing short window | `FirstContact` / `NonInformativeNovelty` counts |
| `Distribute` | ≥ 5 payments this hour with amount dispersion ≤ 0.25 | temporal engine |
| `Payments` | any other outbound event | default outbound stage |
| `Dormant` | idle beyond the declared dormancy threshold | lifecycle |
| `Hold` | **declared, never emitted** | requires a balance model this source does not carry |

`Hold` is in the vocabulary because the brief's canonical layering story is
`Receive → Hold → Split → Forward`. IBM AML `HI-Small` carries no balances, so a
"held" state cannot be distinguished from "idle between events". Rather than
approximate it, the layer declares the stage and never emits it — and the
`LayeringAttempt` pattern is written so it does not depend on it.

## Matching

Ordered **subsequence** over the last six stages, current event included:

```
pattern  = (Receive, Split, Forward)
observed = (Settle, Receive, Payments, Split, Payments, Forward)   → MATCH
observed = (Receive, Forward, Split)                               → NO MATCH (order)
```

Subsequence rather than contiguous run: unrelated activity between the steps of
a story does not break the story, which is precisely how a real dispersal looks
when it is mixed into ordinary traffic.

## Confidence

```
confidence = SCENARIO_PRIOR × len(pattern) / (len(pattern) + 1)
           = 0.90 × len / (len + 1)
```

| Pattern length | Confidence |
|---|---|
| 2 | 0.600 |
| 3 | 0.675 |
| 4 | 0.720 |

Longer patterns are stronger because more independent stages had to line up in
the right order. `SCENARIO_PRIOR` is declared, never fitted.

---

## The six declared patterns

### `LayeringAttempt` — risk

```
Receive ──► Split ──► Forward
```

Value arrived, was broken up across new counterparties, and was forwarded
onward. Confidence 0.675.

The brief's canonical form is `Receive → Hold → Split → Forward → Dormant`. This
pattern is the sub-story that is observable without a balance model; the
`Dormant` tail is covered separately by `DormancyAfterOutflow` so that a real
layering run matches both.

**Counter-evidence recorded**: every mitigating behaviour present — typically
`RoutinePayrollBehaviour` (the "split" is a payment run) or
`SettlementHubBehaviour` (the account is infrastructure).

### `CollectThenForward` — risk

```
Receive ──► Receive ──► Receive ──► Forward
```

Value arrived from several sources and was forwarded onward. Confidence 0.720 —
the strongest pattern in the catalog, because four stages had to align.

This is the mule story, and it is the pattern that most directly targets the
18-of-28 laundering-labelled events the Semantic Runtime left in `ABSTAIN`. It
pairs with `MoneyMuleBehaviour`, but the two are independent: the scenario is
about *order*, the behaviour about *ratios*.

**Counter-evidence recorded**: mitigating behaviours present, e.g.
`LiquidityBalancingBehaviour` (the receipts are internal treasury movement).

### `DormancyAfterOutflow` — risk

```
Forward ──► Dormant
```

Value was forwarded and the account then went quiet. Confidence 0.600.

The tail of a layering run. On a 5.6-hour window this fires rarely, because
`Dormant` needs idle ≥ 8× the account's own mean gap **and** ≥ 60 minutes, and
most accounts in this window are not observed for long enough to establish a
mean gap.

### `DistributionRun` — neutral

```
Receive ──► Distribute
```

Value arrived and was paid out across many counterparties at once.
Confidence 0.600.

Deliberately **not** a risk scenario. The same shape describes a payroll run and
a dispersal; the distinction lives in the behaviour objects
(`RoutinePayrollBehaviour` versus `FanOutDistribution`), not in the stage
sequence. Encoding it as risk here would double-count.

### `NormalConsumerBehaviour` — mitigating

```
Receive ──► Payments ──► Payments
```

Value arrived and was spent in ordinary routine payments. Confidence 0.675.

The brief's `SalaryReceive → BillPayments → Shopping → Savings` story, reduced to
what this source can support: it has no merchant category, no bill/shopping
distinction and no savings product. Emitting the four-stage version would
fabricate three of its four stages.

### `TreasuryCycling` — mitigating

```
Settle ──► Settle
```

Activity is internal settlement between accounts of one party.
Confidence 0.600.

The most frequently matched scenario on this source, which is unsurprising:
34,050 of the 100,000 evaluation events are self-postings. It pairs with
`LiquidityBalancingBehaviour`.

---

## Coverage

| Brief's example | Status |
|---|---|
| `Receive → Hold → Split → Forward → Dormant` | split across `LayeringAttempt` + `DormancyAfterOutflow`; `Hold` not observable |
| `SalaryReceive → BillPayments → Shopping → Savings` | reduced to `NormalConsumerBehaviour`; merchant category, product type and savings are absent from the source |

## Not implemented in v1

- **Cross-account scenarios.** Every pattern here is scoped to one account's
  stage history. A scenario spanning `A → B → C` as a single object requires
  provenance deeper than the declared two hops.
- **Scenario supersession.** A scenario match is re-derived per event rather
  than being an object with a lifecycle of its own.
- **`Hold`.** Needs a balance model.
