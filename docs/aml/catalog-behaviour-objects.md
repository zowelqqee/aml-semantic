# Behaviour Object Catalog

25 declared behavioural hypotheses. Source of truth:
`aml_runtime/behaviour/ontology.py` (declarations) and
`aml_runtime/behaviour/layer.py` (predicates). Changing anything here changes
`BEHAVIOUR_ONTOLOGY_HASH` and invalidates replay of earlier decisions.

**These are hypotheses, not labels.** Every emitted object carries
`counter_evidence`. A claim that cannot state what would weaken it is not
admitted to this catalog.

| | |
|---|---|
| **Purpose** | The closed vocabulary of behavioural hypotheses, with the predicate, horizon, direction, prior and counter-evidence of each. |
| **Inputs** | `aml_runtime/behaviour/ontology.py` (declarations) and `layer.py` (predicates) are the source of truth. |
| **Outputs** | 25 declared types across six families plus one honesty term. |
| **Guarantees** | Every constant is declared from the meaning of its claim, never fitted. Every emitted object carries counter-evidence. Editing this catalog changes `BEHAVIOUR_ONTOLOGY_HASH` and invalidates replay of earlier decisions. |
| **Limitations** | Four catalog terms are declared unsupported on this source. On the frozen window 11 of the 25 types never fired; the reasons are structural and are listed in the benchmark report. |


## Reading the tables

- **Horizon** — the temporal scale the claim is made over. `minutes` = 15 min
  bucket + predecessor; `hours` = 60 min bucket + predecessor.
- **Dir** — `risk` or `mitigation`. Both poles matter; the mitigating pole is
  what removes false positives.
- **Topic** — the corroboration bucket in the frozen `SemanticPolicyEngine`.
  Motif behaviours reuse the existing `network_motif` topic so they inherit
  high-concern status without the policy changing.
- **prior / k** — confidence is `prior × n/(n+k) × coverage`. A large `k` means
  the claim needs a lot of support before it is worth much.

## Declared constants referenced below

| Constant | Value | Meaning |
|---|---|---|
| `BEHAVIOUR_MINIMUM_OBSERVATIONS` | 6 | events before any behavioural claim |
| `FAN_MINIMUM_COUNTERPARTIES` | 8 | distinct counterparties that make a shape |
| `FAN_ASYMMETRY_RATIO` | 4.0 | degree asymmetry a directional shape needs |
| `SUSTAINED_BUCKET_MINIMUM` | 2 | hour buckets before a shape is "sustained" |
| `HUB_MINIMUM_DEGREE` | 20 | symmetric degree of an infrastructural node |
| `HUB_MAXIMUM_ASYMMETRY` | 2.0 | how unbalanced a hub may be |
| `ACCUMULATION_RETENTION` | 0.70 | retained share that means gathering |
| `TRANSIT_RETENTION_TOLERANCE` | 0.25 | in/out agreement that means transiting |
| `FLOW_MINIMUM_EVENTS` | 4 | events per direction before a flow claim |
| `PROMPT_FORWARD_MINUTES` | 30 | window in which a forward is "prompt" |
| `FORWARD_VALUE_TOLERANCE` | 0.15 | value agreement of a forward to its inflow |
| `FORWARD_EVENT_MINIMUM` | 3 | prompt forwards before forwarding is behaviour |
| `CASH_CONCENTRATION_SHARE` | 0.30 | cash share of inflow that concentrates |
| `BURST_RATE_MULTIPLE` | 4.0 | multiple of own bucket rate that is a burst |
| `BURST_MINIMUM_EVENTS` | 4 | floor below which a burst is not a burst |
| `DORMANCY_GAP_MULTIPLE` | 8.0 | multiple of own mean gap that is dormancy |
| `DORMANCY_MINIMUM_MINUTES` | 60 | absolute dormancy floor |
| `PAYROLL_AMOUNT_DISPERSION` | 0.25 | CV below which amounts are homogeneous |
| `PAYROLL_MINIMUM_PAYMENTS` | 5 | payments in a bucket before a run |
| `SUPPLIER_MINIMUM_ESTABLISHED` | 3 | established counterparties for settlement |
| `STABILITY_BUCKET_MINIMUM` | 3 | (×2) stable observations before stability |
| `RELATIONSHIP_GROWTH_RATIO` | 0.50 | new share of the set that is expansion |
| `PROVENANCE_DEPTH` | 2 | hops of funding provenance carried |

---

## Shape over time

What the account's counterparty graph looks like, and whether it has held that
shape for more than one bucket. This family replaces the primitive
"many outgoing transfers" / "many incoming transfers" observation.

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `DistributionBehaviour` | hours | risk | behaviour_shape | 0.85 / 20 | `distinct_out ≥ 8` ∧ `distinct_out ≥ 4 × max(1, distinct_in)` ∧ active in ≥ 2 hour buckets | the inbound counterparty count, or a note that a zero in-degree may be a short-window artefact |
| `FanOutDistribution` | minutes | risk | behaviour_shape | 0.80 / 10 | ≥ 8 *first-time* beneficiaries in the trailing short window | outbound amount dispersion, when it is low enough to also fit a payment run |
| `CollectionBehaviour` | hours | risk | behaviour_shape | 0.85 / 20 | `distinct_in ≥ 8` ∧ `distinct_in ≥ 4 × max(1, distinct_out)` ∧ active in ≥ 2 hour buckets | the outbound counterparty count — value is not only gathering |
| `FanInCollection` | minutes | risk | behaviour_shape | 0.80 / 10 | ≥ 8 *first-time* payers in the trailing short window | — |
| `SettlementHubBehaviour` | hours | **mitigation** | behaviour_shape | 0.85 / 20 | `distinct_out ≥ 20` ∧ `distinct_in ≥ 20` ∧ asymmetry ≤ 2× | prompt-forward count, when it also fits a transit reading |

`SettlementHubBehaviour` is the mitigating pole of this family: an account with
high degree in *both* directions is infrastructure, and a fan shape observed on
it is not a dispersal.

## Flow over time

Where value goes after it arrives. This family is what the semantic layer's
single-event `PassThroughAccount` snapshot cannot express.

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `MoneyAccumulationBehaviour` | hours | risk | behaviour_flow | 0.80 / 10 | `in_count ≥ 4` ∧ retention ≥ 0.70 | the outflow count, when accumulation is only partial |
| `TransitBehaviour` | hours | risk | behaviour_flow | 0.85 / 10 | `in_count ≥ 4` ∧ `out_count ≥ 4` ∧ `|in − out| / in ≤ 0.25` | self-posting count, which inflates both sides |
| `PassThroughBehaviour` | minutes | risk | behaviour_flow | 0.88 / 6 | ≥ 3 value-preserving forwards in the trailing short window | self-posting count |
| `LiquidityBalancingBehaviour` | hours | **mitigation** | behaviour_flow | 0.90 / 10 | `self_count ≥ 4` ∧ self share ≥ 0.80 | external beneficiary count |
| `CashConcentrationBehaviour` | hours | risk | behaviour_flow | 0.88 / 8 | `distinct_in ≥ 8` ∧ cash share of inflows ≥ 0.30 | — |

A **value-preserving forward** is the atom of this family: an outbound event
within 30 minutes of an inflow, in the same currency, whose amount matches the
inflow within 15%. It is computed once in the temporal engine and reused.

`LiquidityBalancingBehaviour` is declared against `*` in the conflict catalog —
an account whose flow is internal treasury movement qualifies every risk item
raised against it.

## Tempo over time

Rate, relative to the account's own rate. This family replaces the primitive
"velocity" observation, which had no baseline at all.

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `BurstActivityBehaviour` | minutes | risk | behaviour_tempo | 0.80 / 8 | short-window outbound count ≥ 4 ∧ > 4 × the account's own mean bucket rate | counterparty count, when a payment run is plausible |
| `HighVelocityLayering` | minutes | risk | **network_motif** | 0.92 / 4 | ≥ 3 preserving forwards ∧ ≥ 2 first-time beneficiaries in the *same* short bucket | — |
| `RapidLayeringBehaviour` | hours | risk | **network_motif** | 0.90 / 6 | ≥ 6 preserving forwards lifetime ∧ mean forward delay ≤ 15 min | retention, when the account also holds value |
| `DormantAccountActivation` | hours | risk | behaviour_lifecycle | 0.82 / 4 | idle ≥ 8 × own mean gap ∧ idle ≥ 60 min | — |

## Network motif over time

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `CircularMoneyMovement` | hours | risk | **network_motif** | 0.93 / 4 | beneficiary is the account's last funder (1 hop) or its funder's funder (2 hops), within the hour horizon | established-relationship status, when reciprocity is ordinary |

Provenance is carried to depth 2 (`PROVENANCE_DEPTH`): each account remembers
who paid it, and who paid *them*. Deeper cycles are not claimed, because a
constant-memory fold cannot support them exactly and an approximate cycle claim
would be an assertion rather than a hypothesis.

## Relationship evolution

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `RelationshipGrowthBehaviour` | hours | risk | behaviour_relationship | 0.75 / 10 | `distinct_out ≥ 8` ∧ new counterparties this hour ≥ 0.5 × the whole set | payments made to established counterparties |
| `RelationshipCollapseBehaviour` | hours | risk | behaviour_relationship | 0.75 / 10 | peak counterparty set ≥ 8 ∧ ≥ 4 payments this hour ∧ no new counterparty this hour | — |

## Regime over time — the mitigating pole

This family is where false positives go to die. Each of these behaviours is
declared in the conflict catalog as the qualifier of a specific semantic risk.

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `RoutinePayrollBehaviour` | hours | **mitigation** | behaviour_regime | 0.88 / 10 | ≥ 5 payments this hour ∧ amount dispersion ≤ 0.25 | how many went to first-time counterparties |
| `PayrollOperatorBehaviour` | hours | **mitigation** | behaviour_regime | 0.90 / 20 | routine payroll ∧ sustained ≥ 2 buckets ∧ `distinct_out ≥ 20` | — |
| `SupplierSettlementBehaviour` | hours | **mitigation** | behaviour_regime | 0.85 / 8 | ≥ 3 established counterparties ∧ ≥ 6 payments ∧ established share ≥ 0.50 | count of counterparties that remain non-established |
| `ExpectedBusinessCycle` | hours | **mitigation** | behaviour_stability | 0.86 / 10 | ≥ 6 consecutive observations inside the party's own value and tempo regimes ∧ zero breaks | — |
| `UnexpectedBusinessCycle` | hours | risk | behaviour_stability | 0.86 / 10 | a regime that held for ≥ 6 observations breaks on this event | — |

`ExpectedBusinessCycle` / `UnexpectedBusinessCycle` are the pair that turns the
brief's "regime shift" requirement into two objects rather than a threshold:
stability is asserted positively, and its breaking is a separate claim with its
own interval and confidence.

## Composite hypotheses

Composites require several independent families to agree. They are the only
behaviours whose predicate reads more than one family, and both are declared
with four inputs so their coverage factor is honest.

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate | Counter-evidence recorded |
|---|---|---|---|---|---|---|
| `MoneyMuleBehaviour` | hours | risk | behaviour_composite | 0.90 / 6 | `distinct_in ≥ 3` ∧ ≥ 3 prompt forwards ∧ retention ≤ 0.25 ∧ **no** established counterparty | observation span, when shorter than one hour |
| `ShellCompanyBehaviour` | hours | risk | behaviour_composite | 0.88 / 6 | incorporated entity form ∧ `out_count ≥ 4` ∧ in/out gap ≤ 0.25 ∧ no established counterparty ∧ ≥ 50% of payments cross a jurisdiction | — |

`ShellCompanyBehaviour` is the only behaviour that reads an *entity* attribute
(legal form, from the account reference data). It is a property of the party,
not of the transaction, and it is resolved rather than inferred.

## Honesty

| Behaviour | Horizon | Dir | Topic | prior/k | Predicate |
|---|---|---|---|---|---|
| `InsufficientBehaviouralHistory` | minutes | none | behaviour_coverage | 0.99 / 1 | fewer than 6 observed events |

Emitted **alone** — when it fires, no other behaviour is claimed. It is the
behavioural analogue of the semantic layer's `NoEstablishedBaseline`, and it
carries `direction = "none"` so the projection skips it: an admission of
ignorance is not evidence in either direction.

Note that `observed_events` counts **inbound and outbound** events. An account
can therefore be behaviourally readable while remaining semantically
baseline-less — five inflows and one outflow gives `NoEstablishedBaseline` from
the semantic layer but six observations here. This asymmetry is deliberate: it
is the mechanism by which the Behaviour Layer can characterise accounts the
Semantic Context Layer honestly abstains on, and it is exactly the shape a mule
account presents.

## Declared unsupported on this source

Never emitted; recorded per decision in `withheld` with the input they lack.

| Term | Missing input |
|---|---|
| `SeasonalBusinessCycle` | multi-season history |
| `MonthlySalaryCadence` | multi-month history |
| `LongTermDormancyReactivation` | multi-week history |
| `AnnualTaxCycle` | multi-year history |

The frozen 500,000-event window spans 5.6 hours. The `days` and `weeks` horizons
exist in the engine and never fill; every `LifecycleObject` reports them under
`horizons_unfillable`.
