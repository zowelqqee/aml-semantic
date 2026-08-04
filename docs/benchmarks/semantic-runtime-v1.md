# Semantic Runtime v1 — first measurement

Architecture: [`semantic_context_architecture.md`](../aml/semantic-context-layer.md).
Implementation: `aml_runtime/semantic/`. Benchmark: `aml_runtime/semantic_benchmark.py`.
Raw outputs: `artifacts/aml_semantic_v1/`.

| | |
|---|---|
| **Purpose** | Measure whether reasoning about semantic objects instead of transactions reduces false positives, before any tuning. |
| **Inputs** | IBM AML `HI-Small`, first 500,000 chronological events; evaluation set **E1** (100,000 events, 28 labels). |
| **Outputs** | `artifacts/aml_semantic_v1/` — results JSON, object census, decision examples, audit stream, curves. |
| **Guarantees** | Constants declared before execution; labels read only for fitting and post-decision evaluation; runs verified byte-identical on repeat. |
| **Limitations** | 28 positives is a small evaluation set. A 5.6-hour window is why 83% of events have no baseline. Recall falls to zero unaided. |


## What changed

The Runtime no longer reasons about transactions. It reasons about semantic
objects — `InternalBookEntry`, `IntraCustomerTransfer`, `UnexpectedLargeTransfer`,
`NonInformativeNovelty`, `NormalOperationalBurst`, `LayeringChainSegment` — that
the Semantic Context Layer infers from causal history, entity reference data and
network shape. Facts are predicates over those objects, not over CSV columns.

The v0.2 Runtime is untouched and retained as a benchmark arm.

## Protocol

Identical to the frozen protocol used by the earlier cascade experiments, so the
arms are comparable to previously published artifacts.

| | |
|---|---|
| Source | `data/ibm_aml_data/HI-Small_Trans.csv` (5,078,345 rows) |
| Reference data | `data/ibm_aml_data/HI-Small_accounts.csv` (518,581 rows) |
| Input cap | first 500,000 chronological events (2022-09-01 00:00 → 05:39) |
| Train | events 0–399,999 (model fitting; context priming) |
| Evaluation horizon | events 400,000–499,999, containing **28** laundering-labelled events |
| Accounts resolved | 327,325 |
| Ontology | `aml-semantic-ontology/1.0`, hash `80103ed6…` |

**Label boundary.** The laundering column is read into a separate array used for
model fitting and post-decision evaluation only. No semantic object, profile,
relationship, evidence item, policy or audit reads it. The context layer is
constructed without access to it.

**Tuning statement.** Every constant in the semantic layer is declared in
`aml_runtime/semantic/ontology.py` and `SemanticPolicyEngine`, fixed before the
benchmark ran, and derived from the meaning of the claim it governs. **No
threshold was selected against the evaluation labels, and no constant was
changed after seeing a result.**

The benchmark was executed three times. Run 1 contained a correctness bug: a
model probability was counted as an independent *semantic* topic in the
corroboration rule, so it escalated events regardless of its value (visible as
`Semantic + CatBoost` alerting on 7,420 events while CatBoost alone alerted on
381 — impossible if the band were being applied). The fix excludes
`ml_probability` from the semantic corroboration count; it is documented in §11
of the architecture and covered by a test. Run 3 regenerated artifacts after a
cosmetic wording change in one `supporting_facts` string, and is **byte-identical
to run 2** in every metric, census and conflict count — which is also the
determinism check for the layer.

The numbers below are run 2/3.

## Results

| Arm | TP | FP | FN | Recall | Precision | Alerts | Alert rate | ML inferences |
|---|---|---|---|---|---|---|---|---|
| **Runtime only** (frozen v0.2) | 14 | 44,680 | 14 | 0.5000 | 0.000313 | 44,694 | 44.69% | 0 |
| **Semantic Runtime only** | 0 | **382** | 28 | 0.0000 | 0.000000 | 382 | **0.38%** | 0 |
| ML only — XGBoost | 3 | 424 | 25 | 0.1071 | 0.007026 | 427 | 0.43% | 100,000 |
| ML only — LightGBM | 20 | 20,120 | 8 | 0.7143 | 0.000993 | 20,140 | 20.14% | 100,000 |
| ML only — CatBoost | 3 | 378 | 25 | 0.1071 | 0.007874 | 381 | 0.38% | 100,000 |
| Runtime + ML — XGBoost | 17 | 45,054 | 11 | 0.6071 | 0.000377 | 45,071 | 45.07% | 100,000 |
| Runtime + ML — LightGBM | 28 | 53,767 | 0 | 1.0000 | 0.000520 | 53,795 | 53.80% | 100,000 |
| Runtime + ML — CatBoost | 16 | 45,057 | 12 | 0.5714 | 0.000355 | 45,073 | 45.07% | 100,000 |
| **Semantic + ML — XGBoost** | 2 | 444 | 26 | 0.0714 | 0.004484 | 446 | 0.45% | 49,310 |
| **Semantic + ML — LightGBM** | 18 | 12,841 | 10 | 0.6429 | 0.001400 | 12,859 | 12.86% | 49,310 |
| **Semantic + ML — CatBoost** | 1 | 429 | 27 | 0.0357 | 0.002326 | 430 | 0.43% | 49,310 |

### The false-positive question the brief asked

Semantic reasoning reduces false positives before any tuning, in every paired
comparison:

| Comparison | FP before | FP after | Change | Precision change |
|---|---|---|---|---|
| Runtime only → Semantic Runtime only | 44,680 | 382 | **−99.1%** | — (both arms' precision is 0 or near 0) |
| Runtime + XGBoost → Semantic + XGBoost | 45,054 | 444 | **−99.0%** | 0.000377 → 0.004484 (**11.9×**) |
| Runtime + LightGBM → Semantic + LightGBM | 53,767 | 12,841 | **−76.1%** | 0.000520 → 0.001400 (**2.7×**) |
| Runtime + CatBoost → Semantic + CatBoost | 45,057 | 429 | **−99.0%** | 0.000355 → 0.002326 (**6.6×**) |

ML inference volume falls from 100,000 to 49,310 (**−50.7%**) because the model
is asked only about events the semantic layer has admitted it cannot
characterise.

### The recall cost, stated plainly

The Semantic Runtime alone captures **0 of 28** laundering-labelled events; the
frozen Runtime captured 14 — while alerting on 44.7% of the stream, which is a
capture rate barely above what alerting on half of everything produces by
chance. `Runtime + LightGBM` reaches recall 1.000 at the cost of alerting on
53.8% of all traffic. These are not operable configurations; the semantic arms
are, and their recall is genuinely lower.

## Where the 28 positives actually land

Post-hoc, after all decisions were made:

| Semantic decision | Positives |
|---|---|
| `ABSTAIN` (undetermined; routed to ML) | **18** |
| `ALLOW` | 10 |
| `REVIEW` / `BLOCK` | 0 |

This is the operationally important number. 18 of 28 fall into the state where
the layer says *"I have no baseline for this party and no reference frame for
this value"* — and `Semantic + LightGBM` recovers **18 of those 18**, i.e. 100%
of the routable ceiling, at 12,841 false positives instead of the 53,767 the
frozen Runtime + LightGBM pays.

All 18 carry exactly the same reading: `NoEstablishedBaseline`,
`UnscaledValue`, `NonInformativeNovelty` — an account with no history paying a
counterparty it has never used, in `ACH` format. The layer is correct that it
cannot characterise them; it is also correct that this is not, on its own, a
finding.

The 10 that reach `ALLOW` all sit on `DistributionNode` accounts and read
`NormalOperationalBurst` + `RoutineValueTransfer` + `ValueRegime`: value inside
the party's own regime, tempo matching the party's own operational shape. On
this source they are structurally indistinguishable from ordinary distribution
activity. That is a limit of the data, not a threshold that needs moving —
separating them requires the inputs listed under coverage gaps below.

## Semantic object census (evaluation horizon)

24 of the 26 emittable types were observed. `CollectionNode` and
`BookkeepingAccount` require shapes the 5.6-hour window never produces.

| Type | Emissions | Type | Emissions |
|---|---|---|---|
| `UnscaledValue` | 83,608 | `DistributionNode` | 6,820 |
| `NoEstablishedBaseline` | 82,935 | `NormalOperationalBurst` | 6,730 |
| `InternalBookEntry` | 34,050 | `FirstContact` | 3,518 |
| `RecentlyCreatedRelationship` | 30,928 | `HighRiskJurisdictionExposure` | 2,697 |
| `NonInformativeNovelty` | 28,845 | `EstablishedRelationship` | 2,659 |
| `CrossJurisdictionTransfer` | 21,030 | `VirtualAssetExposure` | 2,593 |
| `TempoRegime` | 17,065 | `ExpectedHighValueTransfer` | 2,473 |
| `ValueRegime` | 16,392 | `CurrencyConversionTransfer` | 1,047 |
| `RoutineValueTransfer` | 13,553 | `IntraCustomerTransfer` | 1,037 |
| `CashInstrumentSettlement` | 7,407 | `UnexpectedLargeTransfer` | 366 |
| `CounterpartyRegime` | 7,355 | `BehaviourRegimeShift` | 312 |
| | | `PassThroughAccount` | 114 |
| | | `LayeringChainSegment` | 1 |

Two rows carry most of the argument:

- **`InternalBookEntry` = 34,050.** A third of the evaluation horizon is
  self-postings — the same account on both legs. The v0.2 rule vocabulary had no
  way to say "this is not a transfer between parties", so it processed every one
  of them as an inter-party payment to a new beneficiary. This single missing
  concept is the largest identified source of its false positives.
- **`NoEstablishedBaseline` = 82,935.** On a 5.6-hour window, 83% of events
  originate from a party with fewer than five prior outbound events. The frozen
  Runtime called this "new beneficiary" and treated it as evidence. It is not
  evidence; it is the default state of the world.

## Conflicts finally exist

The v0.2 report measured **conflict frequency 0.0 per transaction** on this
source, because the IBM schema carries no controls. Semantic objects supply the
missing negative pole:

| Conflict kind | Count over 100,000 events |
|---|---|
| `value_context` | 5,359 |
| `structural_explanation` | 3,726 |
| `counterparty_context` | 5 |

Mitigating evidence fires 89,347 times against 18,055 risk evidence items. The
conflict engine stopped being decorative.

## Cost

| Stage | Seconds (500k events) |
|---|---|
| Entity resolution (327,325 accounts) | 2.57 |
| ML feature construction | 5.63 |
| Frozen Runtime decisions (100k) | 3.18 |
| Semantic Runtime decisions (100k) | 5.80 |

Semantic decision latency is 0.058 ms/event against the frozen Runtime's
0.032 ms. Peak process RSS for the whole benchmark was 1.21 GB, dominated by the
context fold's per-account state and the 400,000 × 18 training matrix.

## What this does and does not establish

**Established.** Every decision in the semantic arms is expressible in semantic
objects. `artifacts/aml_semantic_v1/decision_examples.md` contains rationales of
the form *"ALLOW: the semantic reading is determined and no unqualified
corroboration remains. Reading: InternalBookEntry, UnscaledValue"* — no rule
identifier appears. Each audit record carries the objects, their causal
evidence, the confidence arithmetic, the claims that were **withheld** and why,
the coverage gaps, and the replay pins. This was the brief's success criterion.

**Established.** Reinterpreting the same stream semantically, with no tuning,
removes 76–99% of the false positives of the corresponding rule-first arm and
halves ML inference volume.

**Not established.** That semantic reasoning improves detection. Recall falls in
every paired comparison. On this window the ceiling for the semantic + ML design
is 18 of 28, and the remaining 10 are unreachable without inputs this source
does not carry.

**Not established.** External validity. A 5.6-hour window is why 83% of events
have no baseline; a multi-month window would move most of `ABSTAIN` into
determinate states and is the single change most likely to alter every number
above.

## Declared coverage gaps

Recorded on every decision, never proxied: `sar_feed`, `kyc_dates`,
`sanctions_list`, `source_of_funds`, `declared_activity`, `multi_month_history`.

Six ontology terms are declared unsupported on this source and are never
emitted: `SalaryDistribution`, `MortgagePayment`, `TaxPayment`,
`DividendDistribution`, `DormantRelationshipReactivated`, `SeasonalBusiness`.
Emitting them would require a payroll calendar, loan schedules, a tax-authority
registry, a shareholder registry, and multi-month history — none of which exist
here.

## Next, without tuning anything

1. **Widen the window.** Re-run on a multi-week chronological prefix. The
   hypothesis is explicit and falsifiable: `NoEstablishedBaseline` falls sharply,
   `ABSTAIN` shrinks, and the semantic arms gain recall without gaining alerts.
2. **Attach the missing feeds.** Every coverage gap above is a semantic type the
   ontology already names.
3. **Second-order network motifs.** `LayeringChainSegment` fired once, because
   `PassThroughAccount` needs five events in each direction and the window
   rarely supplies them. Multi-hop motifs need the wider window from (1).
4. **Replay verification at scale.** The pins exist; a harness that re-executes
   a sampled audit stream and asserts byte equality does not yet.

## Reproduce

```bash
python -m aml_runtime.semantic_benchmark --transactions data/ibm_aml_data/HI-Small_Trans.csv --accounts data/ibm_aml_data/HI-Small_accounts.csv --output-dir artifacts/aml_semantic_v1
```
