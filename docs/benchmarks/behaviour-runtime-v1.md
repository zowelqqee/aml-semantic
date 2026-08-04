# Semantic Behaviour Runtime v1 — measurement and analysis

Design: [`behaviour_layer_architecture.md`](../aml/behaviour-layer.md),
[`behaviour_object_catalog.md`](../aml/catalog-behaviour-objects.md),
[`scenario_catalog.md`](../aml/catalog-scenarios.md),
[`role_transition_catalog.md`](../aml/catalog-roles.md).
Implementation: `aml_runtime/behaviour/`. Benchmark:
`aml_runtime/behaviour_benchmark.py`.
Generated artifacts: `artifacts/aml_behaviour_v1/` —
`benchmark_report.md`, `comparison_report.md`, `case_studies.md`,
`comparison_results.json`, `curves.png`, `audits/`.

Predecessor: [`semantic-runtime-v1.md`](semantic-runtime-v1.md).

| | |
|---|---|
| **Purpose** | Measure whether reasoning about behaviour over time changes the decision, with the Semantic Context Layer held unchanged. |
| **Inputs** | IBM AML `HI-Small`, first 500,000 chronological events; evaluation set **E1** (100,000 events, 28 labels). |
| **Outputs** | `artifacts/aml_behaviour_v1/` — benchmark report, comparison report, case studies for every labelled event, audit stream, curves. |
| **Guarantees** | The frozen policy engine selects every decision; behaviour reaches it as a projection of the catalog, not as new rules. Three defects found during the run were unsatisfiable predicates, not threshold changes, and are documented with their tests. |
| **Limitations** | +1 true positive for +1,824 false positives. 11 of 25 behaviour types never fired. The ML arm uses the semantic feature space and is not comparable to the raw-feature arm of the previous report. |


## What changed

The Runtime now reasons about **behaviour over time**, not about isolated
semantic objects. Between the Semantic Context Layer and the policy engine sits
a temporal fold that accumulates semantic objects per account and emits
behaviour objects, a dynamic role, role transitions, scenario matches and a
lifecycle object.

The Semantic Context Layer is **unchanged** — not subclassed, not patched, not
re-implemented. So is `SemanticPolicyEngine`: the same corroboration rule (two
independent unqualified topics, or one high-concern topic above 0.90) selects
every decision in arms 2–4. Behaviour reaches it as a **projection** of the
catalog onto the existing `Evidence` type, with direction, topic and source
reliability read from each behaviour's own ontology entry. No behaviour rule
engine exists, and no behaviour meaning is declared twice.

## Protocol

Identical to the frozen protocol used by every earlier experiment.

| | |
|---|---|
| Source | `data/ibm_aml_data/HI-Small_Trans.csv`, first 500,000 chronological events |
| Reference data | `data/ibm_aml_data/HI-Small_accounts.csv` — 327,325 accounts resolved |
| Window | 2022-09-01 00:00 → 05:39 (5.6 hours) |
| Train | events 0–399,999 |
| Evaluation horizon | events 400,000–499,999, containing **28** laundering-labelled events |
| Semantic ontology | `aml-semantic-ontology/1.0`, hash `80103ed6…` |
| Behaviour ontology | `aml-behaviour-ontology/1.0`, hash `7b99b0c1…` |

**Label boundary.** The laundering column is read into a separate array used for
model fitting and post-decision evaluation only. No semantic object, behaviour
object, role, scenario, lifecycle object, evidence item, policy or audit reads
it. Both layers are constructed without access to it.

**Tuning statement.** Every constant lives in `semantic/ontology.py` and
`behaviour/ontology.py`, was declared from the meaning of the claim it governs,
and was fixed before the benchmark ran. **No threshold was selected against the
evaluation labels and none was changed after seeing a result.**

**Execution history.** Four runs. Three defects were found and fixed between
runs; each was a predicate that was *unsatisfiable or false by construction*,
not a threshold adjustment:

1. `LiquidityBalancingBehaviour` measured the self-posting share against total
   events. A self-posting increments both the inbound and outbound counter, so
   the ratio capped at 0.5 and could never reach the declared 0.80 — the claim
   was unreachable. The denominator is now outbound events.
2. `TransitBehaviour` counted self-postings as flow, so an account that only
   posted to itself satisfied "value arrives and leaves" trivially. It now
   requires external flow in both directions.
3. Self-postings were being matched as *value-preserving forwards to
   themselves*, manufacturing whole forwarding chains out of book entries. Self
   postings no longer participate in forward or provenance detection.

All three are covered by tests. Run 4 differs from run 3 only in one reporting
field. The numbers below are run 4.

---

## Results

| Arm | TP | FP | FN | Precision | Recall | F1 | Alerts | Alert rate | ML inferences |
|---|---|---|---|---|---|---|---|---|---|
| **Runtime** (frozen v0.2) | 14 | 44,680 | 14 | 0.000313 | 0.5000 | 0.000626 | 44,694 | 44.69% | 0 |
| **Semantic Runtime** | 0 | 382 | 28 | 0.000000 | 0.0000 | 0.000000 | 382 | 0.38% | 0 |
| **Semantic Behaviour Runtime** | **1** | 2,206 | 27 | **0.000453** | 0.0357 | **0.000895** | 2,207 | 2.21% | 0 |
| Semantic Behaviour + ML — XGBoost | 1 | 2,328 | 27 | 0.000429 | 0.0357 | 0.000849 | 2,329 | 2.33% | 48,709 |
| Semantic Behaviour + ML — LightGBM | 13 | 15,787 | 15 | 0.000823 | 0.4643 | 0.001643 | 15,800 | 15.80% | 48,709 |
| Semantic Behaviour + ML — CatBoost | 1 | 2,312 | 27 | 0.000432 | 0.0357 | 0.000854 | 2,313 | 2.31% | 48,709 |

### Behaviour versus the rule-first runtime

| Metric | Runtime | Semantic Behaviour Runtime | Change |
|---|---|---|---|
| False positives | 44,680 | 2,206 | **−95.1%** |
| Alert rate | 44.69% | 2.21% | **−95.1%** |
| Precision | 0.000313 | 0.000453 | **+45%** |
| F1 | 0.000626 | 0.000895 | **+43%** |
| Recall | 0.5000 | 0.0357 | −93% |

Precision and F1 are the best of the three unaided arms. The behavioural runtime
is the only unaided arm that is both better than the rule-first runtime on
precision **and** operable on alert volume.

### Behaviour versus the semantic runtime

| Metric | Semantic Runtime | Semantic Behaviour Runtime |
|---|---|---|
| True positives | 0 | 1 |
| False positives | 382 | 2,206 (+477%) |
| Precision | 0.000000 | 0.000453 |
| F1 | 0.000000 | 0.000895 |

The Behaviour Layer bought its first true positive with 1,824 additional false
positives. That is a poor trade in isolation. It is reported as measured.

**Decision movement** (Semantic → Behaviour), the whole of the difference:

| Movement | Events |
|---|---|
| `ALLOW → REVIEW` | 1,227 |
| `ABSTAIN → REVIEW` | 603 |
| `REVIEW → ALLOW` | 3 |
| `REVIEW → ABSTAIN` | 2 |

1,830 events were escalated because a behavioural topic supplied the second
independent corroboration the frozen policy requires. Exactly one of them was
laundering-labelled. Five events were de-escalated by behavioural mitigation.

---

## Behaviour layer output

| Quantity | Count |
|---|---|
| Behaviour objects generated | 98,913 |
| Role transitions | 16,218 |
| Scenario detections | 28,428 |
| Conflicts | 10,603 |

### Behaviour census — 14 of 25 types fired

| Behaviour | Emissions | | Behaviour | Emissions |
|---|---|---|---|---|
| `InsufficientBehaviouralHistory` | 77,532 | | `LiquidityBalancingBehaviour` | 97 |
| `DistributionBehaviour` | 6,831 | | `TransitBehaviour` | 82 |
| `FanOutDistribution` | 6,593 | | `UnexpectedBusinessCycle` | 50 |
| `BurstActivityBehaviour` | 3,483 | | `RoutinePayrollBehaviour` | 20 |
| `DormantAccountActivation` | 1,788 | | `FanInCollection` | 3 |
| `MoneyAccumulationBehaviour` | 1,524 | | `RelationshipGrowthBehaviour` | 2 |
| `ShellCompanyBehaviour` | 541 | | | |
| `ExpectedBusinessCycle` | 367 | | | |

**77,532 of 100,000 events yield `InsufficientBehaviouralHistory`.** On a
5.6-hour window, 78% of originating accounts have fewer than six observed
events. The behavioural analogue of the semantic layer's `NoEstablishedBaseline`
dominates for the same reason: there is not enough history to say anything.

Eleven types never fired, and the reasons are structural, not accidental:

| Type | Why it could not fire on this window |
|---|---|
| `CollectionBehaviour`, `CashConcentrationBehaviour` | a role is computed for the *originator* of an event; a pure collector rarely originates, so its shape is rarely read |
| `SettlementHubBehaviour` | needs 20 counterparties in *both* directions inside 5.6 hours |
| `PassThroughBehaviour`, `RapidLayeringBehaviour`, `HighVelocityLayering` | need 3–6 prompt value-preserving forwards; after defect 3 was fixed, genuine forwards are rare here |
| `CircularMoneyMovement` | no value returned to a recent funder within the hour horizon |
| `PayrollOperatorBehaviour` | requires `RoutinePayrollBehaviour` (20 emissions) *and* 20 counterparties *and* two buckets |
| `SupplierSettlementBehaviour` | requires ≥ 3 established counterparties and ≥ 50% established share |
| `RelationshipCollapseBehaviour` | requires a broad set that then contracts, over hours |
| `MoneyMuleBehaviour` | requires 3 payers, 3 prompt forwards and no established relationship — reachable in principle, absent here |

The first row is a **design limitation worth naming**: the layer reads behaviour
from the originator's perspective, so accounts that predominantly receive are
under-described. That is not a data limitation.

### Roles — 10 of 13 observed, 16,218 transitions

| Role | Events | | Role | Events |
|---|---|---|---|---|
| `Unknown` | 71,343 | | `ShellCompanyCandidate` | 541 |
| `ActiveCounterparty` | 11,193 | | `TreasuryAccount` | 97 |
| `Distributor` | 6,851 | | `TransitAccount` | 65 |
| `Dormant` | 6,326 | | `Collector` | 3 |
| `SalaryReceiver` | 2,064 | | | |
| `Accumulator` | 1,517 | | | |

Roles do evolve, and the transitions are causal records naming the behaviour
objects that forced them. The most frequent are lifecycle transitions
(`Unknown → Dormant` 5,595; `Unknown → ActiveCounterparty` 3,394;
`Dormant → ActiveCounterparty` 1,704). Behaviour-driven transitions are rarer
and more interesting: `ActiveCounterparty → TreasuryAccount` (50),
`ActiveCounterparty → TransitAccount` (24), `ActiveCounterparty → Distributor`
(21), `SalaryReceiver → TransitAccount` (7).

`SalaryReceiver` (2,064 events) is worth noting: it is inferred from the
*funder's* current role, so it is the one place a role propagates across the
graph, and it required no rule about salaries.

**Role oscillation is real.** `Dormant → Unknown` (2,416) and
`Dormant → ActiveCounterparty` (1,704) run alongside `Unknown → Dormant`
(5,595): sparse accounts flip in and out of dormancy as their own mean gap
shifts. There is no hysteresis, deliberately — adding one would be a tuning
decision.

### Scenarios — 3 of 6 matched

| Scenario | Detections | Laundering-labelled |
|---|---|---|
| `TreasuryCycling` | 17,329 | 3 |
| `NormalConsumerBehaviour` | 11,089 | 0 |
| `CollectThenForward` | 10 | 0 |
| `LayeringAttempt` | 0 | — |
| `DistributionRun` | 0 | — |
| `DormancyAfterOutflow` | 0 | — |

`LayeringAttempt` requires `Receive → Split → Forward` in order within six
stages; `DormancyAfterOutflow` requires a `Forward` followed by a `Dormant`.
Neither occurred. `CollectThenForward` — the mule story, the pattern aimed
squarely at the cases the semantic layer abstained on — matched 10 times and
**never on a laundering-labelled event**.

### Conflicts — 10,603

| Kind | Count |
|---|---|
| `value_context` | 5,359 |
| `structural_explanation` | 5,019 |
| `behaviour_structural` | 146 |
| `behaviour_stability` | 58 |
| `behaviour_regime` | 16 |

The behavioural conflict kinds fire 220 times against 10,383 semantic ones. The
mitigating behaviours that were meant to qualify semantic risk
(`RoutinePayrollBehaviour`, `ExpectedBusinessCycle`,
`LiquidityBalancingBehaviour`) are exactly the ones this window barely produces.

---

## Analysis: every laundering-labelled event

All 28, from the audit stream, after decisions were made.

| Behaviour decision | Positives |
|---|---|
| `ABSTAIN` (routed to ML) | 18 |
| `ALLOW` | 9 |
| `REVIEW` | **1** |

**Which behaviour objects existed?** Only four types appear across all 28
events: `InsufficientBehaviouralHistory` (17), `DistributionBehaviour` (10),
`FanOutDistribution` (9), `DormantAccountActivation` (1).

**Which scenario objects existed?** `TreasuryCycling` on 3 events; none on the
other 25. No risk-class scenario matched any laundering-labelled event.

**Which roles existed?** `Unknown` (17), `Distributor` (10), `Dormant` (1).

**Why did the Runtime succeed or fail?**

*The one success.* `IBM-ML-00446304`. Role `Distributor`, behaviours
`DistributionBehaviour` + `FanOutDistribution`, semantic objects including
`FirstContact`, `CrossJurisdictionTransfer` and `ExpectedHighValueTransfer`. The
Semantic Runtime said `ALLOW` because only one semantic topic
(`counterparty_regime`) survived qualification. The Behaviour Layer added
`behaviour_shape` as an independent second topic, and the frozen corroboration
rule then selected `REVIEW`. This is the mechanism working exactly as designed —
a sustained shape observed over time supplying corroboration that no single
event could.

*The 18 failures in `ABSTAIN`.* 17 carry `InsufficientBehaviouralHistory` and
role `Unknown`. These accounts have fewer than six observed events *in either
direction*. The architectural hope was explicit and is now falsified on this
window: `InsufficientBehaviouralHistory` counts inbound events too, so an
account with five inflows and one outflow would be behaviourally readable while
remaining semantically baseline-less. **These accounts do not have five
inflows.** They are genuinely first-seen. The mule shape the layer was built to
catch is not present in the data at the resolution this window provides.

*The 9 failures in `ALLOW`.* All carry `DistributionBehaviour` +
`FanOutDistribution` with role `Distributor`, on accounts with thousands of
observed events and up to 2,210 counterparties. Their semantic reading is
`RoutineValueTransfer` + `NormalOperationalBurst`: value inside the party's own
regime, tempo matching the party's own shape. Behaviourally they are
indistinguishable from a large payment operator, and the layer says so. One of
these ten (the success above) differed only in carrying
`ExpectedHighValueTransfer` instead of `RoutineValueTransfer`, which left the
novelty topic unqualified.

## Analysis: false positives

`comparison_report.md` lists the twenty most frequent unqualified-evidence
signatures with, for each, the mitigating object the catalog already declares as
its qualifier. Summary of the top causes:

| Rank | Unqualified risk evidence | FP | Declared qualifier that would have prevented it |
|---|---|---|---|
| 1 | `BEH-DistributionBehaviour` + `BEH-FanOutDistribution` + `SEM-R03-INFORMATIVE-NOVELTY` | 237 | `BEH-PayrollOperatorBehaviour`, `BEH-RoutinePayrollBehaviour`, `BEH-SupplierSettlementBehaviour` |
| 2 | `BEH-BurstActivityBehaviour` + `SEM-R02-TEMPO-REGIME-SHIFT` | 187 | `BEH-RoutinePayrollBehaviour`, `SEM-M03-OPERATIONAL-BURST` |
| 3 | `BEH-DistributionBehaviour` + `BEH-FanOutDistribution` + `SEM-R04-JURISDICTION` | 165 | `BEH-PayrollOperatorBehaviour`, `BEH-RoutinePayrollBehaviour` |
| 4 | `BEH-DistributionBehaviour` + `BEH-FanOutDistribution` + `SEM-R07-CASH-INSTRUMENT` | 155 | `BEH-ExpectedBusinessCycle`, `BEH-RoutinePayrollBehaviour` |
| 5 | `SEM-R04-JURISDICTION` + `SEM-R07-CASH-INSTRUMENT` | 147 | `BEH-ExpectedBusinessCycle`, `SEM-M05-ROUTINE-VALUE` |

**Which behaviour object caused it?** Overwhelmingly the fan-out family:
`DistributionBehaviour` and `FanOutDistribution` appear in the signature of the
largest false-positive groups, usually alongside a semantic topic
(`INFORMATIVE-NOVELTY`, `JURISDICTION`, `CASH-INSTRUMENT`) that the fan-out
turned into a second corroborating topic.

**Could a missing behaviour object have prevented it?** Yes, and the catalog
already names it. Of the top twenty signatures, nineteen have a declared
qualifier — nearly always `RoutinePayrollBehaviour`, `PayrollOperatorBehaviour`
or `ExpectedBusinessCycle`. Those three fired 20, 0 and 367 times respectively.
The catalog is not missing a concept; **the window is missing the evidence those
concepts need.** `RoutinePayrollBehaviour` needs five payments in one hour with
amount dispersion ≤ 0.25, and on a source whose amounts span nine orders of
magnitude, homogeneity is rare.

One signature of the twenty has **no declared qualifier**:
`BEH-DormantAccountActivation` + `BEH-ShellCompanyBehaviour` (22 false
positives). That is a genuine gap in the conflict catalog, and it is the one
place the analysis says to add a declared pair rather than to widen the window.

---

## Analysis: the semantic feature space

74 features, none of them a transaction column: 27 semantic-object
confidences, 25 behaviour-object confidences, 6 scenario confidences, 4 role
features, 6 lifecycle features, 6 evidence-state features. No amount, currency,
bank, payment format, hour or account identifier reaches the model.

The comparison with the previous experiment is informative, and it is not
flattering to the semantic representation. Both use the same routing rule and
almost the same routed volume:

| Model | Feature space | Routed | TP | FP |
|---|---|---|---|---|
| LightGBM (v1) | 18 raw transaction features | 49,310 | **18** | 12,841 |
| LightGBM (v2) | 74 semantic/behavioural features | 48,709 | 13 | 15,787 |
| XGBoost (v1) | raw | 49,310 | 2 | 444 |
| XGBoost (v2) | semantic | 48,709 | 1 | 2,328 |

**The semantic feature space is weaker than raw transaction features on this
task.** LightGBM loses 5 of 18 true positives and gains 2,946 false positives.
That is a real finding: the vocabularies do not yet carry everything the raw
amount, currency and format columns carry.

Feature importances say *what* the model leans on, which is the payoff of a
named feature space. All three models agree on the top of the list:

- `sem_InternalBookEntry` (XGBoost 0.46, CatBoost 35.7) — the structural
  distinction is the single most useful feature, confirming the v1 finding
- `evidence_mitigation_count` (XGBoost 0.09, CatBoost 12.9) — how much of the
  event has an ordinary explanation
- lifecycle features — `idle_minutes` and `age_minutes` are LightGBM's top two
- role features — `role_code`, `role_tenure_minutes`, `role_transition_count`
  all rank in the top 15 for every model

The model is using roles, lifecycles and behavioural mitigation. It is not using
them *enough* to match raw features.

## Cost

| Stage | Seconds (500k events) |
|---|---|
| Entity resolution (327,325 accounts) | 2.73 |
| Semantic feature pass (semantic + behaviour over 500k) | 61.97 |
| Frozen Runtime decisions (100k) | 3.89 |
| Behaviour runtime decisions (100k) | 11.81 |

Behavioural decision latency is 0.118 ms/event against the semantic layer's
0.058 ms and the frozen runtime's 0.039 ms. Peak process RSS 1.17 GB — the
temporal engine's constant-memory-per-account design holds: adding a full
behavioural state for 327,325 accounts cost nothing over the semantic-only
benchmark's 1.21 GB.

---

## What this establishes, and what it does not

**Established: the Runtime reasons about behaviour.** Every decision in arms 2–4
is stated in behaviour, role and scenario objects. A rationale reads
*"ALLOW: … Role: Distributor. Behaviour: DistributionBehaviour,
FanOutDistribution. Scenario: none matched."* No rule identifier and no
transaction field appears. Roles evolve — 16,218 transitions, each a causal
record naming the behaviour objects that forced it. Every behaviour object
carries its interval, its supporting semantic objects, its supporting
observations and its counter-evidence.

**Established: behavioural corroboration is a working mechanism.** The one
recovered true positive was recovered exactly as designed — a sustained shape
observed across hour buckets supplied a second independent topic to an unchanged
policy. 1,830 events moved on that mechanism.

**Established: the false-positive analysis is now actionable in the
vocabulary's own terms.** For nineteen of the twenty largest false-positive
groups, the catalog already names the object that would have prevented the
alert. That is a qualitatively different diagnostic from "rule R01 fired too
often".

**Not established: that behaviour improves detection on this window.** One true
positive for 1,824 additional false positives is a poor trade. Precision and F1
improve over both other unaided arms, but from a base where every arm is far
below any operational bar.

**Not established: that the semantic feature space is sufficient for ML.** It
measurably is not — 13 true positives against 18 for raw features.

**Falsified: the architectural hypothesis about `ABSTAIN`.** The layer was built
expecting that counting inbound events would let behaviour characterise accounts
the semantic layer abstains on. On this window it does not: 17 of the 18
abstained positives have fewer than six events in *either* direction.

## What the measurement says to do next — without tuning

1. **Widen the window.** This is now the dominant constraint, and the evidence is
   specific rather than general: 78% `InsufficientBehaviouralHistory`, the
   `days` and `weeks` horizons structurally empty, and nineteen of twenty
   false-positive groups blocked by mitigating behaviours
   (`RoutinePayrollBehaviour`, `PayrollOperatorBehaviour`) that a 5.6-hour window
   cannot supply. The source spans 18 days.
2. **Read behaviour from the beneficiary's perspective too.** `Collector`
   appeared 3 times and `CollectionBehaviour` never, because a role is computed
   for the originator. This is a design limitation, not a data one, and it
   directly blocks the collection half of the mule pattern.
3. **Add the one missing conflict pair.**
   `BEH-DormantAccountActivation` + `BEH-ShellCompanyBehaviour` is the only
   top-twenty false-positive signature with no declared qualifier.
4. **Enrich the feature space rather than replacing it.** The named features the
   models do lean on (structural distinction, mitigation count, role tenure,
   lifecycle age) suggest what is missing is magnitude-in-context — a *semantic*
   encoding of value relative to regime — rather than the raw columns themselves.

## Reproduce

```bash
python -m aml_runtime.behaviour_benchmark --transactions data/ibm_aml_data/HI-Small_Trans.csv --accounts data/ibm_aml_data/HI-Small_accounts.csv --output-dir artifacts/aml_behaviour_v1
```
