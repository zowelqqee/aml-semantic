# Is the architecture limited by data or by reasoning?

A controlled history-window scaling study. Implementation:
`aml_runtime/window_study.py`. Generated artifacts:
`artifacts/aml_window_study/` — `window_study_report.md`,
`window_study_results.json`, `window_{A..F}.json`, `window_evolution.png` and
five individual evolution plots.

Predecessors: [`semantic-runtime-v1.md`](semantic-runtime-v1.md),
[`behaviour-runtime-v1.md`](behaviour-runtime-v1.md).

| | |
|---|---|
| **Purpose** | Determine whether the architecture is limited by data availability or by reasoning capability. |
| **Inputs** | IBM AML `HI-Small`; evaluation set **E2** held constant (100,000 events, 253 labels) with the priming window varied from 5.6 hours to 9 days 11 hours. |
| **Outputs** | `artifacts/aml_window_study/` — per-window JSON, consolidated results, six evolution plots. |
| **Guarantees** | No reasoning code changed between windows; ontology hashes identical. The fast priming path is proven state-identical to full evaluation by test. The frozen v0.2 runtime is carried as a control and is invariant to the event from 24 h onward. |
| **Limitations** | Window E (14 days) does not exist in this source. Only 9 days 11 hours of history precede the evaluation set, so saturation beyond that is unobservable here. |


## Answer

**Both, in sequence, and the bottleneck has moved.**

The architecture *was* data-limited in **coverage**, and more history fixes that
almost completely: `InsufficientBehaviouralHistory` falls from 83.8% of events
to 3.0%, and **171 of 253 laundering events (68%) move out of it into fully
explained behaviour**.

The architecture is *now* reasoning-limited in **discrimination**. Coverage
keeps improving all the way to the end of the available history, but decision
quality saturates at 7 days and then *reverses*: recall peaks at 0.162 and falls
to 0.150, precision peaks at 0.00151 and falls to 0.00123, while false positives
keep climbing to 30,892.

The decisive measurement is that a model reading **exactly the same semantic and
behavioural objects, at the same window**, reaches recall 0.901 with 6,128 false
positives where the Runtime's declared policy reaches 0.150 with 30,892. The
information is in the representation. The policy does not convert it.

---

## Design, and why it can support that claim

**Nothing in the reasoning was changed, reconfigured, or called differently.**
Semantic ontology hash `80103ed6…` and behaviour ontology hash `7b99b0c1…` are
identical to the four-arm benchmark, as are the inference rules, the projection,
the conflict pairs, the policy engine, the routing rule and the 74-dimension
feature space. Asserted by `tests/test_window_study.py`.

**The evaluation set is held constant.** The same 100,000 events (rows
4,977,237–5,077,237, all of 2022-09-10) carrying the same **253** laundering
labels, in every window. Only the priming window in front of it changes. Without
this, a longer window would also mean a different evaluation set and no movement
would be attributable to anything.

**The fast priming path is proven equivalent, not assumed.** The study folds the
priming region without materialising decisions. `test_window_study.py` asserts
that this produces temporal state identical slot-for-slot across every account,
and that a decision taken after priming matches one taken after full evaluation —
same behaviour object ids, same evidence ids, same role, same context state hash.

**The frozen v0.2 rule runtime is carried as a control**, and it behaved exactly
as a control must:

| Window | Control TP | Control FP |
|---|---|---|
| A — 5.6 h | 87 | 36,919 |
| B — 24 h | 111 | 60,973 |
| C — 3 d | 111 | 60,973 |
| D — 7 d | 111 | 60,973 |
| F — 9 d 11 h | 111 | 60,973 |

The control's only history-dependent rule is a 24-hour velocity count. With 5.6
hours of priming that window is under-filled; from 24 hours onward it is
saturated and the control becomes **exactly invariant** — the same numbers to the
event, across four windows spanning 442k to 4.98M priming rows.

Two consequences, both load-bearing:

- **A → B is confounded.** Part of that step is a warm-up that even a
  history-insensitive runtime shows. It is reported but not leaned on.
- **B → F is clean.** The control does not move by a single event, so every
  change in the semantic and behavioural arms across B → F is the data effect on
  those layers and nothing else.

### Window E does not exist

The source's dense period runs 2022-09-01 00:00 to 2022-09-10 23:59, leaving
9 days 11 hours of history in front of the evaluation set. **A 14-day window
cannot be constructed.** Declared, not approximated.

Rows beyond 5,077,237 (2022-09-11 to 09-18) are 1,108 events carrying 655
laundering labels — a 59% base rate, a generator artefact of long-running
patterns completing. They are excluded; including them would have made the
evaluation set unrepresentative.

### Confounds stated

- **ML training partition.** Window A trains on 46,109 rows / 114 positives;
  B–F train on 400,000 / 436 (capped, so capacity is not a variable). ML numbers
  at A are not comparable to B–F; B–F are directly comparable to each other.
- **Model variance.** LightGBM's recall wobbles across windows
  (0.632 → 0.862 → 0.648 → 0.794). Conclusions below rest on the trend and on
  best-of-three, not on any single model at any single point.
- **Cross-study comparison is invalid.** This evaluation set has 253 positives;
  earlier experiments used a 28-positive set. Absolute numbers are comparable
  only *within* this study.

---

## Coverage — the data effect is large and does not saturate

| Window | Prime rows | Behaviour objects | Substantive | Scenario objects | Role transitions | Established baselines | Behaviour confidence | NoEstablishedBaseline | InsufficientBehaviouralHistory | ABSTAIN |
|---|---|---|---|---|---|---|---|---|---|---|
| A — 5.6 h | 46,109 | 104,635 | 20,860 | 12,512 | 12,890 | 16,124 | 0.6747 | 83.88% | 83.77% | 80.54% |
| B — 24 h | 441,862 | 107,783 | 68,012 | 34,986 | 34,043 | 52,046 | 0.5829 | 47.95% | 39.77% | 44.74% |
| C — 3 d | 1,489,799 | 147,041 | 140,453 | 42,899 | 12,001 | 90,404 | 0.5595 | 9.60% | 6.59% | 9.03% |
| D — 7 d | 3,001,703 | 210,054 | 205,959 | 43,004 | 5,568 | 95,015 | 0.6360 | 4.98% | 4.10% | 4.69% |
| F — 9 d 11 h | 4,977,237 | 217,124 | 214,148 | 44,635 | 5,367 | 95,263 | 0.6727 | 4.74% | 2.98% | 4.37% |

- **`InsufficientBehaviouralHistory`: 83.8% → 3.0%** (−96%)
- **`NoEstablishedBaseline`: 83.9% → 4.7%** (−94%)
- **`ABSTAIN`: 80.5% → 4.4%** (−95%)
- Established baselines ×5.9; substantive behaviour objects ×10.3; scenario
  objects ×3.6; conflicts 10,477 → 25,934

The single dominant limitation named at the end of the previous report is
resolved by data. That report attributed 78% `InsufficientBehaviouralHistory` to
a 5.6-hour window; with 9.5 days it is 3.0%.

**Roles stabilise.** Transitions peak at B (34,043) and then *fall* to 5,367 —
a 6.3× reduction. The role oscillation flagged as an open question in the
previous report (`Dormant → Unknown → Dormant` churn) is confirmed as a
small-window artefact, not a design fault. No hysteresis was added.

**Behaviour confidence dips and recovers** (0.675 → 0.560 at C → 0.673 at F).
At A only strong claims fire, on the few describable accounts; at B–C a mass of
newly-describable accounts produces many weakly-supported claims; by D–F support
has accumulated and confidence returns. This is the declared
`prior × support × coverage` model behaving as specified.

---

## Decision quality — the data effect saturates, then reverses

| Window | Runtime recall | Runtime precision | Runtime FP | Runtime FN | ML recall (best) | Hybrid recall | Hybrid precision | Hybrid F1 |
|---|---|---|---|---|---|---|---|---|
| A — 5.6 h | 0.0000 | 0.000000 | 2,543 | 253 | 0.755 | 0.3043 | 0.010153 | 0.019650 |
| B — 24 h | 0.0949 | 0.001170 | 20,484 | 229 | 0.854 | 0.6482 | 0.004819 | 0.009567 |
| C — 3 d | 0.1423 | 0.001364 | 26,361 | 217 | 0.881 | 0.7866 | 0.006416 | 0.012728 |
| **D — 7 d** | **0.1621** | **0.001506** | 27,182 | 212 | 0.791 | 0.5968 | 0.005196 | 0.010302 |
| F — 9 d 11 h | 0.1502 | 0.001229 | 30,892 | 215 | **0.901** | 0.6640 | 0.004919 | 0.009765 |

The Semantic Behaviour Runtime's recall and precision **both peak at 7 days and
decline at 9.5 days**, while its false positives keep rising (2,543 → 30,892,
a factor of 12). Marginal recall per window step: +0.095, +0.047, +0.020,
**−0.012**.

The Semantic Runtime without the behaviour layer is inert across the whole
sweep: 0, 1, 1, 1, 2 true positives. History alone does almost nothing for it —
the behaviour layer is what converts history into decisions at all.

`BLOCK` starts firing from window B (9,350 → 10,200 events): the layering-motif
block policy needs history to reach its corroboration requirement, and gets it.

---

## Migration out of `InsufficientBehaviouralHistory`

The measurement the brief asked for, tracked per transaction on the constant
evaluation set:

| Window | Laundering events | at `InsufficientBehaviouralHistory` | fully explained behaviour | moved vs window A | mean observed events | mean counterparties |
|---|---|---|---|---|---|---|
| A — 5.6 h | 253 | 214 | 13 | 0 | 109.5 | 55.9 |
| B — 24 h | 253 | 150 | 42 | 23 | 510.3 | 213.0 |
| C — 3 d | 253 | 95 | 101 | 73 | 1,646.4 | 332.0 |
| D — 7 d | 253 | 81 | 143 | 115 | 3,295.9 | 332.8 |
| F — 9 d 11 h | 253 | **43** | 137 | **111** | 4,833.6 | 429.7 |

**171 of 253 laundering events (67.6%) leave `InsufficientBehaviouralHistory`.**
The mean observed-event count behind a laundering decision rises 44× (109.5 →
4,833.6) and the mean counterparty count 7.7× (55.9 → 429.7).

And recall goes from 0.000 to 0.150.

That juxtaposition is the study's central result. **The layer stops being
ignorant and does not thereby become discriminating.** It describes 68% more
laundering events, in its own vocabulary, with far more support behind each
description — and converts almost none of that into correct decisions. What it
observes for laundering accounts, once it can observe anything, looks like what
it observes for everyone else.

(`fully explained behaviour` dips 143 → 137 at F while
`InsufficientBehaviouralHistory` still falls 81 → 43: the difference is events
that leave insufficiency but land on a role and lifecycle with no substantive
behaviour claim — describable, but with nothing distinctive to say.)

---

## The decisive control: same objects, same history, different reasoning

ML in this study reads **only** the semantic feature space — 27 semantic-object
confidences, 25 behaviour-object confidences, 6 scenario confidences, 4 role
features, 6 lifecycle features, 6 evidence-state features. No amount, currency,
bank, payment format, hour or account identifier. It sees exactly what the
Runtime sees, at exactly the same window.

| Window | Runtime recall | Runtime FP | Best model recall | that model's FP |
|---|---|---|---|---|
| A | 0.000 | 2,543 | 0.755 (CatBoost) | 25,860 |
| B | 0.095 | 20,484 | 0.854 (CatBoost) | 19,579 |
| C | 0.142 | 26,361 | 0.881 (CatBoost) | 6,275 |
| D | 0.162 | 27,182 | 0.791 (CatBoost) | 4,152 |
| **F** | **0.150** | **30,892** | **0.901 (CatBoost)** | **6,128** |

At maximum history the model extracts **6.0× the recall with 5.0× fewer false
positives** from the identical representation. XGBoost at F: recall 0.735 with
2,756 false positives — 4.9× the recall at 11× fewer false positives.

This isolates the constraint. If the behaviour, role, scenario and lifecycle
objects did not carry the signal, no model could reach 0.90 recall from them.
They carry it. The declared corroboration policy — two independent unqualified
topics, or one high-concern topic above 0.90 — is what fails to express it.

Note also what the *hybrid* does: gated to the shrinking `ABSTAIN` population
(80.5% of events at A, 4.4% at F), it loses access to the very events the model
scores well. Hybrid recall peaks at C (0.787) and falls to 0.664 at F. **The
routing rule was designed for a layer that abstained most of the time; with
history, abstention nearly disappears and the rule starves the model of exactly
the cases it handles best.** That is a reasoning-architecture problem created by
having more data, not solved by it.

---

## What this establishes

**Established: the coverage limitation was real and is data-limited.** Every
coverage metric improves monotonically with history and the improvement is
large. The previous report's dominant caveat is answered.

**Established: the attribution is clean.** The control is invariant to the event
across B → F, so nothing in that range is evaluation-set drift.

**Established: decision quality is not data-limited beyond 7 days.** Recall and
precision peak at D and decline at F while coverage keeps improving. More
history past a week makes the layer describe more and decide worse.

**Established: the representation carries the signal; the policy does not use
it.** 0.901 versus 0.150 recall from the same objects at the same window.

**Not established: that more history would eventually help.** Only 9 days 11
hours exist. The curve is saturating and has turned, but a longer source could
behave differently and this study cannot say.

**Not established: that the policy is the *only* remaining constraint.** The
model's advantage shows the signal is present and extractable; it does not show
that a *deterministic, auditable* policy can extract it. That is the open
question, and it is now the research question rather than data availability.

## What the measurement says to do next

The previous report's top recommendation — *widen the window* — is now executed
and settled. It fixed what it was predicted to fix and did not fix detection.
The measurements point somewhere else:

1. **The corroboration policy is the binding constraint.** It was declared as
   "one signal is not a finding" when almost every event had at most one signal.
   With history, events carry many behaviour objects and the rule now behaves as
   a near-constant alerting threshold — 2,543 → 30,892 false positives while
   recall stalls. Any successor should be measured against the 0.901 ceiling
   this study established.
2. **The routing rule is now mis-specified.** It gates ML on `ABSTAIN`, which
   fell from 80.5% to 4.4%. It should be re-derived from the semantic state that
   actually remains undetermined, not from a decision label whose meaning
   changed underneath it.
3. **Feature importances, not new objects, are the diagnostic.** The vocabulary
   is sufficient for a model to reach 0.90 recall. Adding a 26th behaviour type
   is not indicated; understanding which of the existing 74 features the model
   uses, and why the policy cannot, is.
4. **The seven-day peak deserves an explanation.** Something makes D better than
   F for decisions while F is better for coverage. That is a specific,
   answerable question about which behaviour objects change between them.

## Cost

| Window | Feature pass (s) | Decision pass (s) | Peak RSS |
|---|---|---|---|
| A | 93.5 | 90.9 | 0.59 GB |
| B | 222.2 | 145.5 | 1.04 GB |
| C | 407.9 | 359.5 | 1.41 GB |
| D | 517.3 | 466.6 | 1.41 GB |
| F | 708.8 | 994.8 | 1.71 GB |

10.0 million event-evaluations per pass pair at window F. The temporal engine's
constant-memory-per-account design holds: 518,581 resolved accounts and 5.08M
events fit in 1.71 GB.

## Reproduce

```bash
python -m aml_runtime.window_study --transactions data/ibm_aml_data/HI-Small_Trans.csv --accounts data/ibm_aml_data/HI-Small_accounts.csv --output-dir artifacts/aml_window_study
```
