# Did the Semantic Runtime create information, or re-encode the raw features?

Critical analysis of `artifacts/semantic_vs_raw/`. Implementation:
`aml_runtime/semantic_vs_raw.py`. Every number below is read from
`comparison_results.json`, `feature_importance.csv`, `shap_summary.csv`,
`feature_groups.csv`, `feature_correlation.csv` or `ablation_results.csv`.

| | |
|---|---|
| **Purpose** | Decide whether the semantic runtime created information or merely re-encoded the raw transaction features. |
| **Inputs** | Evaluation set **E2**; three feature spaces (18 raw, 74 semantic, 92 combined) under one frozen CatBoost. |
| **Outputs** | `artifacts/semantic_vs_raw/` — results JSON, feature importance, SHAP summary, feature groups, correlations, ablations, ROC and PR curves. |
| **Guarantees** | Everything except the feature columns is frozen, including seed and hyperparameters. The semantic arm reproduces the window-study result exactly, which is the harness's own correctness check. SHAP is exact TreeSHAP over all 100,000 evaluation rows. |
| **Limitations** | Single-seed fits, and the seed is frozen, so no variance estimate exists. This measures a model's ability to exploit a representation, not the Runtime's. |


## Final conclusion

> **2. Semantic Runtime complements Raw features.**

Carried by three independent measurements, stated in full below: Raw + Semantic
strictly dominates both single spaces on all seven metrics; 38 of 74 semantic
features have max |r| < 0.3 against every raw feature; and dropping either space
from the union degrades it (dropping raw costs PR-AUC −0.044 and +1,801 FP,
dropping the semantic space costs PR-AUC −0.219 and +338 FP).

---

## Protocol and fidelity

Priming rows 0–4,977,237; ML training rows 4,577,237–4,977,237 (400,000 events,
436 positives); evaluation rows 4,977,237–5,077,237 (100,000 events,
**253 positives**). CatBoost: 200 iterations, depth 6, learning rate 0.05,
Logloss, seed 20260804, `scale_pos_weight` from the partition, 4 threads. Alert
threshold 0.50. Runtime, Semantic Context Layer, Behaviour Layer, policies,
routing, thresholds, CatBoost hyperparameters, chronological split, train/test
protocol and history window are unchanged.

**Reproduction check passed.** Experiment B had to reproduce the window study's
`ml_only/CatBoost` at window F. Measured recall `0.9011857707509882`, FP `6128`
— exact match to the published 0.901 / 6,128. `"matches": true` in
`comparison_results.json`. The `-raw` ablation independently reproduces the same
numbers a second time.

SHAP is exact TreeSHAP computed by CatBoost's own
`get_feature_importance(type="ShapValues")` over all 100,000 evaluation rows
(the `shap` package is not installed on this machine; this is the same algorithm
`shap.TreeExplainer` dispatches to for CatBoost models).

---

## Final table

| Features | Recall | Precision | F1 | FP | FN | ROC | PR |
|----------|-------:|----------:|---:|---:|---:|----:|---:|
| Raw (18) | 0.9012 | 0.046597 | 0.088613 | 4,665 | 25 | 0.9827 | 0.27426 |
| Semantic (74) | 0.9012 | 0.035872 | 0.068997 | 6,128 | 25 | 0.9771 | 0.44877 |
| **Raw + Semantic (92)** | **0.9130** | **0.050680** | **0.096030** | **4,327** | **22** | **0.9846** | **0.49274** |

| Arm | TP | FP | FN | TN | Alert rate | Train s | Predict s | Inference ms/event |
|---|---|---|---|---|---|---|---|---|
| Raw | 228 | 4,665 | 25 | 95,082 | 4.893% | 7.0 | 0.013 | 0.00013 |
| Semantic | 228 | 6,128 | 25 | 93,619 | 6.356% | 9.3 | 0.026 | 0.00026 |
| Raw + Semantic | 231 | 4,327 | 22 | 95,420 | 4.558% | 11.7 | 0.034 | 0.00034 |

Feature generation over the 5.08M-event pass: **raw 80.5 s, semantic 703.9 s**
(8.7× more expensive), single pass total 904.1 s. Peak RSS 2.26 GB.

---

## 1. Does Semantic outperform Raw?

**No at the operating point; yes on ranking quality.**

At threshold 0.50 the two spaces select *exactly the same* 228 true positives
and miss exactly the same 25 — identical TP and FN. They differ only in false
positives, and raw wins:

| | Raw | Semantic | Semantic − Raw |
|---|---|---|---|
| FP | 4,665 | 6,128 | **+1,463 (+31.4%)** |
| Precision | 0.046597 | 0.035872 | −0.010725 |
| F1 | 0.088613 | 0.068997 | −0.019616 |
| ROC-AUC | 0.9827 | 0.9771 | −0.0056 |
| **PR-AUC** | 0.27426 | **0.44877** | **+0.17451 (+63.6%)** |

The PR-AUC reversal is the substantive finding: across all thresholds the
semantic space ranks the 253 positives markedly higher, while at this particular
threshold it is less precise. 18 raw features and 74 semantic features arrive at
the same recall, so semantic does not outperform raw — but it is not inferior
either; it is differently shaped.

## 2. Does Raw + Semantic outperform both?

**Yes, strictly, on all seven metrics.**

| Metric | vs Raw | vs Semantic |
|---|---|---|
| TP | +3 (228 → 231) | +3 |
| FP | **−338** (4,665 → 4,327) | **−1,801** (6,128 → 4,327) |
| FN | −3 (25 → 22) | −3 |
| Recall | +0.0118 | +0.0118 |
| Precision | +0.004083 | +0.014808 |
| F1 | +0.007417 | +0.027033 |
| ROC-AUC | +0.0018 | +0.0075 |
| PR-AUC | **+0.21848 (+79.7%)** | +0.04397 (+9.8%) |

The union gains true positives *and* removes false positives simultaneously
against both parents. There is no metric on which either single space wins.

## 3. Which semantic concepts contribute the largest information gain?

Two measurements, and they disagree in an informative way.

**By attribution in the combined model** (`feature_groups.csv`), the semantic
space takes **39.3% of CatBoost importance and 36.7% of total |SHAP|** against
raw's 60.7% / 63.3%:

| Group | Features | Importance | Share | SHAP share |
|---|---|---|---|---|
| raw | 18 | 60.690 | 60.7% | 63.3% |
| context (`sem_*`) | 27 | 15.402 | 15.4% | 17.4% |
| lifecycle | 6 | 10.881 | 10.9% | 7.3% |
| behaviour (`beh_*`) | 25 | 4.811 | 4.8% | 4.4% |
| evidence | 6 | 4.387 | 4.4% | 4.1% |
| role | 4 | 3.389 | 3.4% | 3.0% |
| scenario | 6 | 0.440 | 0.4% | 0.4% |

Top attributed semantic features in the combined model (importance / SHAP rank):
`sem_EstablishedRelationship` (6.296, rank 3 in both), `lifecycle_idle_minutes`
(3.332), `lifecycle_age_minutes` (2.663), `sem_CounterpartyRegime` (2.153),
`lifecycle_buckets_active` (2.106), `role_tenure_minutes` (1.911),
`sem_ValueRegime` (1.122), `sem_CrossJurisdictionTransfer` (SHAP rank 9).

In the semantic-only model, `sem_EstablishedRelationship` alone carries **19.2%
of importance and 30.4% of total |SHAP|** — the single most valuable concept the
Semantic Context Layer produces.

**By drop-one ablation** (`ablation_results.csv`) — the causal measurement —
only two vocabularies have positive information gain given everything else:

| Dropped from Raw+Semantic | TP | FP | Recall | F1 | PR-AUC | ΔPR-AUC |
|---|---|---|---|---|---|---|
| *(nothing)* | 231 | 4,327 | 0.9130 | 0.09603 | 0.49274 | — |
| raw | 228 | 6,128 | 0.9012 | 0.06900 | 0.44877 | **−0.04397** |
| lifecycle | 234 | 4,742 | 0.9249 | 0.08950 | 0.42478 | **−0.06797** |
| evidence | 230 | 3,987 | 0.9091 | 0.10291 | 0.49298 | +0.00024 |
| context | 232 | 4,349 | 0.9170 | 0.09599 | 0.50588 | +0.01313 |
| scenario | 230 | 4,071 | 0.9091 | 0.10101 | 0.52431 | +0.03157 |
| behaviour | 228 | 3,864 | 0.9012 | 0.10495 | 0.52706 | +0.03432 |
| role | 228 | 4,245 | 0.9012 | 0.09648 | 0.55016 | +0.05742 |

**Answer: `lifecycle` is the only semantic vocabulary whose removal degrades
PR-AUC** (−0.068, the largest single drop in the table, larger than removing all
18 raw features). Every other semantic vocabulary can be removed from the union
with PR-AUC *rising*.

Two caveats stated rather than glossed: these are single-seed fits, and the seed
is frozen by the brief, so no variance estimate exists; and the ablations select
different models, so the recall/FP trade shifts as well as the ranking quality —
e.g. `-behaviour` drops 3 TP while removing 463 FP.

## 4. Which raw features become unnecessary after semantic reasoning?

`comparison_results.json → raw_features_demoted_by_semantic`. Importance
retained is combined ÷ raw-only:

| Raw feature | Importance raw-only | in Raw+Semantic | Rank | Retained |
|---|---|---|---|---|
| `same_bank` | 0.596 | 0.042 | 17 → 62 | **7%** |
| `originator_24h_count` | 5.671 | 0.611 | 7 → 35 | **11%** |
| `day_offset` | 0.868 | 0.107 | 15 → 56 | **12%** |
| `originator_prior_count` | 6.114 | 0.921 | 5 → 27 | **15%** |
| `amount_ratio` | 0.169 | 0.054 | 18 → 61 | 32% |
| `from_bank_code` | 9.329 | 3.240 | 3 → 8 | 35% |
| `hour` | 2.921 | 1.089 | 12 → 21 | 37% |

The displaced features are precisely the **history, velocity and degree
counters**, and `feature_correlation.csv` explains why causally:

- `lifecycle_observed_events` correlates with `originator_prior_count` at
  **r = 0.9999999**
- `lifecycle_distinct_counterparties` with `originator_prior_count` at
  **r = 0.99995**
- `sem_NonInformativeNovelty` with `new_beneficiary` at **r = 0.9934**

The semantic layer re-encodes those counters to seven decimal places, so the
model simply switches to the semantic copy. What survives are the raw features
the semantic vocabulary has no equivalent for: `payment_format_code` (88%
retained, rank 1 in both models, the single most important feature overall),
`log_amount_received` (96%), `pair_prior_count` (73%, rank 2), and
`new_beneficiary`, which *gains* importance (120%).

## 5. Which semantic features are never used?

**11 of 74** have exactly zero CatBoost importance **and** zero mean |SHAP| in
both the semantic-only and the combined model:

| Never used | Group |
|---|---|
| `beh_MoneyMuleBehaviour` | behaviour |
| `beh_HighVelocityLayering` | behaviour |
| `beh_PayrollOperatorBehaviour` | behaviour |
| `beh_RoutinePayrollBehaviour` | behaviour |
| `beh_SettlementHubBehaviour` | behaviour |
| `beh_FanOutDistribution` | behaviour |
| `beh_RelationshipGrowthBehaviour` | behaviour |
| `scn_LayeringAttempt` | scenario |
| `scn_DormancyAfterOutflow` | scenario |
| `sem_BookkeepingAccount` | context |
| `sem_CoverageGap` | context |

A further **12 of 74 are constant across the whole evaluation set** and
therefore carry no information by construction.

Seven of the eleven are behaviour objects, and they include the two flagship
risk constructs of the architecture — the `MoneyMuleBehaviour` composite and the
`LayeringAttempt` scenario. The concepts the design invested most in are the
ones the model finds least usable. This is consistent with the behaviour-layer
report's finding that those types fire rarely or never on this window.

## 6. Does the Semantic Runtime add information, or merely re-encode?

**Both, and the proportions are measurable.**

*Evidence of re-encoding* (`feature_correlation.csv`, max |Pearson r| of each
semantic feature against any raw feature, computed on the evaluation set):

| max &#124;r&#124; band | Semantic features |
|---|---|
| ≥ 0.9 | 10 |
| 0.7 – 0.9 | 8 |
| 0.5 – 0.7 | 10 |
| 0.3 – 0.5 | 8 |
| **< 0.3** | **38** |

Ten features are near-duplicates of a raw column, three of them to four decimal
places. That is literal re-encoding and it is why the history counters in
question 4 got displaced.

*Evidence of new information:*

1. **38 of 74 semantic features have max |r| < 0.3 against every raw feature.**
   More than half the space is not a linear re-encoding of anything raw.
2. **Adding semantic to raw raises PR-AUC by 79.7%** (0.27426 → 0.49274) and
   simultaneously buys +3 TP and −338 FP. A gradient-boosted model already
   holding all 18 raw columns cannot gain 80% ranking quality from a re-encoding
   of those same columns.
3. **Removing raw from the union costs PR-AUC −0.044 and +1,801 FP**, so the
   semantic space does not subsume raw either. `payment_format_code` — rank 1 in
   both models — has no semantic equivalent at all.

The two spaces are non-nested: each holds signal the other lacks.

---

## What this means for the project

**The honest reading is narrower than the conclusion sounds.** The complement is
real but small at the operating point: +3 true positives and −338 false
positives over raw alone, for 8.7× the feature-generation cost (703.9 s versus
80.5 s per 5.08M events). The large win is in ranking (PR-AUC +80%), which
matters if alerts are triaged by score rather than by a fixed threshold.

**The value is concentrated in the concepts nearest the data, not the ones
furthest from it.** `sem_EstablishedRelationship` and the lifecycle counters
carry the semantic space; the behavioural composites and scenarios carry nothing
measurable. Ranked by attribution the ordering is context (15.4%) > lifecycle
(10.9%) > behaviour (4.8%) > evidence (4.4%) > role (3.4%) > scenario (0.4%) —
almost exactly the inverse of the order in which the architecture was built.

**This does not measure the Runtime.** Every number here is CatBoost's ability
to exploit a representation. The window study already established that the
declared policy converts the same representation into recall 0.150 where the
model reaches 0.901. This experiment says the representation is worth having;
it says nothing about the policy that consumes it, which remains the binding
constraint identified in [`window-scaling.md`](window-scaling.md).

## Reproduce

```bash
python -m aml_runtime.semantic_vs_raw --transactions data/ibm_aml_data/HI-Small_Trans.csv --accounts data/ibm_aml_data/HI-Small_accounts.csv --output-dir artifacts/semantic_vs_raw
```
