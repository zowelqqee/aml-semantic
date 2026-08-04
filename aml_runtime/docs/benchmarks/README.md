# Benchmarks — semantic decision runtime

Every number on this page was produced by a committed module, written to
`artifacts/`, and is reproducible with the command shown beside it. Nothing is
estimated, extrapolated or rounded in the runtime's favour.

| | |
|---|---|
| **Purpose** | One page holding every measured result for the AML track, with the protocol each number came from. |
| **Inputs** | IBM AML `HI-Small` (5,078,345 transactions, 518,581 accounts). |
| **Outputs** | `artifacts/aml_semantic_v1/`, `artifacts/aml_behaviour_v1/`, `artifacts/aml_window_study/`, `artifacts/semantic_vs_raw/`. |
| **Guarantees** | Labels enter only model fitting and post-decision evaluation. Every semantic and behavioural constant was declared before execution; none was selected against evaluation labels. Runs are deterministic and were verified identical across repeats. |
| **Limitations** | One dataset, one generator, one label definition. Absolute rates are not comparable across experiments that use different evaluation sets — the set is named in every table below. |

---

## Two evaluation sets

Results are **only comparable within a set**. Both are stated everywhere.

| Set | Rows | Events | Positives | Base rate | Used by |
|---|---|---|---|---|---|
| **E1** | 400,000–500,000 | 100,000 | 28 | 0.028% | semantic-runtime-v1, behaviour-runtime-v1 |
| **E2** | 4,977,237–5,077,237 | 100,000 | 253 | 0.253% | window-scaling, semantic-vs-raw |

E2 is the last 100,000 events of the source's dense period. Rows beyond it
(2022-09-11 to 09-18) hold 1,108 events with 655 laundering labels — a 59% base
rate, a generator artefact — and are excluded from every experiment.

---

## 1 · Does semantic reasoning reduce false positives?

**Set E1.** [Full report →](semantic-runtime-v1.md) · `artifacts/aml_semantic_v1/`

| Arm | TP | FP | Recall | Precision | Alert rate | ML calls |
|---|---:|---:|---:|---:|---:|---:|
| Runtime (frozen v0.2) | 14 | 44,680 | 0.5000 | 0.000313 | 44.69% | 0 |
| **Semantic Runtime** | 0 | **382** | 0.0000 | — | **0.38%** | 0 |
| ML only — LightGBM | 20 | 20,120 | 0.7143 | 0.000993 | 20.14% | 100,000 |
| Runtime + LightGBM | 28 | 53,767 | 1.0000 | 0.000520 | 53.80% | 100,000 |
| Semantic + LightGBM | 18 | 12,841 | 0.6429 | 0.001400 | 12.86% | **49,310** |

**−99.1% false positives** against the rule-first runtime, and ML inference
volume halved. Precision improves 2.7×–11.9× in every paired comparison.
**Recall falls to zero unaided**; 18 of the 28 positives land in `ABSTAIN`, and
`Semantic + LightGBM` recovers 18 of those 18 — the whole routable ceiling.

The largest single finding is a missing concept, not a threshold: 34,050 of the
100,000 events are self-postings, which the rule vocabulary had no way to
express and therefore processed as inter-party transfers.

```bash
python -m aml_runtime.semantic_benchmark
```

## 2 · Does reasoning about behaviour over time change the decision?

**Set E1.** [Full report →](behaviour-runtime-v1.md) · `artifacts/aml_behaviour_v1/`

| Arm | TP | FP | Precision | F1 | Alert rate |
|---|---:|---:|---:|---:|---:|
| Runtime (frozen v0.2) | 14 | 44,680 | 0.000313 | 0.000626 | 44.69% |
| Semantic Runtime | 0 | 382 | 0 | 0 | 0.38% |
| **Semantic Behaviour Runtime** | **1** | 2,206 | **0.000453** | **0.000895** | 2.21% |
| Semantic Behaviour + LightGBM | 13 | 15,787 | 0.000823 | 0.001643 | 15.80% |

Best precision and F1 of the three unaided arms. Layer output over the horizon:
**98,913 behaviour objects · 16,218 role transitions · 28,428 scenario
detections · 10,603 conflicts**.

The one recovered true positive was recovered by the intended mechanism — a
sustained shape observed across hour buckets supplied a second independent topic
to an unchanged policy. It cost 1,824 additional false positives.

For 19 of the 20 largest false-positive groups the catalog already names the
object that would have suppressed the alert. The catalog is not missing a
concept; the window is missing the evidence for it.

```bash
python -m aml_runtime.behaviour_benchmark
```

## 3 · Is the architecture limited by data or by reasoning?

**Set E2, held constant across every window.** Only the history in front of it
varies. [Full report →](window-scaling.md) · `artifacts/aml_window_study/`

The frozen v0.2 runtime is the control. Its only history-dependent rule is a
24-hour velocity count, and from window B onward it is **invariant to the
event** — so movement from B to F is the data effect and nothing else.

| Window | History | InsufficientBehaviouralHistory | ABSTAIN | Runtime recall | Runtime FP | Control FP |
|---|---|---:|---:|---:|---:|---:|
| A | 5.6 h | 83.77% | 80.54% | 0.0000 | 2,543 | 36,919 |
| B | 24 h | 39.77% | 44.74% | 0.0949 | 20,484 | 60,973 |
| C | 3 d | 6.59% | 9.03% | 0.1423 | 26,361 | 60,973 |
| **D** | **7 d** | 4.10% | 4.69% | **0.1621** | 27,182 | 60,973 |
| F | 9 d 11 h | **2.98%** | **4.37%** | 0.1502 | 30,892 | 60,973 |

> Window E (14 days) **does not exist**. The dense period leaves 9 days 11 hours
> in front of the evaluation set. Declared, not approximated.

**Coverage is data-limited and the limit is removable**: `InsufficientBehaviouralHistory`
falls 83.8% → 3.0%, and **171 of 253 laundering events (68%) leave it** for
fully explained behaviour.

**Discrimination is not**: recall and precision peak at 7 days and *decline* at
9.5 days while coverage keeps improving. Marginal recall per step:
+0.095, +0.047, +0.020, **−0.012**.

The decisive contrast — same objects, same window, different reasoning:

| At window F | Recall | FP |
|---|---:|---:|
| Semantic Behaviour Runtime (declared policy) | 0.150 | 30,892 |
| **CatBoost over the identical feature space** | **0.901** | **6,128** |

```bash
python -m aml_runtime.window_study
```

## 4 · Did the semantic layer create information, or re-encode raw features?

**Set E2.** [Full report →](semantic-vs-raw.md) · `artifacts/semantic_vs_raw/`

Three feature spaces, one frozen CatBoost, everything else identical.

| Features | Recall | Precision | F1 | FP | FN | ROC | PR |
|----------|-------:|----------:|---:|---:|---:|----:|---:|
| Raw (18) | 0.9012 | 0.046597 | 0.088613 | 4,665 | 25 | 0.9827 | 0.27426 |
| Semantic (74) | 0.9012 | 0.035872 | 0.068997 | 6,128 | 25 | 0.9771 | 0.44877 |
| **Raw + Semantic (92)** | **0.9130** | **0.050680** | **0.096030** | **4,327** | **22** | **0.9846** | **0.49274** |

**Conclusion: the semantic runtime complements raw features.** The union
strictly dominates both on all seven metrics — **+3 TP and −338 FP
simultaneously** against raw, PR-AUC **+79.7%**.

Neither space subsumes the other: 38 of 74 semantic features have max |r| < 0.3
against every raw feature, while removing raw from the union costs −0.044 PR-AUC
and +1,801 FP.

Attribution splits **39.3% semantic / 60.7% raw** by CatBoost importance
(36.7% / 63.3% by SHAP), ordered context 15.4% > lifecycle 10.9% >
behaviour 4.8% > evidence 4.4% > role 3.4% > scenario 0.4% — close to the
inverse of the order in which the architecture was built.

Honest counterweights, both measured:

- **11 of 74 semantic features are never used** in either model, including the
  `MoneyMuleBehaviour` composite and the `LayeringAttempt` scenario.
- Semantic feature generation costs **703.9 s against 80.5 s** for raw over the
  same 5.08M events — 8.7× — for +3 TP and −338 FP at the operating point.

```bash
python -m aml_runtime.semantic_vs_raw
```

---

## Cost

| Stage | Events | Wall time | Peak RSS |
|---|---:|---:|---:|
| Entity resolution (518,581 accounts) | — | 2.7 s | — |
| Frozen runtime decisions | 100,000 | 3.9 s | — |
| Semantic runtime decisions | 100,000 | 5.8 s | — |
| Behaviour runtime decisions | 100,000 | 11.8 s | — |
| Semantic + behaviour feature pass | 5,077,237 | 704 s | 1.71 GB |
| Full window study (5 windows) | 19.7 M | ~62 min | 1.71 GB |

Decision latency: frozen 0.039 ms/event · semantic 0.058 · behaviour 0.118.

## Reproducing

```bash
# 3.13 with numpy, scikit-learn, xgboost, lightgbm, catboost, matplotlib
python -m pytest tests/ -q          # 87 tests
python -m aml_runtime.semantic_benchmark
python -m aml_runtime.behaviour_benchmark
python -m aml_runtime.window_study
python -m aml_runtime.semantic_vs_raw
```

The window study and the raw-vs-semantic experiment expect the chronologically
sorted input at `artifacts/aml_ml_benchmark/HI-Small_Trans.chronological.csv`,
which `aml_runtime.ml_benchmark` produces on first run.
