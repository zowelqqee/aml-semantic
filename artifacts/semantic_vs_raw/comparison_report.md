# Semantic versus Raw — did the Semantic Runtime create information?

aml-semantic-vs-raw/1.0

## Protocol

Priming rows 0..4,977,237; ML training rows 4,577,237..4,977,237 (436 positives); evaluation rows 4,977,237..5,077,237 (100,000 events, 253 positives).

Runtime, Semantic Context Layer, Behaviour Layer, policies, routing, thresholds, CatBoost hyperparameters, chronological split, train/test protocol and history window are unchanged.

**Reproduction check.** Semantic-only must reproduce the window study's `ml_only/CatBoost` at window F (recall 0.901, FP 6,128). Measured: recall 0.9012, FP 6,128 — **matches**.

## Final table

| Features | Recall | Precision | F1 | FP | FN | ROC | PR |
|---|---|---|---|---|---|---|---|
| Raw | 0.9012 | 0.046597 | 0.088613 | 4,665 | 25 | 0.9827 | 0.27426 |
| Semantic | 0.9012 | 0.035872 | 0.068997 | 6,128 | 25 | 0.9771 | 0.44877 |
| Raw + Semantic | 0.9130 | 0.050680 | 0.096030 | 4,327 | 22 | 0.9846 | 0.49274 |

## Full metrics

| Arm | Features | TP | FP | FN | TN | Alert rate | Train s | Predict s | Inference ms/event | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 18 | 228 | 4665 | 25 | 95082 | 0.04893 | 7.0 | 0.01 | 0.00013 | 2,261,696,512 |
| semantic | 74 | 228 | 6128 | 25 | 93619 | 0.06356 | 9.3 | 0.03 | 0.00026 | 2,261,696,512 |
| raw_plus_semantic | 92 | 231 | 4327 | 22 | 95420 | 0.04558 | 11.7 | 0.03 | 0.00034 | 2,261,696,512 |

Feature generation: raw 80.5 s, semantic 703.9 s, single pass total 904.1 s.

## Importance by vocabulary

| Model | Group | CatBoost importance | Importance share | SHAP share |
|---|---|---|---|---|
| semantic | behaviour | 8.3125 | 0.0831 | 0.0721 |
| semantic | context | 48.1410 | 0.4814 | 0.6391 |
| semantic | evidence | 9.7074 | 0.0971 | 0.0743 |
| semantic | lifecycle | 22.4691 | 0.2247 | 0.1432 |
| semantic | role | 10.2954 | 0.1030 | 0.0653 |
| semantic | scenario | 1.0745 | 0.0107 | 0.0060 |
| semantic | semantic_total | 100.0000 | 1.0000 | 1.0000 |
| semantic | raw_total | 0.0000 | 0.0000 | 0.0000 |
| raw_plus_semantic | behaviour | 4.8113 | 0.0481 | 0.0437 |
| raw_plus_semantic | context | 15.4020 | 0.1540 | 0.1741 |
| raw_plus_semantic | evidence | 4.3874 | 0.0439 | 0.0413 |
| raw_plus_semantic | lifecycle | 10.8806 | 0.1088 | 0.0732 |
| raw_plus_semantic | raw | 60.6902 | 0.6069 | 0.6334 |
| raw_plus_semantic | role | 3.3886 | 0.0339 | 0.0303 |
| raw_plus_semantic | scenario | 0.4399 | 0.0044 | 0.0039 |
| raw_plus_semantic | semantic_total | 39.3098 | 0.3931 | 0.3666 |
| raw_plus_semantic | raw_total | 60.6902 | 0.6069 | 0.6334 |

## Ablations — drop one vocabulary from Raw + Semantic

| Dropped | Features | TP | FP | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|---|---|---|
| raw | 74 | 228 | 6,128 | 0.9012 | 0.035872 | 0.068997 | 0.44877 |
| context | 65 | 232 | 4,349 | 0.9170 | 0.050644 | 0.095987 | 0.50588 |
| behaviour | 67 | 228 | 3,864 | 0.9012 | 0.055718 | 0.104948 | 0.52706 |
| scenario | 86 | 230 | 4,071 | 0.9091 | 0.053476 | 0.101010 | 0.52431 |
| role | 88 | 228 | 4,245 | 0.9012 | 0.050973 | 0.096488 | 0.55016 |
| lifecycle | 86 | 234 | 4,742 | 0.9249 | 0.047026 | 0.089501 | 0.42478 |
| evidence | 86 | 230 | 3,987 | 0.9091 | 0.054541 | 0.102908 | 0.49298 |

## Most raw-correlated semantic features

| Semantic feature | Group | max |r| with any raw feature | Closest raw feature |
|---|---|---|---|
| lifecycle_observed_events | lifecycle | 1.0000 | originator_prior_count |
| lifecycle_distinct_counterparties | lifecycle | 1.0000 | originator_prior_count |
| beh_RapidLayeringBehaviour | behaviour | 0.9973 | from_bank_code |
| sem_NonInformativeNovelty | context | 0.9934 | new_beneficiary |
| sem_DistributionNode | context | 0.9904 | from_bank_code |
| beh_CashConcentrationBehaviour | behaviour | 0.9814 | from_bank_code |
| sem_NormalOperationalBurst | context | 0.9786 | from_bank_code |
| lifecycle_buckets_active | lifecycle | 0.9693 | from_bank_code |
| beh_RelationshipCollapseBehaviour | behaviour | 0.9612 | from_bank_code |
| beh_DistributionBehaviour | behaviour | 0.9336 | from_bank_code |
| sem_UnscaledValue | context | 0.8737 | new_beneficiary |
| sem_NoEstablishedBaseline | context | 0.8697 | new_beneficiary |
| sem_ValueRegime | context | 0.8427 | new_beneficiary |
| sem_TempoRegime | context | 0.8106 | new_beneficiary |
| sem_CounterpartyRegime | context | 0.7604 | from_bank_code |

## Files

- `comparison_results.json`, `feature_importance.csv`, `shap_summary.csv`
- `feature_groups.csv`, `feature_correlation.csv`, `ablation_results.csv`
- `roc_curves.png`, `pr_curves.png`

## Critical analysis and final conclusion

The six analysis questions and the single final conclusion are answered from these files in [`docs/semantic_vs_raw_analysis.md`](../../docs/benchmarks/semantic-vs-raw.md).
