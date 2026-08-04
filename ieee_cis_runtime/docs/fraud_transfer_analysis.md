# IEEE-CIS Semantic Runtime Transfer Analysis

All values in this report are from the executed `artifacts/fraud_semantic/comparison_results.json` experiment. This is a chronological transfer test, not a Kaggle optimisation run.

## Frozen protocol

The full 590,540-row labelled train stream was split chronologically: rows 0–472,431 trained all arms and rows 472,432–590,539 (118,108 rows; 4,064 fraud labels) evaluated them. No random split, future transaction state, evaluation-label feature, or unlabelled later Kaggle test row was used.

All arms used CatBoost with 200 iterations, depth 6, learning rate 0.05, seed 20260804, four threads, and the same chronological-training class weight. The fixed decision threshold is 0.50. Raw categoricals use mappings fit only on the training prefix; evaluation categories unseen in that prefix map to `-1`.

## Three-arm result

| Arm | Recall | Precision | F1 | ROC-AUC | PR-AUC | FP | FN | Alert rate | Train s | Inference ms/event | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw CatBoost | 0.6420 | 0.09469 | 0.16504 | 0.78877 | 0.17956 | 24,944 | 1,455 | 23.33% | 8.98 | 0.000114 | 549.6 MB |
| Semantic CatBoost | 0.6297 | 0.08162 | 0.14451 | 0.75401 | 0.12038 | 28,794 | 1,505 | 26.55% | 11.57 | 0.000152 | 717.0 MB |
| Raw + Semantic CatBoost | 0.6730 | 0.09283 | 0.16316 | 0.79535 | 0.18347 | 26,727 | 1,329 | 24.94% | 12.43 | 0.000138 | 717.0 MB |

Feature generation took 3.91 s for raw fields and 113.17 s for the causal semantic fold (147.47 s wall-clock for the one-pass build). Exact CatBoost TreeSHAP was computed over the first 25,000 chronological evaluation rows for each arm.

Relative to Raw, Raw + Semantic changed recall by **+3.10 percentage points** (+126 true positives), ROC-AUC by **+0.00658**, and PR-AUC by **+0.00390**. At the fixed 0.50 threshold it also added 1,783 false positives, reducing precision by 0.00186 and F1 by 0.00188. Thus the measured improvement is a ranking/recall improvement, not an unqualified fixed-threshold operational improvement.

## Importance, SHAP, and correlation

In Raw + Semantic, CatBoost importance assigns 25.3% to semantic groups: lifecycle 10.37%, semantic objects 10.21%, roles 2.95%, behaviours 1.21%, evidence 0.50%, and scenarios 0.09%. Exact TreeSHAP gives those groups 27.1% combined: lifecycle 13.04%, semantic objects 7.90%, roles 4.50%, behaviours 1.10%, evidence 0.50%, and scenarios 0.08%.

The strongest semantic SHAP features in the combined arm are `lifecycle_idle_minutes` (0.1627), `lifecycle_distinct_devices` (0.0591), `sem_DeviceRegime` (0.0570), `role_tenure_minutes` (0.0478), `role_transition_count` (0.0465), and `sem_ChannelRegime` (0.0410). These are all history-derived features, not raw label lookups.

Correlation is intentionally calculated only against raw numeric fields; ordinal category codes are excluded because their numeric order is arbitrary. Some semantic state is expectedly correlated with a raw observation—for example `sem_UnverifiedDeviceContext` with raw identity coverage (|r|≈1.0) and an established signature relation with identity coverage (|r|≈0.797). Other prominent combined features encode temporal history rather than a same-row raw value, including idle time and role tenure. The complete feature-level table is `artifacts/fraud_semantic/feature_correlation.csv`.

## Vocabulary ablations

Dropping the raw vocabulary reproduces Semantic-only (PR-AUC 0.12038). In the combined arm, removing any semantic vocabulary did not improve all metrics simultaneously:

| Removed group | Recall | F1 | ROC-AUC | PR-AUC | FP |
|---|---:|---:|---:|---:|---:|
| none (Raw + Semantic) | 0.6730 | 0.16316 | 0.79535 | 0.18347 | 26,727 |
| semantic objects | 0.6668 | 0.16071 | 0.79477 | 0.18538 | 26,952 |
| behaviours | 0.6750 | 0.16218 | 0.79699 | 0.18204 | 27,019 |
| lifecycle | 0.6752 | 0.16510 | 0.79337 | 0.18685 | 26,432 |
| roles | 0.6764 | 0.16652 | 0.79871 | 0.18492 | 26,205 |
| scenarios | 0.6718 | 0.16136 | 0.79308 | 0.17752 | 27,044 |
| evidence | 0.6722 | 0.16262 | 0.79500 | 0.18430 | 26,803 |

The ablations show that the complete vocabulary is not uniformly best under every fixed-threshold metric. They do, however, confirm that the semantic groups are active: removing them changes ROC-AUC, PR-AUC, recall, false positives, and feature attribution rather than leaving the result invariant.

## Conclusion

**Qualified yes: the Semantic Runtime transfers useful information to this independent card-fraud domain.** On the full chronological IEEE-CIS experiment, Raw + Semantic improves both discrimination metrics (ROC-AUC 0.79535 vs 0.78877; PR-AUC 0.18347 vs 0.17956) and catches 126 additional fraud cases at the frozen threshold. The transfer is not yet an operational win at that threshold: it increases false positives and lowers F1 slightly. Therefore the evidence supports cross-domain semantic information transfer, while rejecting the stronger claim that the current fraud vocabulary is already threshold-optimal.

## Artifact inventory

- `comparison_results.json` — protocol and all metrics
- `source_profile.json` — full-stream ontology/behaviour coverage
- `feature_importance.csv`, `shap_summary.csv`, `feature_groups.csv`
- `feature_correlation.csv`, `ablation_results.csv`
- `roc_curves.png`, `pr_curves.png`
