# IEEE-CIS Semantic Runtime Transfer Experiment

Self-contained, reproducible package for the completed chronological transfer experiment. It contains the fraud-domain Runtime port, its minimal domain-independent evidence/conflict core, tests, documentation, and the executed benchmark outputs.

## Layout

- `src/ieee_cis_semantic_transfer/` — standalone runtime and experiment harness.
- `tests/` — chronology, no-future-information, and label-isolation tests.
- `docs/` — ontology/source limitations and measured transfer analysis.
- `artifacts/fraud_semantic/` — executed full-stream tables, SHAP, ablations, correlations, and plots.

The 1.4 GB IEEE-CIS CSV source is deliberately not duplicated here. Provide the labelled `train_transaction.csv` and `train_identity.csv` from the existing `ieee_cis_data/` directory or an equivalent local copy. The later unlabelled Kaggle test split is never read.

## Reproduce

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src MPLCONFIGDIR=.matplotlib .venv/bin/python -m ieee_cis_semantic_transfer.fraud_runtime.semantic_vs_raw \
  --transactions ../../ieee_cis_data/train_transaction.csv \
  --identity ../../ieee_cis_data/train_identity.csv \
  --output-dir artifacts/fraud_semantic
PYTHONPATH=src .venv/bin/python -m ieee_cis_semantic_transfer.fraud_runtime.profile \
  --transactions ../../ieee_cis_data/train_transaction.csv \
  --identity ../../ieee_cis_data/train_identity.csv \
  --output artifacts/fraud_semantic/source_profile.json
```

The experiment uses the full 590,540-row labelled stream, a strict chronological 80/20 split, identical CatBoost hyperparameters for Raw, Semantic, and Raw + Semantic, and a fixed 0.50 decision threshold. Category vocabularies are fitted only on the chronological training prefix.

## Result snapshot

Executed Raw + Semantic improved ranking and recall over Raw: ROC-AUC 0.79535 vs 0.78877, PR-AUC 0.18347 vs 0.17956, and recall 0.67298 vs 0.64198. It added false positives at the fixed threshold, so the conclusion is a qualified transfer success rather than a threshold-optimised operational claim. See [docs/fraud_transfer_analysis.md](docs/fraud_transfer_analysis.md).
