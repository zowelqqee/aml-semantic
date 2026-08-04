#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${IEEE_CIS_DATA_DIR:-../../ieee_cis_data}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/fraud_semantic}"

PYTHONPATH=src "${PYTHON_BIN}" -m unittest discover -s tests -v
PYTHONPATH=src MPLCONFIGDIR=.matplotlib "${PYTHON_BIN}" -m ieee_cis_semantic_transfer.fraud_runtime.semantic_vs_raw \
  --transactions "${DATA_DIR}/train_transaction.csv" \
  --identity "${DATA_DIR}/train_identity.csv" \
  --output-dir "${OUTPUT_DIR}"
PYTHONPATH=src "${PYTHON_BIN}" -m ieee_cis_semantic_transfer.fraud_runtime.profile \
  --transactions "${DATA_DIR}/train_transaction.csv" \
  --identity "${DATA_DIR}/train_identity.csv" \
  --output "${OUTPUT_DIR}/source_profile.json"
