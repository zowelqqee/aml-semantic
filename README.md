# MicroWorld Semantic Runtimes

This repository contains two self-contained, reproducible experiments in
auditable, semantic decision runtimes. Both projects make semantic state
explicit, preserve an inspectable evidence trail, and compare that
representation with conventional ML baselines.

| Project | Domain | What it contains |
|---|---|---|
| [AML Decision Runtime](aml_runtime/README.md) | Anti-money-laundering transactions | A deterministic rule, semantic-context, and behaviour runtime; design documentation; tests; and benchmark artifacts. |
| [IEEE-CIS Semantic Runtime Transfer](ieee_cis_runtime/README.md) | Card-payment fraud | A standalone fraud-domain port and chronological transfer experiment comparing raw and semantic features. |

## Repository layout

```text
aml_runtime/        AML decision runtime, documentation, tests, and artifacts
ieee_cis_runtime/   IEEE-CIS fraud transfer runtime, documentation, tests, and artifacts
AUTHORS             Original author information
LICENSE             Apache License 2.0
NOTICE              Copyright and attribution notice
```

## Quick start

The projects have separate dependencies and should be run from their own
directories.

### AML Decision Runtime

```bash
cd aml_runtime
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python demo.py
```

The deterministic runtime itself uses only the standard library. The optional
ML benchmark modules require the packages in `aml_runtime/requirements.txt`.
The bundled sample data is sufficient for the tests and demo; reproducing the
large benchmarks additionally requires IBM AML `HI-Small` data at
`aml_runtime/data/ibm_aml_data/`. Full design notes and benchmark protocols
are indexed in [the AML documentation](aml_runtime/docs/README.md).

### IEEE-CIS Semantic Runtime Transfer

```bash
cd ieee_cis_runtime
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Reproducing the full transfer experiment requires the labelled IEEE-CIS
`train_transaction.csv` and `train_identity.csv`; the source data is not
included. The complete command, frozen chronological protocol, and measured
results are in the [project README](ieee_cis_runtime/README.md) and
[transfer analysis](ieee_cis_runtime/docs/fraud_transfer_analysis.md).

## Reproducibility and scope

Published artifacts are committed with the repository. The two experiments
use different data sources, protocols, dependencies, and package layouts, so
their metrics should be interpreted within their respective projects rather
than compared directly.

## License

Copyright © 2026 Arseniy Abramidze. Licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and [AUTHORS](AUTHORS)
for attribution.
