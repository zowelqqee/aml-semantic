# MicroWorld

**A deterministic semantic runtime for reasoning over event streams.**

MicroWorld converts low-level events into explicit semantic state before a
deterministic policy or machine-learning model evaluates them.

```mermaid
flowchart TB
  R["Raw events"] --> S["Semantic runtime"]
  S --> E["Entities"]
  S --> L["Relationships"]
  S --> C["Context"]
  S --> B["Behaviour"]
  E --> V["Evidence and conflicts"]
  L --> V
  C --> V
  B --> V
  V --> M["Machine learning or policy"]
  M --> D["Decision and complete audit"]
```

Classical ML learns directly from event vectors.

MicroWorld first constructs an explicit semantic world, then lets a model or a
declared policy reason over it.

> **Semantic runtime**
>
> Deterministic. Replayable. Auditable. Transferable across evaluated domains.

## Why?

Events describe what happened, not what it means. A transaction is not yet an
entity, a relationship, a behavioural pattern, or a scenario. Treating those
as explicit state makes the path to a decision inspectable.

```mermaid
flowchart TB
  T["Transaction"] --> E["Entity"]
  E --> B["Behaviour"]
  B --> S["Scenario"]
  S --> V["Evidence"]
  V --> D["Decision"]
```

The runtime keeps every stage available for inspection. Rules and models emit
evidence; the policy is the only component that selects an AML decision.

## Results

### AML

An identical CatBoost protocol compares raw transaction features with the
same features augmented by semantic state.

| Configuration | Recall | False positives | PR-AUC | Result |
|---|---:|---:|---:|---|
| Raw CatBoost | 90.12% | 4,665 | 0.27426 | Baseline |
| Semantic only | 90.12% | 6,128 | 0.44877 | Better ranking; lower F1 than raw |
| **Raw + Semantic** | **91.30%** | **4,327** | **0.49274** | **+1.18 pp recall, 338 fewer false positives, PR-AUC +80%** |

The semantic representation complements raw fields; it does not replace them.
Full protocol, ablations, and artifacts: [AML benchmarks](aml_runtime/docs/benchmarks/README.md).

### IEEE-CIS fraud transfer

The same representation was ported to the 590,540-row IEEE-CIS labelled
payment stream. The evaluation uses a strict chronological 80/20 split.

| Configuration | Recall | Fraud cases detected | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Raw | 64.20% | 2,609 | 0.78877 | 0.17956 |
| **Raw + Semantic** | **67.30%** | **2,735** | **0.79535** | **0.18347** |
| **Change** | **+3.10 pp** | **+126** | **+0.00658** | **+0.00390** |

At the fixed 0.50 threshold, the combined model also produced 1,783 more
false positives. The result improves ranking and recall; it is not a
threshold-optimised deployment claim. See the complete
[transfer analysis](ieee_cis_runtime/docs/fraud_transfer_analysis.md).

## Architecture

The AML runtime is layered so that new semantic state composes with, rather
than replaces, the decision runtime below it.

```mermaid
flowchart LR
  T["Event stream"] --> S["Semantic context"]
  S --> B["Behaviour layer"]
  S --> E["Evidence"]
  B --> E
  E --> C["Conflict engine"]
  C --> P["Versioned policy"]
  P --> A["Audit record"]
```

The semantic context resolves entities and creates typed objects with causal
evidence. The behaviour layer derives temporal objects, dynamic roles, and
scenarios. The conflict engine preserves both risk and control evidence; it
does not delete one to make room for the other.

> **Runtime controls**
>
> - Content-addressed object, evidence, conflict, and audit identifiers
> - Stable ordering and logical time for byte-stable replay
> - Explicit coverage gaps for absent input
> - Audits that trace a decision back to source observations

The IEEE-CIS implementation uses the same semantic and behaviour concepts as
a causal feature-construction layer. Its tests enforce chronology,
no-future-information, and label isolation.

## Reproduce

The two implementations are independent.

```bash
cd aml_runtime
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python demo.py
```

```bash
cd ieee_cis_runtime
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The AML demo and tests use bundled data. Large AML benchmarks require IBM AML
`HI-Small` at `aml_runtime/data/ibm_aml_data/`. The IEEE-CIS experiment requires
labelled `train_transaction.csv` and `train_identity.csv`; neither source
dataset is included.

## Limitations

- The AML results depend on the supplied data and evaluation protocol.
  Behavioural coverage is limited by available history, and more history did
  not monotonically improve quality.
- Semantic-only features trail raw features on AML F1 and IEEE-CIS ranking
  metrics. The measured gains come from the combined representation.
- The AML policy is declared and inspectable, not learned or calibrated for a
  production threshold.
- Production use requires domain validation, calibration, monitoring, and data
  governance beyond the committed experiments.

## Repository layout

```text
aml_runtime/        AML runtime, documentation, tests, and result artifacts
ieee_cis_runtime/   IEEE-CIS transfer runtime, documentation, tests, and artifacts
AUTHORS             Original author information
LICENSE             Apache License 2.0
NOTICE              Copyright and attribution notice
```

## License

Copyright © 2026 Arseniy Abramidze. Licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and [AUTHORS](AUTHORS).
