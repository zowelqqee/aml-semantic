# Decision Runtime (v0.2)

The deterministic rule-first AML decision runtime. It is **frozen**: it is
retained unchanged as the control arm of every later experiment, and no
threshold in it has been altered since the v0.2 measurement.

Supersedes the former root-level `architecture.md`, `pipeline.md`,
`decision_examples.md` and `research_notes.md`, which described the same
component and have been merged here.

| | |
|---|---|
| **Purpose** | Turn one transaction into an auditable ALLOW / REVIEW / BLOCK / ABSTAIN decision that can be replayed byte-for-byte. |
| **Inputs** | IBM AMLSim / IBM AML transaction CSVs, optional account CSVs, and the graph history built from them. |
| **Outputs** | `DecisionRecord`, a canonical JSON audit per decision (`artifacts/aml_audit/`), and portable Graphviz DOT traces (`artifacts/aml_graphs/`). |
| **Guarantees** | Deterministic: stable SHA-256 identifiers, fixed evidence/policy/audit key ordering, a logical execution clock, and byte-stable replay for unchanged input. Rules produce evidence only; the policy engine is the sole component that selects an outcome. |
| **Limitations** | No model, no fitted parameter, no probability calibration. On IBM AML `HI-Small` it alerts on 44.7% of the stream. Four of its ten rules can never fire because the source carries no SAR, KYC, sanctions or source-of-funds feed. |

---

## Pipeline

```mermaid
flowchart LR
  TX[Transaction] --> FE[FactExtractor]
  FE -->|typed Facts| RE[AMLRuleEngine]
  RE -->|Evidence, never decisions| CE[ConflictEngine]
  CE -->|risk qualified by controls| PE[AMLPolicyEngine]
  PE -->|ordered PolicyOutcomes| DR[Decision]
  DR --> AU[AuditEngine]
  AU --> RP[replay]
  RP -.->|same dataset fingerprint| DR
```

Stages are isolated but orchestrated by `AMLDecisionRuntime`:

1. `FactExtractor` derives typed observations from the transaction and prior graph history.
2. `AMLRuleEngine` maps facts to immutable evidence; it never decides.
3. `ConflictEngine` pairs risk evidence with qualifying controls.
4. `AMLPolicyEngine` evaluates a fixed, ordered policy set.
5. `AuditEngine` writes canonical JSON, including every intermediate artefact.
6. `replay()` re-executes the transaction against the same immutable dataset fingerprint.

## Running it

```bash
python demo.py
```

```bash
python demo.py --data /path/to/amlsim/output
AMLSIM_DATASET=/path/to/transactions.csv python demo.py --transaction-id TX-123
```

Rendering a trace is optional:

```bash
dot -Tpng artifacts/aml_graphs/TX-1004.dot -o trace.png
```

## Loader

The loader accepts IBM AMLSim transaction logs (`Timestamp`, `From Bank`,
`Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`,
`Is Laundering`) and compact equivalent headers. It constructs Accounts,
Countries, Transactions, Relationships, Alerts, HistoricalBehavior records and
SARFlags from the supplied data and graph context. Unmapped source columns are
preserved in transaction metadata, so new evidence rules do not require a schema
migration.

## Vocabulary

**Facts.** `LargeTransfer`, `NewBeneficiary`, `HighRiskCountry`,
`VelocityIncrease`, `OldKYC`, `PreviousSAR`, `ConnectedToSAR`,
`RepeatedDestination`.

**Controls** — the counter-evidence needed for explicit conflict handling:
verified source of funds, recent manual verification, expected payroll event.

**Decisions.** `ALLOW`, `REVIEW`, `BLOCK`, `ABSTAIN`. Block dominates review,
review dominates allow, and abstain is used only when neither risk nor
qualifying evidence is present — so missing information is never silently
treated as a clean transaction.

## Worked examples

From the bundled AMLSim-format fixture used by `python demo.py`:

| Transaction | Signals | Policy | Decision |
|---|---|---|---|
| `TX-1004` | Large transfer, high-risk country, stale KYC, repeated destination | AML-04, AML-09 | `REVIEW` |
| `TX-1006` | Known-SAR originator, direct SAR connection, large high-risk transfer | AML-01, AML-02 | `BLOCK` |
| `TX-1008` | Stale KYC plus recent manual verification and expected payroll context | AML-10 | `ALLOW` |

`TX-1008` illustrates the load-bearing distinction: **a mitigation does not
delete risk evidence.** Both evidence records remain in the audit, and the
conflict object states why the risk is qualified. The policy engine then
deterministically chooses its result from the complete evidence set.

## Design notes

The research object is the decision runtime, not prediction quality. The central
invariant is that every decision reduces to source facts, explicit rule
evidence, explicit conflicts, and an ordered policy outcome.

Laundering labels are ingested only as historical SAR context. They are not used
to train or infer a classifier, which permits experiments on decision governance
without conflating the evaluation with a fitted model.

Determinism has a practical constraint: wall-clock duration is deliberately
excluded from replay equality because it is not reproducible. The audit reports
`execution_time_ms: 0` under a `deterministic-logical-clock` model; production
instrumentation belongs outside the immutable decision record.

## Measured behaviour

Full v0.2 measurement: [`../research/aml-v0-2-report.md`](../research/aml-v0-2-report.md).
Earlier diagnostic: [`../research/aml-v0-1-diagnostic.md`](../research/aml-v0-1-diagnostic.md).

On the 100,000-event evaluation horizon used by the v0.2 report, four of ten
rules account for every activation and six never fire at all — the finding that
motivated the [Semantic Context Layer](semantic-context-layer.md).
