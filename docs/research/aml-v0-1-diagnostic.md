# AML Decision Runtime v0.1 diagnostic

## Scope and method

This diagnosis is a post-decision analysis of immutable audits from the
label-isolated IBM AML-Data `HI-Small_Trans.csv` experiment. It covers the
initial chronological 100,000-event slice (00:00--00:09 on 2022-09-01). The
`Is Laundering` column is not used below: counts come from facts, evidence,
conflicts, policies, and decisions already present in audit files.

## Observed collapse

| v0.1 result | Count | Share |
|---|---:|---:|
| REVIEW | 94,043 | 94.043% |
| ABSTAIN | 5,957 | 5.957% |
| ALLOW | 0 | 0.000% |
| BLOCK | 0 | 0.000% |

The audit-level evidence-count distribution was 4,381 with zero evidence,
72,246 with one item, 22,783 with two, and 590 with three. Its entropy was
1.066 bits and the mean was 1.196 evidence items per event. A further 1,576
transactions had large-transfer evidence but no policy trigger, so the runtime
fell through to `ABSTAIN`; this explains the difference between zero evidence
and abstain counts.

## Why REVIEW dominates

`FactExtractor` emits `NewBeneficiary` whenever no earlier payment exists from
the originator to the beneficiary. In this short early slice that occurred
92,967 times. Rule `AML-R07-BEHAVIOUR` turns either `NewBeneficiary` or
`RepeatedDestination` into the same risk evidence item; it activated 93,487
times. Policy `AML-09` triggers on *any* unqualified behavioural evidence, so
it reviewed every weak, isolated novelty signal. A data-coverage observation
was therefore converted into escalation without requiring corroboration.

| Rule | Activations | Interpretation |
|---|---:|---|
| AML-R07-BEHAVIOUR | 93,487 | Dominant; mostly a first observed counterparty, not necessarily anomalous behaviour. |
| AML-R01-LARGE | 23,772 | Amount threshold crossed. |
| AML-R03-VELOCITY | 2,323 | At least four causal outbound transfers in 24 hours. |
| AML-R02/R04/R05/R06/R08/R09/R10 | 0 | Required attributes were absent from the public adapter. |

The largest evidence combinations were `R07` alone (70,297), `R01 + R07`
(21,423), and `R03 + R07` (1,177). Thus one low-specificity rule determined
the decision distribution even before considering labels.

## Why BLOCK never occurs

The v0.1 block policies require a known SAR originator (`AML-01`) or a SAR
connection with a large or rapid transfer (`AML-02`). The public IBM
transaction/account schema has no independent SAR field. To preserve label
isolation, the adapter leaves account SAR flags false and returns no SAR graph
path. `PreviousSAR` and `ConnectedToSAR` therefore never occur. This is correct
for supplied information, not evidence that the block path is unimplemented.

## Why ALLOW never occurs

v0.1 `AML-10` allows only when evidence exists and every risk item is qualified
by mitigation evidence. The IBM adapter has no verified-source-of-funds,
manual-KYC, or payroll-control fields, so it emits no mitigation facts. All
observed evidence is risk-direction evidence and the allow predicate is
unsatisfiable on this schema, although it works on the AMLSim fixture.

## Why conflicts are zero

v0.1 conflict detection only has three exact risk/control pairs: risk versus
verified source of funds, stale KYC versus manual verification, and velocity
versus payroll. None of the three control facts is available in the IBM schema,
so no pair can be instantiated. The engine also records no evidence quality,
source-strength, or recency comparison. Zero conflicts is a schema consequence,
but the narrow pair-only implementation limits research value.

## Information unavailable to v0.1

The IBM public schema supplies timestamp, bank-qualified account identifiers,
received and paid amounts/currencies, payment format, and a post-hoc laundering
label. It does not supply country/jurisdiction, KYC age, independent SAR,
sanctions, source-of-funds verification, manual review events, expected payroll
status, or customer risk. The label remains excluded from facts, graph state,
evidence, policies, decisions, and audits.

High-risk-country, old-KYC, SAR, SAR-connection, and all control rules receive
no IBM input. v0.2 does not manufacture those attributes or infer them from
labels; it makes available transaction and causal-history evidence measurable.
