# Role Transition Catalog

A role is **a state an account is currently in**, not a classification of what
it is. The same account may hold four roles in an afternoon. Source of truth:
`RoleType` in `aml_runtime/behaviour/ontology.py` and `_ROLE_PRIORITY` /
`BehaviourLayer._role` in `aml_runtime/behaviour/layer.py`.

Roles are **not labels**. Nothing in the assignment reads the laundering column,
and no role is a verdict: `MoneyMuleCandidate` and `ShellCompanyCandidate` are
named *candidate* because they are hypotheses carried by behaviour objects that
each record their own counter-evidence.

| | |
|---|---|
| **Purpose** | The dynamic role vocabulary: what an account is doing *now*, how it got there, and what caused the change. |
| **Inputs** | The behaviour object set for the event, the account's lifecycle state, and — for `SalaryReceiver` — the funder's current role. |
| **Outputs** | 13 roles, a declared priority order, and a `RoleTransition` record naming the behaviour objects that forced each change. |
| **Guarantees** | Roles are states, not classifications; nothing in the assignment reads the label column. `MoneyMuleCandidate` and `ShellCompanyCandidate` are hypotheses carried by objects that record their own counter-evidence. |
| **Limitations** | No hysteresis: a role flips as soon as its behaviour set changes, so sparse accounts oscillate. A role is computed for the originator, so `Collector` is rare. |


---

## The 13 roles

| Role | Meaning | Assigned when |
|---|---|---|
| `Unknown` | nothing is known yet | fewer than 6 observed events, no behaviour claimed |
| `ActiveCounterparty` | ordinary, characterised, unremarkable | ≥ 6 observed events, no behaviour family matched |
| `SalaryReceiver` | receives from a payment operator | the account's **funder** currently holds `PayrollOperator` or `Distributor`, and this account has ≤ 2 distinct payers |
| `Distributor` | pays many | `DistributionBehaviour`, `FanOutDistribution` or `RoutinePayrollBehaviour` |
| `Collector` | receives from many | `CollectionBehaviour`, `FanInCollection` or `CashConcentrationBehaviour` |
| `SettlementHub` | infrastructure | `SettlementHubBehaviour` |
| `PayrollOperator` | sustained payment-run operator | `PayrollOperatorBehaviour` |
| `TreasuryAccount` | internal accounting location | `LiquidityBalancingBehaviour` |
| `Accumulator` | gathers and holds | `MoneyAccumulationBehaviour` |
| `TransitAccount` | value passes through | `PassThroughBehaviour`, `TransitBehaviour`, `RapidLayeringBehaviour` or `HighVelocityLayering` |
| `MoneyMuleCandidate` | collection then prompt forwarding, no baseline | `MoneyMuleBehaviour` |
| `ShellCompanyCandidate` | incorporated conduit with no operating regime | `ShellCompanyBehaviour` |
| `Dormant` | quiet relative to its own tempo | idle ≥ 8 × own mean gap ∧ idle ≥ 60 min, and no behaviour family matched |

## Assignment is a declared priority order, not a classifier

```
1  MoneyMuleCandidate      ← MoneyMuleBehaviour
2  ShellCompanyCandidate   ← ShellCompanyBehaviour
3  TransitAccount          ← PassThrough / Transit / RapidLayering / HighVelocityLayering
4  PayrollOperator         ← PayrollOperatorBehaviour
5  SettlementHub           ← SettlementHubBehaviour
6  TreasuryAccount         ← LiquidityBalancingBehaviour
7  Collector               ← Collection / FanIn / CashConcentration
8  Distributor             ← Distribution / FanOut / RoutinePayroll
9  Accumulator             ← MoneyAccumulationBehaviour
   ── no behaviour family matched ──
10 SalaryReceiver          ← funder's role + ≤ 2 distinct payers
11 Dormant                 ← lifecycle idle
12 ActiveCounterparty      ← ≥ 6 observed events
13 Unknown                 ← otherwise
```

The ordering is itself a declared design choice and is load-bearing in two
places:

- **Transit above Distribution.** An account that both fans out *and* forwards
  value promptly is more informatively described as a conduit than as a payer.
  Reversing these two would hide every layering account behind a distribution
  reading.
- **Composites above everything.** `MoneyMuleBehaviour` and
  `ShellCompanyBehaviour` already require several independent families to agree,
  so when one of them fires it is the most specific available description.

Role confidence is the **strongest supporting behaviour's** confidence, or 0.5
for the lifecycle-derived roles (`SalaryReceiver`, `Dormant`,
`ActiveCounterparty`), or 0.0 for `Unknown`.

## Roles inferred across the graph

Two roles are not derived from the subject account's own behaviour:

- **`SalaryReceiver`** reads the *funder's* current role. If the account that
  last paid this one currently holds `PayrollOperator` or `Distributor`, and
  this account has at most two distinct payers, it is receiving what looks like
  salary. This is a causal inference over the role graph — the funder's role was
  itself established from that account's own prior behaviour — and it is the
  only place a role propagates between accounts.
- **`Dormant`** is a lifecycle state measured against the account's own mean
  inter-event gap, not against a global clock. An account that transacts every
  two minutes is dormant after sixteen; an account that transacts hourly is not.

## Transitions

Every role change emits a `RoleTransition`:

| Field | Contents |
|---|---|
| `from_role`, `to_role` | the states |
| `at_minute`, `at_event` | when, and on which transaction |
| `caused_by` | the behaviour object ids that forced the change |
| `explanation` | written form, naming the behaviours or "lifecycle state alone" |

Transitions are causal records, not annotations: `caused_by` points at objects
that each carry their own supporting observations and counter-evidence, so a
transition can be walked all the way down to raw observations.

`RoleObject` additionally carries `since_minute`, `tenure_minutes` and
`transition_count`. All three enter the semantic feature space — an account that
has changed role four times in an hour is describable as unstable without any
rule saying so.

## The lifecycle the brief asked for

```
Unknown ──► SalaryReceiver ──► Distributor ──► TransitAccount ──► MoneyMuleCandidate ──► Dormant
```

Every arrow in that chain is expressible in this catalog. Whether any given
arrow is *observed* depends on the window: on the frozen 5.6-hour prefix the
common transitions are lifecycle ones (`Unknown → Dormant`,
`Unknown → ActiveCounterparty`, `Dormant → ActiveCounterparty`), because most
accounts never accumulate enough history for a behaviour family to fire. The
measured transition census is in
`artifacts/aml_behaviour_v1/benchmark_report.md`.

## Not implemented in v1

- **Role hysteresis.** A role flips as soon as its behaviour set changes; there
  is no minimum tenure and no cost to oscillating. `Dormant → ActiveCounterparty
  → Dormant` therefore appears frequently on sparse accounts. Whether that is
  noise or signal is an open question, and adding hysteresis would be a tuning
  decision, so it was not made.
- **Role history as an object.** Only the current role plus the transition count
  is materialised; the full ordered role history is recoverable from the audit
  stream but is not a first-class object.
- **Roles for accounts that never originate.** A role is computed for the
  originator of an event. A pure receiver's role is only established when it
  first sends, which is why `Collector` is rare on this window.
