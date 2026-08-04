# Comparison report — rule-first vs semantic vs behavioural

## Headline

| Metric | Runtime | Semantic Runtime | Semantic Behaviour Runtime | Semantic vs Runtime | Behaviour vs Semantic |
|---|---|---|---|---|---|
| True positives | 14 | 0 | 1 | -100.0% | n/a |
| False positives | 44,680 | 382 | 2,206 | -99.1% | +477.5% |
| False negatives | 14 | 28 | 27 | +100.0% | -3.6% |
| Precision | 0.000313 | 0.000000 | 0.000453 | -100.0% | n/a |
| Recall | 0.5000 | 0.0000 | 0.0357 | -100.0% | n/a |
| F1 | 0.000626 | 0.000000 | 0.000895 | -100.0% | n/a |
| Alert rate | 0.4469 | 0.0038 | 0.0221 | -99.1% | +477.7% |

## What the Behaviour Layer changed relative to the Semantic Runtime

| Decision movement | Events |
|---|---|
| ALLOW -> REVIEW | 1,227 |
| ABSTAIN -> REVIEW | 603 |
| REVIEW -> ALLOW | 3 |
| REVIEW -> ABSTAIN | 2 |

## With ML over the semantic feature space

| Model | TP | FP | Recall | Precision | Alerts | ML inferences | ML share of stream |
|---|---|---|---|---|---|---|---|
| XGBoost | 1 | 2,328 | 0.0357 | 0.000429 | 2,329 | 48,709 | 48.7% |
| LightGBM | 13 | 15,787 | 0.4643 | 0.000823 | 15,800 | 48,709 | 48.7% |
| CatBoost | 1 | 2,312 | 0.0357 | 0.000432 | 2,313 | 48,709 | 48.7% |

## False-positive causation

Each row is a distinct set of *unqualified* risk evidence that produced alerts, the false and true positives it produced, and the mitigating behaviour or semantic object that the catalog already declares as its qualifier. Where the qualifier column names an object, the alert would not have fired had that object been inferable from this window; where it says none exists, the catalog has a gap.

| Unqualified risk evidence | False positives | True positives | Declared qualifier that would have prevented it |
|---|---|---|---|
| BEH-DistributionBehaviour + BEH-FanOutDistribution + SEM-R03-INFORMATIVE-NOVELTY | 237 | 1 | BEH-PayrollOperatorBehaviour, BEH-RoutinePayrollBehaviour, BEH-SupplierSettlementBehaviour, SEM-M05-ROUTINE-VALUE |
| BEH-BurstActivityBehaviour + SEM-R02-TEMPO-REGIME-SHIFT | 187 | 0 | BEH-RoutinePayrollBehaviour, SEM-M03-OPERATIONAL-BURST |
| BEH-DistributionBehaviour + BEH-FanOutDistribution + SEM-R04-JURISDICTION | 165 | 0 | BEH-PayrollOperatorBehaviour, BEH-RoutinePayrollBehaviour |
| BEH-DistributionBehaviour + BEH-FanOutDistribution + SEM-R07-CASH-INSTRUMENT | 155 | 0 | BEH-ExpectedBusinessCycle, BEH-PayrollOperatorBehaviour, BEH-RoutinePayrollBehaviour, SEM-M05-ROUTINE-VALUE |
| SEM-R04-JURISDICTION + SEM-R07-CASH-INSTRUMENT | 147 | 0 | BEH-ExpectedBusinessCycle, SEM-M05-ROUTINE-VALUE |
| BEH-BurstActivityBehaviour + SEM-R07-CASH-INSTRUMENT | 122 | 0 | BEH-ExpectedBusinessCycle, BEH-RoutinePayrollBehaviour, SEM-M05-ROUTINE-VALUE |
| BEH-MoneyAccumulationBehaviour + SEM-R07-CASH-INSTRUMENT | 106 | 0 | BEH-ExpectedBusinessCycle, SEM-M05-ROUTINE-VALUE |
| BEH-BurstActivityBehaviour + BEH-MoneyAccumulationBehaviour | 103 | 0 | BEH-ExpectedBusinessCycle, BEH-RoutinePayrollBehaviour |
| BEH-BurstActivityBehaviour + BEH-ShellCompanyBehaviour | 103 | 0 | BEH-RoutinePayrollBehaviour |
| BEH-DistributionBehaviour + BEH-FanOutDistribution + SEM-R03-INFORMATIVE-NOVELTY + SEM-R07-CASH-INSTRUMENT | 98 | 0 | BEH-ExpectedBusinessCycle, BEH-PayrollOperatorBehaviour, BEH-RoutinePayrollBehaviour, BEH-SupplierSettlementBehaviour, SEM-M05-ROUTINE-VALUE |
| BEH-BurstActivityBehaviour + BEH-DormantAccountActivation | 90 | 0 | BEH-RoutinePayrollBehaviour |
| BEH-DormantAccountActivation + BEH-MoneyAccumulationBehaviour | 76 | 0 | BEH-ExpectedBusinessCycle |
| BEH-BurstActivityBehaviour + SEM-R04-JURISDICTION | 69 | 0 | BEH-RoutinePayrollBehaviour |
| BEH-DormantAccountActivation + SEM-R07-CASH-INSTRUMENT | 45 | 0 | BEH-ExpectedBusinessCycle, SEM-M05-ROUTINE-VALUE |
| BEH-BurstActivityBehaviour + SEM-R01-VALUE-REGIME-BREAK | 27 | 0 | BEH-ExpectedBusinessCycle, BEH-RoutinePayrollBehaviour, SEM-M07-ESTABLISHED-RELATIONSHIP |
| BEH-ShellCompanyBehaviour + SEM-R07-CASH-INSTRUMENT | 25 | 0 | BEH-ExpectedBusinessCycle, SEM-M05-ROUTINE-VALUE |
| BEH-MoneyAccumulationBehaviour + SEM-R04-JURISDICTION | 24 | 0 | BEH-ExpectedBusinessCycle |
| BEH-DormantAccountActivation + SEM-R01-VALUE-REGIME-BREAK | 22 | 0 | BEH-ExpectedBusinessCycle, SEM-M07-ESTABLISHED-RELATIONSHIP |
| BEH-DormantAccountActivation + BEH-ShellCompanyBehaviour | 22 | 0 | no declared qualifier exists |
| BEH-DistributionBehaviour + BEH-FanOutDistribution + BEH-UnexpectedBusinessCycle + SEM-R01-VALUE-REGIME-BREAK + SEM-R03-INFORMATIVE-NOVELTY | 20 | 0 | BEH-ExpectedBusinessCycle, BEH-PayrollOperatorBehaviour, BEH-RoutinePayrollBehaviour, BEH-SupplierSettlementBehaviour, SEM-M05-ROUTINE-VALUE, SEM-M07-ESTABLISHED-RELATIONSHIP |
