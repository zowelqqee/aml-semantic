# Semantic Behaviour Runtime — benchmark report

Protocol: first 500,000 chronological events; 400,000 train; 100,000-event horizon with 28 laundering-labelled events.

Semantic ontology `80103ed60dd19c70…`; behaviour ontology `7b99b0c109dbb8bd…`.

Every semantic and behavioural constant was declared before this benchmark executed; no threshold was selected against the evaluation labels and none was changed after seeing a result.

## Arms

| Arm | TP | FP | FN | Precision | Recall | F1 | Alerts | Alert rate | ML inferences |
|---|---|---|---|---|---|---|---|---|---|
| runtime_only | 14 | 44680 | 14 | 0.000313 | 0.5000 | 0.000626 | 44694 | 0.4469 | 0 |
| semantic_behaviour_runtime | 1 | 2206 | 27 | 0.000453 | 0.0357 | 0.000895 | 2207 | 0.0221 | 0 |
| semantic_behaviour_runtime_plus_ml/CatBoost | 1 | 2312 | 27 | 0.000432 | 0.0357 | 0.000854 | 2313 | 0.0231 | 48709 |
| semantic_behaviour_runtime_plus_ml/LightGBM | 13 | 15787 | 15 | 0.000823 | 0.4643 | 0.001643 | 15800 | 0.1580 | 48709 |
| semantic_behaviour_runtime_plus_ml/XGBoost | 1 | 2328 | 27 | 0.000429 | 0.0357 | 0.000849 | 2329 | 0.0233 | 48709 |
| semantic_runtime | 0 | 382 | 28 | 0.000000 | 0.0000 | 0.000000 | 382 | 0.0038 | 0 |

## Behaviour layer output

| Quantity | Count |
|---|---|
| Behaviour objects generated | 98,913 |
| Role transitions | 16,218 |
| Scenario detections | 28,428 |
| Conflicts | 10,603 |

## Behaviour census

| Behaviour | Emissions |
|---|---|
| InsufficientBehaviouralHistory | 77,532 |
| DistributionBehaviour | 6,831 |
| FanOutDistribution | 6,593 |
| BurstActivityBehaviour | 3,483 |
| DormantAccountActivation | 1,788 |
| MoneyAccumulationBehaviour | 1,524 |
| ShellCompanyBehaviour | 541 |
| ExpectedBusinessCycle | 367 |
| LiquidityBalancingBehaviour | 97 |
| TransitBehaviour | 82 |
| UnexpectedBusinessCycle | 50 |
| RoutinePayrollBehaviour | 20 |
| FanInCollection | 3 |
| RelationshipGrowthBehaviour | 2 |

## Role census

| Role | Events |
|---|---|
| Unknown | 71,343 |
| ActiveCounterparty | 11,193 |
| Distributor | 6,851 |
| Dormant | 6,326 |
| SalaryReceiver | 2,064 |
| Accumulator | 1,517 |
| ShellCompanyCandidate | 541 |
| TreasuryAccount | 97 |
| TransitAccount | 65 |
| Collector | 3 |

## Role transitions

| Transition | Count |
|---|---|
| Unknown->Dormant | 5,595 |
| Unknown->ActiveCounterparty | 3,394 |
| Dormant->Unknown | 2,416 |
| Dormant->ActiveCounterparty | 1,704 |
| Unknown->SalaryReceiver | 916 |
| Unknown->Accumulator | 733 |
| ActiveCounterparty->Dormant | 707 |
| Unknown->ShellCompanyCandidate | 159 |
| ShellCompanyCandidate->ActiveCounterparty | 88 |
| ActiveCounterparty->ShellCompanyCandidate | 87 |
| Accumulator->ActiveCounterparty | 81 |
| ActiveCounterparty->TreasuryAccount | 50 |
| SalaryReceiver->Accumulator | 46 |
| Dormant->ShellCompanyCandidate | 41 |
| ActiveCounterparty->TransitAccount | 24 |
| ActiveCounterparty->SalaryReceiver | 22 |
| ActiveCounterparty->Distributor | 21 |
| ActiveCounterparty->Accumulator | 20 |
| TreasuryAccount->ActiveCounterparty | 16 |
| SalaryReceiver->ActiveCounterparty | 13 |
| SalaryReceiver->ShellCompanyCandidate | 9 |
| Distributor->ActiveCounterparty | 7 |
| Accumulator->SalaryReceiver | 7 |
| SalaryReceiver->TransitAccount | 7 |
| SalaryReceiver->Dormant | 5 |
| Dormant->TransitAccount | 5 |
| ShellCompanyCandidate->Dormant | 5 |
| Dormant->Accumulator | 5 |
| SalaryReceiver->Unknown | 4 |
| ShellCompanyCandidate->SalaryReceiver | 4 |
| Unknown->Collector | 3 |
| TreasuryAccount->Dormant | 3 |
| Accumulator->Dormant | 2 |
| SalaryReceiver->TreasuryAccount | 2 |
| Accumulator->TreasuryAccount | 2 |
| Accumulator->Distributor | 2 |
| TransitAccount->ShellCompanyCandidate | 2 |
| Dormant->SalaryReceiver | 2 |
| Dormant->TreasuryAccount | 2 |
| ShellCompanyCandidate->TransitAccount | 1 |
| Unknown->Distributor | 1 |
| Accumulator->ShellCompanyCandidate | 1 |
| TreasuryAccount->Accumulator | 1 |
| TreasuryAccount->SalaryReceiver | 1 |
| TransitAccount->ActiveCounterparty | 1 |
| Collector->ActiveCounterparty | 1 |

## Scenario census

| Scenario | Detections |
|---|---|
| TreasuryCycling | 17,329 |
| NormalConsumerBehaviour | 11,089 |
| CollectThenForward | 10 |

## Conflicts

| Conflict kind | Count |
|---|---|
| value_context | 5,359 |
| structural_explanation | 5,019 |
| behaviour_structural | 146 |
| behaviour_stability | 58 |
| behaviour_regime | 16 |
| counterparty_context | 5 |

## Semantic feature importances

The model sees only behaviour, role, scenario, lifecycle and semantic objects.

### XGBoost

| Feature | Importance |
|---|---|
| sem_InternalBookEntry | 0.45592 |
| evidence_mitigation_count | 0.09784 |
| role_code | 0.04753 |
| sem_EstablishedRelationship | 0.04115 |
| lifecycle_observed_events | 0.03958 |
| role_transition_count | 0.03866 |
| sem_NonInformativeNovelty | 0.02887 |
| scn_NormalConsumerBehaviour | 0.02748 |
| sem_CrossJurisdictionTransfer | 0.02079 |
| lifecycle_horizons_filled | 0.01572 |
| sem_NormalOperationalBurst | 0.01468 |
| lifecycle_age_minutes | 0.01117 |
| lifecycle_idle_minutes | 0.01082 |
| evidence_effective_risk | 0.00913 |
| lifecycle_distinct_counterparties | 0.00911 |

### LightGBM

| Feature | Importance |
|---|---|
| lifecycle_idle_minutes | 22.00000 |
| lifecycle_age_minutes | 18.00000 |
| lifecycle_observed_events | 16.00000 |
| lifecycle_distinct_counterparties | 15.00000 |
| role_tenure_minutes | 13.00000 |
| sem_CrossJurisdictionTransfer | 9.00000 |
| evidence_mitigation_count | 7.00000 |
| role_transition_count | 5.00000 |
| evidence_risk_count | 5.00000 |
| evidence_effective_risk | 5.00000 |
| evidence_strongest_risk | 5.00000 |
| sem_ValueRegime | 4.00000 |
| sem_InternalBookEntry | 4.00000 |
| scn_TreasuryCycling | 4.00000 |
| evidence_unqualified_topics | 4.00000 |

### CatBoost

| Feature | Importance |
|---|---|
| sem_InternalBookEntry | 31.47474 |
| sem_NonInformativeNovelty | 10.33109 |
| evidence_mitigation_count | 8.99493 |
| lifecycle_idle_minutes | 6.65948 |
| lifecycle_age_minutes | 4.91950 |
| role_tenure_minutes | 4.03114 |
| role_transition_count | 3.68880 |
| lifecycle_horizons_filled | 2.99273 |
| sem_CrossJurisdictionTransfer | 2.86861 |
| lifecycle_distinct_counterparties | 2.19380 |
| sem_ValueRegime | 2.13539 |
| beh_FanOutDistribution | 1.91210 |
| role_code | 1.66227 |
| evidence_unqualified_topics | 1.64549 |
| evidence_effective_risk | 1.56762 |

## Cost

| Stage | Seconds |
|---|---|
| entity_resolution_seconds | 3.38 |
| semantic_feature_pass_seconds | 64.92 |
| frozen_runtime_seconds | 3.78 |
| behaviour_runtime_seconds | 11.28 |

Peak process RSS: 1,113,473,024 bytes.
