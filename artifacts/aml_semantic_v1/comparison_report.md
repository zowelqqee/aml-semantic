# Semantic Context Layer — five-arm benchmark

Protocol: `data/ibm_aml_data/HI-Small_Trans.csv`, first 500,000 chronological events; 400,000 train; 100,000-event evaluation horizon containing 28 laundering-labelled events.

Ontology `aml-semantic-ontology/1.0` hash `80103ed60dd19c70…`; 327,325 accounts resolved from reference data.

Every semantic constant was declared in ontology.py before this benchmark executed; no threshold was selected against the evaluation labels; the benchmark was executed once.

## Results

| Arm | TP | FP | FN | Recall | Precision | Alerts | Alert rate | ML inferences |
|---|---|---|---|---|---|---|---|---|
| ml_only/CatBoost | 3 | 378 | 25 | 0.1071 | 0.007874 | 381 | 0.0038 | 100000 |
| ml_only/LightGBM | 20 | 20120 | 8 | 0.7143 | 0.000993 | 20140 | 0.2014 | 100000 |
| ml_only/XGBoost | 3 | 424 | 25 | 0.1071 | 0.007026 | 427 | 0.0043 | 100000 |
| runtime_only | 14 | 44680 | 14 | 0.5000 | 0.000313 | 44694 | 0.4469 | 0 |
| runtime_plus_ml/CatBoost | 16 | 45057 | 12 | 0.5714 | 0.000355 | 45073 | 0.4507 | 100000 |
| runtime_plus_ml/LightGBM | 28 | 53767 | 0 | 1.0000 | 0.000520 | 53795 | 0.5380 | 100000 |
| runtime_plus_ml/XGBoost | 17 | 45054 | 11 | 0.6071 | 0.000377 | 45071 | 0.4507 | 100000 |
| semantic_runtime_only | 0 | 382 | 28 | 0.0000 | 0.000000 | 382 | 0.0038 | 0 |
| semantic_runtime_plus_ml/CatBoost | 1 | 429 | 27 | 0.0357 | 0.002326 | 430 | 0.0043 | 49310 |
| semantic_runtime_plus_ml/LightGBM | 18 | 12841 | 10 | 0.6429 | 0.001400 | 12859 | 0.1286 | 49310 |
| semantic_runtime_plus_ml/XGBoost | 2 | 444 | 26 | 0.0714 | 0.004484 | 446 | 0.0045 | 49310 |

## Semantic object census (evaluation horizon)

| Semantic type | Emissions |
|---|---|
| UnscaledValue | 83,608 |
| NoEstablishedBaseline | 82,935 |
| InternalBookEntry | 34,050 |
| RecentlyCreatedRelationship | 30,928 |
| NonInformativeNovelty | 28,845 |
| CrossJurisdictionTransfer | 21,030 |
| TempoRegime | 17,065 |
| ValueRegime | 16,392 |
| RoutineValueTransfer | 13,553 |
| CashInstrumentSettlement | 7,407 |
| CounterpartyRegime | 7,355 |
| DistributionNode | 6,820 |
| NormalOperationalBurst | 6,730 |
| FirstContact | 3,518 |
| HighRiskJurisdictionExposure | 2,697 |
| EstablishedRelationship | 2,659 |
| VirtualAssetExposure | 2,593 |
| ExpectedHighValueTransfer | 2,473 |
| CurrencyConversionTransfer | 1,047 |
| IntraCustomerTransfer | 1,037 |
| UnexpectedLargeTransfer | 366 |
| BehaviourRegimeShift | 312 |
| PassThroughAccount | 114 |
| LayeringChainSegment | 1 |

## Files

- `comparison_results.json` — every measured number
- `audits/semantic_decisions.jsonl.gz` — one compact audit line per evaluated event
- `audits/samples/` — full semantic audit records with replay pins
- `decision_examples.md` — decisions stated in semantic objects
- `curves.png` — ROC and precision-recall
