# IEEE-CIS Fraud Semantic Runtime

This report records the completed second-domain port. All counts below come from the executed full-stream profile in `artifacts/fraud_semantic/source_profile.json`; no labels enter the runtime fold.

## Dataset and chronology

The public labelled IEEE-CIS train stream contains 590,540 transactions: 20,663 fraud labels (3.4990%). `TransactionDT` is non-decreasing from 86,400 to 15,811,131 seconds (182.0 days). The identity table joins to 144,233 rows (24.4239%). The later Kaggle test partition is not read: it has no labels and falls entirely after the train range.

For every row the runtime executes `observe -> feature/evidence -> commit`. The profile recorded 590,540 semantic commits and 590,540 Behaviour-Layer commits. The causality tests in `tests/test_fraud_runtime.py` also passed: altering a future value/label cannot change a prior audit reading, and replacing labels changes neither semantic nor behavioural objects.

## Supported semantic objects

| Family | Objects emitted from prior-only state | Full-stream emissions |
|---|---|---:|
| History/regime | `NoEstablishedCardHistory`, `AmountRegime`, `TempoRegime`, `DeviceRegime`, `ChannelRegime` | 48,996 / 541,544 / 541,544 / 488,121 / 541,544 |
| Card–client-signature relation | `FirstDeviceContact`, `RecentlyLinkedDevice`, `EstablishedDeviceRelationship`, `NonInformativeDeviceNovelty` | 11,256 / 20,238 / 91,185 / 18,176 |
| Event context | `RoutineSpendingAmount`, `ExpectedHighValueSpend`, `UnexpectedSpendingAmount`, `MinimalTestAmount`, `UnexpectedBillingRegion`, `TempoRegimeShift` | 458,859 / 81,706 / 853 / 126 / 27,441 / 32,342 |
| Coverage/context | `UnverifiedDeviceContext`, `SharedClientSignatureExposure` | 446,307 / 101,898 |

`SharedClientSignatureExposure` deliberately means repeated `DeviceType + DeviceInfo` *signature* across card identities—not physical-device sharing. The profile found signature reuse frequently, while the source supplies no persistent hardware identifier; treating it as a device-sharing fraud finding would not be supportable.

## Behaviour Layer

The Behaviour Layer retains the AML lifecycle pattern: infer against prior temporal state, project behaviours to generic evidence, detect declared conflicts, then commit. Emitted hypotheses were `CardTestingBehaviour` (57), `DeviceRotationBehaviour` (56), `VelocityBurstBehaviour` (66,405), `CompromisedCardBehaviour` (822), `UnexpectedSpendingBehaviour` (44,486), `DormantCardReactivation` (7,427), `TrustedDeviceBehaviour` (72,723), `NormalSpendingBehaviour` (460,474), `ExpectedVelocityBehaviour` (1,137), and explicit `InsufficientCardHistory` (48,996).

These are hypotheses over observed card identities and client signatures, not ground-truth explanations of fraud. In particular, `CompromisedCardBehaviour` requires an established prior card-signature relation, a new signature, and a contemporaneous regime break; it does not assert that a card was in fact compromised.

## Explicitly unsupported concepts

| Not emitted | Why the public release cannot support it |
|---|---|
| Customer, TrustedIdentity, SyntheticIdentityBehaviour | no verified person identity or identity-linkage ground truth |
| Merchant, KnownMerchant, MerchantFanout, MerchantAbuseBehaviour, TrustedMerchantBehaviour | no merchant/seller identifier; email domains are purchaser/recipient providers, not merchants |
| Physical `Device`, DeviceSharingBehaviour, SharedDeviceCluster | client signature/model/browser string is not a persistent physical-device identifier |
| ImpossibleTravel, KnownLocation | `addr1`/`addr2` are anonymised; `dist1`/`dist2` have no documented geocoded endpoints or travel-time model |
| CredentialStuffing, RepeatedDeclineRecovery | no authentication-attempt or approval/decline event log |
| CrossJurisdictionSpend, calibrated region risk | no jurisdiction mapping or risk list |
| VirtualAssetExposure, CurrencyConversionSpend | no venue registry or multi-currency ledger |

## Architecture Reuse

**57.1% reused unchanged under a strict source-level count (4 of 7 generic subsystems):** generic Evidence/Conflict/Decision models, `ConflictEngine`/`ConflictSpecification`, stable identifiers/replay primitives, and serialization/audit primitives are retained unchanged in the local `core.py`. This intentionally excludes components whose *vocabulary or state shape* had to change for card purchases.

- **Reused reasoning pipeline:** yes — prior-only semantic objects → facts → generic evidence → declared conflicts → policy decision.
- **Reused behaviour pipeline:** yes — independent temporal `observe`/`commit`, behaviour objects, role transitions, scenarios, and evidence projection. Card/device-signature predicates replace AML flow predicates.
- **Reused evidence system:** yes, unchanged data types and evidence contract.
- **Reused audit system:** yes, immutable semantic/behaviour objects and replay hashes; fraud adds ontology/version pins.
- **Reused lifecycle system:** yes in mechanism; its domain fields are card age, idle time, observed purchases, client signatures, and active buckets rather than account counterparties.
- **New ontology concepts:** card identity, client signature, card–signature relationship, personal amount/tempo/channel/region regimes, test-sized charges, and card lifecycle/behaviour hypotheses listed above.

The completed benchmark and transfer conclusion are in [fraud_transfer_analysis.md](fraud_transfer_analysis.md).
