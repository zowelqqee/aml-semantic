# Documentation

Read in this order — each layer composes the one above it and none modifies
it. Every design doc states its **Purpose, Inputs, Outputs, Guarantees and
Limitations**.

## Layers

| Document | Contents |
|---|---|
| [aml/decision-runtime.md](aml/decision-runtime.md) | The frozen v0.2 rule-first runtime: pipeline, vocabulary, worked examples, determinism |
| [aml/semantic-context-layer.md](aml/semantic-context-layer.md) | Ontology, object hierarchy, lifecycle, confidence model, graph model, replay and audit contracts |
| [aml/behaviour-layer.md](aml/behaviour-layer.md) | Temporal engine, behaviour objects, dynamic roles, scenarios, the composition guarantee |

## Catalogs

| Catalog | Contents |
|---|---|
| [aml/catalog-behaviour-objects.md](aml/catalog-behaviour-objects.md) | 25 behavioural hypotheses: predicate, horizon, prior, counter-evidence |
| [aml/catalog-scenarios.md](aml/catalog-scenarios.md) | 8 stages, 6 declared stage patterns, matching and confidence |
| [aml/catalog-roles.md](aml/catalog-roles.md) | 13 dynamic roles, declared priority order, transition records |

## Benchmarks

Engineering results. Start with the [benchmark index](benchmarks/README.md).

| Report | Question it answers |
|---|---|
| [benchmarks/README.md](benchmarks/README.md) | All measured numbers in one place |
| [benchmarks/semantic-runtime-v1.md](benchmarks/semantic-runtime-v1.md) | Does semantic reasoning reduce false positives? |
| [benchmarks/behaviour-runtime-v1.md](benchmarks/behaviour-runtime-v1.md) | Does reasoning about behaviour over time change the decision? |
| [benchmarks/window-scaling.md](benchmarks/window-scaling.md) | Is the architecture limited by data or by reasoning? |
| [benchmarks/semantic-vs-raw.md](benchmarks/semantic-vs-raw.md) | Did the semantic layer create information, or re-encode raw features? |

## Research history

| Document | Contents |
|---|---|
| [research/aml-v0-2-report.md](research/aml-v0-2-report.md) | The frozen v0.2 measurement that motivated the semantic layer |
| [research/aml-v0-1-diagnostic.md](research/aml-v0-1-diagnostic.md) | Earlier diagnostic pass |
