"""Anti-Fraud Semantic Runtime — the same architecture, a different ontology.

This package is a *port*, not a redesign, of the AML Runtime. The generic
decision machinery — ``Evidence``, ``Conflict``, ``PolicyOutcome``,
``DecisionRecord``, ``Decision``, ``ConflictEngine`` — is retained in the
local ``ieee_cis_semantic_transfer.core`` module without a runtime dependency
on the AML package. Nothing about those classes is AML-specific.

Only the ontology changes: what an entity is, what a semantic object claims,
what a behaviour hypothesis looks like, which roles an account moves through.
Those are declared fresh here, from the IEEE-CIS card-fraud domain, in
``semantic/ontology.py`` and ``behaviour/ontology.py``.

See ``docs/fraud_runtime_report.md`` for the architecture-port map, vocabulary,
source limitations, and audit discipline.
"""
