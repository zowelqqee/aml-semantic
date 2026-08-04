"""Deterministic research measurements for AML decision-runtime executions."""

from __future__ import annotations

import html
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .models import Decision
from .runtime import RuntimeResult


def _entropy(counts: Iterable[int]) -> float:
    values = [count for count in counts if count]
    total = sum(values)
    if not total:
        return 0.0
    return round(-sum((count / total) * math.log2(count / total) for count in values), 6)


class ResearchMetrics:
    """Accumulates label-free, audit-derived runtime measurements.

    Rule contribution is intentionally an association: a rule's presence in a
    decision is measured, but causal attribution is not claimed.
    """

    def __init__(self, rule_ids: tuple[str, ...]) -> None:
        self.rule_ids = tuple(sorted(rule_ids))
        self.transaction_count = 0
        self.decisions: Counter[str] = Counter()
        self.evidence_counts: Counter[int] = Counter()
        self.rule_activations: Counter[str] = Counter()
        self.rule_confidence_sums: Counter[str] = Counter()
        self.rule_decisions: dict[str, Counter[str]] = defaultdict(Counter)
        self.interactions: Counter[tuple[str, str]] = Counter()
        self.policy_activations: Counter[str] = Counter()
        self.conflict_kinds: Counter[str] = Counter()
        self.conflict_dimensions: Counter[str] = Counter()
        self.decision_depth_total = 0
        self.evidence_graph_nodes_total = 0
        self.evidence_graph_edges_total = 0
        self.audit_bytes_total = 0

    def observe(self, result: RuntimeResult) -> None:
        self.transaction_count += 1
        outcome = result.decision.decision.value
        self.decisions[outcome] += 1
        self.evidence_counts[len(result.evidence)] += 1
        active_rules = tuple(sorted(item.rule_id for item in result.evidence))
        for evidence in result.evidence:
            self.rule_activations[evidence.rule_id] += 1
            self.rule_confidence_sums[evidence.rule_id] += evidence.confidence
            self.rule_decisions[evidence.rule_id][outcome] += 1
        for pair in combinations(active_rules, 2):
            self.interactions[pair] += 1
        triggered = tuple(policy for policy in result.policies if policy.triggered)
        for policy in triggered:
            self.policy_activations[policy.policy_id] += 1
        for conflict in result.conflicts:
            self.conflict_kinds[conflict.kind] += 1
            self.conflict_dimensions.update(conflict.dimensions)
        self.decision_depth_total += len(result.facts) + len(result.evidence) + len(result.conflicts) + len(triggered)
        self.evidence_graph_nodes_total += 2 + len(result.facts) + len(result.evidence) + len(result.conflicts) + len(triggered)
        self.evidence_graph_edges_total += (
            len(result.facts)
            + sum(len(item.supporting_facts) for item in result.evidence)
            + 2 * len(result.conflicts)
            + sum(len(item.evidence_ids) for item in triggered)
        )
        self.audit_bytes_total += Path(result.audit["path"]).stat().st_size

    def to_dict(self) -> dict[str, object]:
        total = self.transaction_count or 1
        rule_contribution: dict[str, dict[str, object]] = {}
        for rule_id in self.rule_ids:
            activations = self.rule_activations[rule_id]
            rule_contribution[rule_id] = {
                "activation_count": activations,
                "contribution_to_allow": self.rule_decisions[rule_id][Decision.ALLOW.value],
                "contribution_to_review": self.rule_decisions[rule_id][Decision.REVIEW.value],
                "contribution_to_block": self.rule_decisions[rule_id][Decision.BLOCK.value],
                "contribution_to_abstain": self.rule_decisions[rule_id][Decision.ABSTAIN.value],
                "average_confidence": round(self.rule_confidence_sums[rule_id] / activations, 6) if activations else 0.0,
            }
        interaction_rows = [
            {"rule_a": left, "rule_b": right, "coactivation_count": count}
            for (left, right), count in sorted(self.interactions.items(), key=lambda item: (-item[1], item[0]))
        ]
        return {
            "transactions_measured": self.transaction_count,
            "decision_distribution": {decision.value: self.decisions[decision.value] for decision in Decision},
            "evidence_distribution": dict(sorted(self.rule_activations.items())),
            "evidence_count_distribution": {str(count): self.evidence_counts[count] for count in sorted(self.evidence_counts)},
            "evidence_entropy_bits": _entropy(self.rule_activations.values()),
            "evidence_count_entropy_bits": _entropy(self.evidence_counts.values()),
            "average_evidence_count": round(sum(count * frequency for count, frequency in self.evidence_counts.items()) / total, 6),
            "rule_activation_frequency": {rule_id: round(self.rule_activations[rule_id] / total, 6) for rule_id in self.rule_ids},
            "conflict_frequency": round(sum(self.conflict_kinds.values()) / total, 6),
            "conflicts_by_kind": dict(sorted(self.conflict_kinds.items())),
            "conflict_dimensions": dict(sorted(self.conflict_dimensions.items())),
            "policy_activation_frequency": {policy_id: round(count / total, 6) for policy_id, count in sorted(self.policy_activations.items())},
            "policy_activation_counts": dict(sorted(self.policy_activations.items())),
            "average_decision_depth": round(self.decision_depth_total / total, 6),
            "decision_depth_definition": "facts + evidence + conflicts + triggered policies",
            "average_evidence_graph_nodes": round(self.evidence_graph_nodes_total / total, 6),
            "average_evidence_graph_edges": round(self.evidence_graph_edges_total / total, 6),
            "average_audit_size_bytes": round(self.audit_bytes_total / total, 6),
            "rule_contribution": rule_contribution,
            "rule_interactions": interaction_rows,
            "contribution_definition": "Rule present in a decision evidence set; observational co-occurrence, not causal attribution.",
        }


def write_metrics_chart(metrics: dict[str, object], output_path: str | Path) -> Path:
    """Write a dependency-free SVG chart for decision, rule, and policy counts."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions = metrics["decision_distribution"]
    rules = metrics["evidence_distribution"]
    policies = metrics["policy_activation_counts"]
    assert isinstance(decisions, dict) and isinstance(rules, dict) and isinstance(policies, dict)
    groups = (
        ("Decisions", list(decisions.items()), "#4f46e5"),
        ("Rule activations", list(rules.items()), "#b45309"),
        ("Policy activations", list(policies.items()), "#047857"),
    )
    rows = sum(max(1, len(values)) + 2 for _title, values, _colour in groups)
    height = max(360, 42 + rows * 25)
    max_value = max([1] + [value for _title, values, _colour in groups for _label, value in values])
    y = 28
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; fill: #172033; } .title { font-size: 16px; font-weight: 700; } .label { font-size: 12px; } .value { font-size: 12px; font-weight: 600; }</style>',
        '<text x="24" y="20" class="title">AML Decision Runtime v0.2 — label-free runtime metrics</text>',
    ]
    for title, values, colour in groups:
        y += 28
        lines.append(f'<text x="24" y="{y}" class="title">{html.escape(title)}</text>')
        for label, value in values:
            y += 20
            width = round(560 * (value / max_value), 2)
            lines.extend((
                f'<text x="32" y="{y}" class="label">{html.escape(str(label))}</text>',
                f'<rect x="270" y="{y - 13}" width="{width}" height="14" rx="3" fill="{colour}"/>',
                f'<text x="{280 + width}" y="{y}" class="value">{value}</text>',
            ))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_publication_report(report: dict[str, object], output_path: str | Path, chart_path: str | Path) -> Path:
    """Render a publication-style report using only measured runtime metrics."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report["research_metrics"]
    dataset = report["dataset"]
    chronology = report["chronological_sample"]
    performance = report["performance"]
    assert isinstance(metrics, dict) and isinstance(dataset, dict)
    assert isinstance(chronology, dict) and isinstance(performance, dict)
    contribution = metrics["rule_contribution"]
    interactions = metrics["rule_interactions"]
    policies = metrics["policy_activation_counts"]
    decisions = metrics["decision_distribution"]
    assert isinstance(contribution, dict) and isinstance(interactions, list)
    assert isinstance(policies, dict) and isinstance(decisions, dict)
    rule_rows = []
    for rule_id, item in contribution.items():
        assert isinstance(item, dict)
        rule_rows.append((rule_id, item["activation_count"], item["contribution_to_allow"], item["contribution_to_review"], item["contribution_to_block"], item["contribution_to_abstain"], item["average_confidence"]))
    interaction_rows = []
    for item in interactions[:12]:
        assert isinstance(item, dict)
        interaction_rows.append((item["rule_a"], item["rule_b"], item["coactivation_count"]))
    metric_rows = (
        ("Evidence entropy (rule distribution, bits)", metrics["evidence_entropy_bits"]),
        ("Evidence-count entropy (bits)", metrics["evidence_count_entropy_bits"]),
        ("Average evidence count", metrics["average_evidence_count"]),
        ("Conflict frequency per transaction", metrics["conflict_frequency"]),
        ("Average decision depth", metrics["average_decision_depth"]),
        ("Average evidence-graph nodes", metrics["average_evidence_graph_nodes"]),
        ("Average evidence-graph edges", metrics["average_evidence_graph_edges"]),
        ("Average audit size (bytes)", metrics["average_audit_size_bytes"]),
    )
    conflict_rows = (("Conflicts by kind", metrics["conflicts_by_kind"]), ("Conflict dimensions", metrics["conflict_dimensions"]))
    benchmark_rows = (
        ("Scan and chronological selection", f"{performance['scan_and_selection_seconds']:.3f} s"),
        ("Decision loop", f"{performance['runtime_seconds']:.3f} s"),
        ("End to end", f"{performance['end_to_end_seconds']:.3f} s"),
        ("Mean latency", f"{performance['average_latency_ms']:.3f} ms"),
        ("p95 latency", f"{performance['p95_latency_ms']:.3f} ms"),
        ("Peak memory", f"{performance['peak_memory_bytes']:,} bytes"),
    )
    lines = [
        "# Deterministic AML Decision Runtime v0.2", "",
        "## Background", "",
        "This experiment studies deterministic decision-runtime behaviour, not fraud-detection accuracy. IBM laundering labels are read only after each decision for exploratory evaluation; they never enter facts, evidence, conflicts, policies, decisions, graph state, or audits.", "",
        "## Architecture", "",
        "The architecture remains `Transaction → Facts → Evidence → Conflicts → Policies → Decision → Audit`. v0.2 adds evidence-quality metadata, declared semantic conflicts, configurable policy thresholds, and label-free measurement without adding machine learning.", "",
        "## Dataset", "",
        f"`{dataset['transaction_file']}` contained {dataset['source_rows']:,} validated rows and {dataset['rejected_rows']:,} rejected rows. The causal sample contains {chronology['processed_transactions']:,} transactions from {chronology['first_timestamp']} to {chronology['last_timestamp']} across {chronology['unique_accounts']:,} accounts.", "",
        "The public schema omits independent SAR, KYC, jurisdiction, sanctions, and control evidence. v0.2 does not manufacture those attributes.", "",
        "## Runtime and decision flow", "",
        "An isolated new-beneficiary signal is low-severity evidence and can be allowed. Review requires noisy-OR effective risk of at least 0.90 or a designated single high-concern rule. Block requires unqualified SAR evidence or the independently corroborated large-value, velocity, and behavioural combination at effective risk of at least 0.995. These frozen defaults are not label-fitted.", "",
        "## Decision and evidence metrics", "",
        _markdown_table(("Decision", "Count"), tuple((name, count) for name, count in decisions.items())), "",
        _markdown_table(("Metric", "Value"), metric_rows), "",
        f"![Decision, rule, and policy activation chart]({chart_path})", "",
        "## Rule analysis", "",
        "Contribution is observational co-occurrence of evidence with the final decision, not causal attribution.", "",
        _markdown_table(("Rule", "Activations", "ALLOW", "REVIEW", "BLOCK", "ABSTAIN", "Mean confidence"), rule_rows), "",
        "### Rule interactions", "",
        _markdown_table(("Rule A", "Rule B", "Coactivations"), interaction_rows), "",
        "## Policy analysis", "",
        _markdown_table(("Policy", "Triggered"), tuple((policy, count) for policy, count in policies.items())), "",
        "Every policy audit stores effective risk, qualified/unqualified risk counts, and the configured review/block thresholds.", "",
        "## Conflict analysis", "",
        "Conflicts are first-class positive-versus-negative evidence objects. They carry confidence and source-reliability values plus dimensions for confidence asymmetry, source-strength asymmetry, and old-versus-recent supersession when applicable. Absence of a control in the IBM schema is reported as missing data coverage rather than treated as a contradiction.", "",
        _markdown_table(("Conflict metric", "Value"), conflict_rows), "",
        "## Benchmark", "",
        _markdown_table(("Metric", "Measured value"), benchmark_rows), "",
        "## Limitations", "",
        "This is not an AML effectiveness study. The sample is an early ten-minute causal window; rule confidence is an explicit policy parameter, not a calibrated probability; and contribution is association, not intervention. Per-transaction JSON audits are also a deliberate transparency/storage tradeoff.", "",
        "## Future work", "",
        "Evaluate frozen configurations on later chronological windows and other datasets; attach independently sourced KYC, jurisdiction, SAR, and control feeds; compare policy changes through controlled ablations; and add counterfactual contribution analysis without consulting laundering labels.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
