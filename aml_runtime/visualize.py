"""Graphviz DOT rendering for a decision execution trace."""

from __future__ import annotations

from pathlib import Path

from .runtime import RuntimeResult


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_decision_graph(result: RuntimeResult, output_path: str | Path) -> Path:
    """Write a portable DOT file; use `dot -Tpng input.dot -o output.png` if desired."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tx = result.transaction
    lines = [
        "digraph aml_decision {", "  rankdir=LR;", "  graph [fontname=Helvetica];", "  node [fontname=Helvetica];",
        f'  tx [shape=box, style="rounded,filled", fillcolor="#dbeafe", label="Transaction\\n{_escape(tx.id)}\\n${tx.amount:,.2f}"];',
        f'  origin [shape=ellipse, label="Account\\n{_escape(tx.originator_account_id)}"];',
        f'  beneficiary [shape=ellipse, label="Account\\n{_escape(tx.beneficiary_account_id)}"];',
        "  origin -> tx [label=originates];", "  tx -> beneficiary [label=beneficiary];",
    ]
    for index, fact in enumerate(result.facts):
        node = f"fact_{index}"
        lines.append(f'  {node} [shape=note, fillcolor="#fef3c7", style=filled, label="Fact\\n{_escape(fact.type.value)}"];')
        lines.append(f"  tx -> {node};")
    for index, evidence in enumerate(result.evidence):
        node = f"evidence_{index}"
        colour = "#fee2e2" if evidence.direction == "risk" else "#dcfce7"
        lines.append(f'  {node} [shape=component, fillcolor="{colour}", style=filled, label="Evidence\\n{_escape(evidence.rule_id)}"];')
        for fact_index, fact in enumerate(result.facts):
            if fact.id in evidence.supporting_facts:
                lines.append(f"  fact_{fact_index} -> {node};")
    lines.append(f'  decision [shape=doubleoctagon, style="filled,bold", fillcolor="#ddd6fe", label="Decision\\n{result.decision.decision.value}"];')
    for index, evidence in enumerate(result.evidence):
        lines.append(f"  evidence_{index} -> decision;")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
