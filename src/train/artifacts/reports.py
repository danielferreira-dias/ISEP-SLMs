"""Self-contained HTML and Markdown reports for thesis-ready run results."""

from __future__ import annotations

import base64
import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path

from src.train.evaluation.models import ClassificationMetrics

from .serialization import classification_metrics_from_json
from .store import ArtifactStore, atomic_write_text
from .tables import (
    write_classification_tables,
    write_csv_table,
    write_latex_table,
)
from .types import ReportArtifacts, TableCell


@dataclass(frozen=True, slots=True)
class _ReportContext:
    """Optional scientific run metadata embedded in human-readable reports."""

    resolved_config_json: str | None
    best_checkpoint: tuple[tuple[str, str], ...]
    resource_summary: tuple[tuple[str, str], ...]


def _load_metrics(store: ArtifactStore) -> ClassificationMetrics:
    path = store.path("metrics", "classification.json")
    if not path.is_file():
        raise FileNotFoundError(f"Classification metrics are missing: {path}")
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return classification_metrics_from_json(payload)


def _overall_rows(
    metrics: ClassificationMetrics,
) -> tuple[tuple[TableCell, ...], ...]:
    return (
        ("Samples", metrics.sample_count),
        ("Top-1 accuracy", f"{metrics.top1_accuracy:.6f}"),
        ("Macro-F1", f"{metrics.macro_f1:.6f}"),
        ("Balanced accuracy", f"{metrics.balanced_accuracy:.6f}"),
        ("Invalid-output rate", f"{metrics.invalid_rate:.6f}"),
    )


def _formatted_json_if_present(path: Path) -> str | None:
    """Load and deterministically format an optional JSON document."""

    if not path.is_file():
        return None
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _mapping_rows_if_present(path: Path) -> tuple[tuple[str, str], ...]:
    """Load an optional JSON object as stable display rows."""

    if not path.is_file():
        return ()
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Report metadata must be a JSON object: {path}")
    return tuple(
        (str(key), _display_value(value))
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
    )


def _display_value(value: object) -> str:
    """Format one JSON-derived report value without losing nested structure."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str | int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _resource_rows_if_present(path: Path) -> tuple[tuple[str, str], ...]:
    """Read the thesis resource summary table when it has been generated."""

    if not path.is_file():
        return ()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["metric", "value"]:
            raise ValueError(f"Unexpected resource-summary columns: {path}")
        rows: list[tuple[str, str]] = []
        for index, row in enumerate(reader):
            metric = row.get("metric")
            value = row.get("value")
            if metric is None or value is None:
                raise ValueError(f"Invalid resource-summary row {index}: {path}")
            rows.append((metric, value))
    return tuple(rows)


def _report_context(store: ArtifactStore) -> _ReportContext:
    """Collect optional configuration, selection, and cost evidence."""

    return _ReportContext(
        resolved_config_json=_formatted_json_if_present(
            store.path("manifests", "config.resolved.json")
        ),
        best_checkpoint=_mapping_rows_if_present(
            store.path("manifests", "best_checkpoint.json")
        ),
        resource_summary=_resource_rows_if_present(
            store.path("tables", "resource_summary.csv")
        ),
    )


def _markdown_table(
    heading: str,
    rows: tuple[tuple[str, str], ...],
) -> list[str]:
    """Render optional key-value evidence as a Markdown section."""

    if not rows:
        return []
    lines = ["", f"## {heading}", "", "| Field | Value |", "|---|---|"]
    lines.extend(
        f"| {key.replace('|', '\\|')} | {value.replace('|', '\\|')} |"
        for key, value in rows
    )
    return lines


def _markdown(
    title: str,
    run_directory: Path,
    metrics: ClassificationMetrics,
    limitations: tuple[str, ...],
    context: _ReportContext,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Run: `{run_directory}`",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in _overall_rows(metrics))
    if context.resolved_config_json is not None:
        lines.extend(
            (
                "",
                "## Resolved configuration",
                "",
                "```json",
                context.resolved_config_json,
                "```",
            )
        )
    lines.extend(_markdown_table("Selected checkpoint", context.best_checkpoint))
    lines.extend(_markdown_table("Resource summary", context.resource_summary))
    lines.extend(
        (
            "",
            "## Per-class metrics",
            "",
            "| Label | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        )
    )
    lines.extend(
        "| "
        f"{item.label} | {item.support} | {item.precision:.4f} | "
        f"{item.recall:.4f} | {item.f1:.4f} |"
        for item in metrics.per_class
    )
    lines.extend(("", "## Interpretation constraints", ""))
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- Development-set results; no external benchmark selection.")
    lines.extend(
        (
            "",
            "The CSV files adjacent to every figure are the authoritative "
            "source data for dissertation plots.",
            "",
        )
    )
    return "\n".join(lines)


def _figure_html(figure_directory: Path) -> str:
    blocks: list[str] = []
    for path in sorted(figure_directory.glob("*.png")):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        label = escape(path.stem.replace("_", " ").title())
        blocks.append(
            "<figure>"
            f'<img alt="{label}" src="data:image/png;base64,{encoded}">'
            f"<figcaption>{label}</figcaption>"
            "</figure>"
        )
    return "".join(blocks) or "<p>No figures were generated for this run.</p>"


def _html(
    title: str,
    run_directory: Path,
    metrics: ClassificationMetrics,
    limitations: tuple[str, ...],
    figure_directory: Path,
    context: _ReportContext,
) -> str:
    overall = "".join(
        f"<tr><th>{escape(str(name))}</th><td>{escape(str(value))}</td></tr>"
        for name, value in _overall_rows(metrics)
    )
    per_class = "".join(
        "<tr>"
        f"<th>{escape(item.label)}</th><td>{item.support}</td>"
        f"<td>{item.precision:.4f}</td><td>{item.recall:.4f}</td>"
        f"<td>{item.f1:.4f}</td></tr>"
        for item in metrics.per_class
    )
    limitation_items = limitations or (
        "Development-set results; no external benchmark selection.",
    )
    limitation_html = "".join(f"<li>{escape(item)}</li>" for item in limitation_items)
    configuration_html = (
        "<h2>Resolved configuration</h2>"
        f"<pre>{escape(context.resolved_config_json)}</pre>"
        if context.resolved_config_json is not None
        else ""
    )
    checkpoint_html = _key_value_html("Selected checkpoint", context.best_checkpoint)
    resources_html = _key_value_html("Resource summary", context.resource_summary)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;
margin:2rem auto;padding:0 1rem;color:#172033}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}
th,td{{border:1px solid #ccd3df;padding:.45rem;text-align:right}}
th:first-child{{text-align:left}}
.figures{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
figure{{margin:0;border:1px solid #dde2ea;padding:.75rem}}
img{{width:100%;height:auto}}
code{{overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:1rem}}
</style></head><body><h1>{escape(title)}</h1>
<p>Run: <code>{escape(str(run_directory))}</code></p>
<h2>Overall metrics</h2><table>{overall}</table>
{configuration_html}{checkpoint_html}{resources_html}
<h2>Per-class metrics</h2><table><thead><tr>
<th>Label</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th>
</tr></thead><tbody>{per_class}</tbody></table>
<h2>Figures</h2><div class="figures">{_figure_html(figure_directory)}</div>
<h2>Interpretation constraints</h2><ul>{limitation_html}</ul>
</body></html>"""


def _key_value_html(
    heading: str,
    rows: tuple[tuple[str, str], ...],
) -> str:
    """Render optional key-value evidence as an HTML section."""

    if not rows:
        return ""
    body = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"
        for key, value in rows
    )
    return f"<h2>{escape(heading)}</h2><table>{body}</table>"


def generate_report(
    run_directory: Path,
    *,
    title: str | None = None,
    limitations: tuple[str, ...] = (),
) -> ReportArtifacts:
    """Generate self-contained HTML, Markdown, CSV, and LaTeX reports."""

    store = ArtifactStore.at(run_directory)
    metrics = _load_metrics(store)
    context = _report_context(store)
    report_title = title or f"Training run {run_directory.name}"
    markdown_path = store.path("report", "thesis_summary.md")
    html_path = store.path("report", "report.html")
    atomic_write_text(
        markdown_path,
        _markdown(report_title, run_directory, metrics, limitations, context),
    )
    atomic_write_text(
        html_path,
        _html(
            report_title,
            run_directory,
            metrics,
            limitations,
            store.layout.figures,
            context,
        ),
    )
    overall_rows = _overall_rows(metrics)
    metrics_csv_path = write_csv_table(
        store.path("tables", "overall_metrics.csv"),
        ("metric", "value"),
        overall_rows,
    )
    metrics_latex_path = write_latex_table(
        store.path("tables", "overall_metrics.tex"),
        ("metric", "value"),
        overall_rows,
        caption="Label-only validation metrics",
        label="tab:label_only_validation_metrics",
    )
    write_classification_tables(store.layout.tables, metrics)
    return ReportArtifacts(
        markdown_path=markdown_path,
        html_path=html_path,
        metrics_csv_path=metrics_csv_path,
        metrics_latex_path=metrics_latex_path,
    )
