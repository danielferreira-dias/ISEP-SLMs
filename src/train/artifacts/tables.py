"""Atomic CSV and LaTeX tables ready for dissertation inclusion."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from src.train.evaluation.models import ClassificationMetrics

from .store import atomic_write_text
from .types import TableCell


def write_csv_table(
    path: Path,
    headers: tuple[str, ...],
    rows: tuple[tuple[TableCell, ...], ...],
) -> Path:
    """Write a rectangular RFC-4180-compatible UTF-8 CSV atomically."""

    if not headers:
        raise ValueError("CSV table requires at least one column")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("CSV rows must have the same width as headers")
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    return path


def _latex_escape(value: TableCell) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_latex_table(
    path: Path,
    headers: tuple[str, ...],
    rows: tuple[tuple[TableCell, ...], ...],
    *,
    caption: str,
    label: str,
) -> Path:
    """Write a dependency-free booktabs LaTeX table atomically."""

    if not headers or any(len(row) != len(headers) for row in rows):
        raise ValueError("LaTeX table must be rectangular and non-empty")
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(_latex_escape(value) for value in row) + r" \\" for row in rows
    )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def classification_table_rows(
    metrics: ClassificationMetrics,
) -> tuple[tuple[TableCell, ...], ...]:
    """Return per-class rows with fixed precision for thesis tables."""

    return tuple(
        (
            item.label,
            item.support,
            item.predicted_count,
            f"{item.precision:.6f}",
            f"{item.recall:.6f}",
            f"{item.f1:.6f}",
        )
        for item in metrics.per_class
    )


def write_classification_tables(
    directory: Path,
    metrics: ClassificationMetrics,
    *,
    stem: str = "per_class_metrics",
) -> tuple[Path, Path]:
    """Write matching CSV and LaTeX per-class metric tables."""

    headers = (
        "label",
        "support",
        "predicted_count",
        "precision",
        "recall",
        "f1",
    )
    rows = classification_table_rows(metrics)
    csv_path = write_csv_table(directory / f"{stem}.csv", headers, rows)
    latex_path = write_latex_table(
        directory / f"{stem}.tex",
        headers,
        rows,
        caption="Per-class label-only evaluation metrics",
        label=f"tab:{stem}",
    )
    return csv_path, latex_path
