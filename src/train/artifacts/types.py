"""Shared serializable types for training artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type TableCell = str | int | float | bool | None
type RunStatus = Literal["created", "running", "completed", "failed", "interrupted"]


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """Paths for a figure and the exact source data used to draw it."""

    name: str
    png_path: Path
    svg_path: Path
    source_csv_path: Path


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Human-readable report outputs for one training run."""

    markdown_path: Path
    html_path: Path
    metrics_csv_path: Path
    metrics_latex_path: Path


@dataclass(frozen=True, slots=True)
class PredictionArtifacts:
    """Machine-readable prediction outputs for one checkpoint."""

    csv_path: Path
    parquet_path: Path


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Result of a non-overwriting thesis artefact export."""

    copied: tuple[Path, ...]
    skipped_existing: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunSnapshotArtifacts:
    """Canonical files required to reconstruct one comparable run."""

    metadata_path: Path
    contract_path: Path
    metrics_path: Path
    predictions: PredictionArtifacts


@dataclass(frozen=True, slots=True)
class ComparisonArtifacts:
    """Machine-readable and thesis-ready paired comparison outputs."""

    json_path: Path
    csv_path: Path
    latex_path: Path
    markdown_path: Path
    html_path: Path
    figure: FigureArtifact | None
