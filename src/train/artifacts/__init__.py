"""Public storage, plotting, reporting, and comparison API."""

from .comparison import compare_runs
from .export import export_thesis_artifacts
from .plot_models import (
    CheckpointMetricPoint,
    DistributionPoint,
    QualityCostPoint,
    ResourcePoint,
    TrainableParameterPoint,
    TrainingHistoryPoint,
)
from .plotting import PlottingUnavailableError
from .predictions import read_prediction_parquet, write_prediction_files
from .reports import generate_report
from .snapshots import load_comparable_run, write_comparable_run_snapshot
from .store import ArtifactStore, RunLayout, atomic_write_bytes, atomic_write_text
from .thesis_plots import ThesisPlotter
from .types import (
    ComparisonArtifacts,
    ExportResult,
    FigureArtifact,
    PredictionArtifacts,
    ReportArtifacts,
    RunSnapshotArtifacts,
)

__all__ = [
    "ArtifactStore",
    "CheckpointMetricPoint",
    "ComparisonArtifacts",
    "DistributionPoint",
    "ExportResult",
    "FigureArtifact",
    "PlottingUnavailableError",
    "PredictionArtifacts",
    "QualityCostPoint",
    "ReportArtifacts",
    "ResourcePoint",
    "RunLayout",
    "RunSnapshotArtifacts",
    "ThesisPlotter",
    "TrainableParameterPoint",
    "TrainingHistoryPoint",
    "atomic_write_bytes",
    "atomic_write_text",
    "compare_runs",
    "export_thesis_artifacts",
    "generate_report",
    "load_comparable_run",
    "read_prediction_parquet",
    "write_comparable_run_snapshot",
    "write_prediction_files",
]
