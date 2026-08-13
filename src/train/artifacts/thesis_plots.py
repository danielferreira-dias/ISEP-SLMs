"""High-level thesis figures with exact adjacent CSV source data."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from src.train.evaluation.models import ClassificationMetrics

from .plot_models import (
    CheckpointMetricPoint,
    DistributionPoint,
    QualityCostPoint,
    ResourcePoint,
    TrainableParameterPoint,
    TrainingHistoryPoint,
)
from .plotting import (
    LineSeries,
    grouped_bar_figure,
    heatmap_figure,
    line_figure,
)
from .secondary_plots import (
    plot_data_distribution,
    plot_quality_cost,
    plot_resource_usage,
    plot_trainable_parameters,
)
from .tables import write_csv_table
from .types import FigureArtifact, TableCell


def _finite(values: tuple[float, ...], context: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{context} contains non-finite values")


class ThesisPlotter:
    """Render the standard figure set for one training run."""

    def __init__(self, figure_directory: Path) -> None:
        """Initialize a plotter whose figures and CSV files stay together."""

        self.figure_directory = figure_directory

    def _source(
        self,
        name: str,
        headers: tuple[str, ...],
        rows: tuple[tuple[TableCell, ...], ...],
    ) -> Path:
        return write_csv_table(self.figure_directory / f"{name}.csv", headers, rows)

    def training_history(
        self, points: tuple[TrainingHistoryPoint, ...]
    ) -> tuple[FigureArtifact, ...]:
        """Plot optimization loss and learning rate against global step."""

        if not points:
            return ()
        source = self._source(
            "training_history_source",
            ("step", "epoch", "train_loss", "eval_loss", "learning_rate"),
            tuple(
                (
                    point.step,
                    point.epoch,
                    point.train_loss,
                    point.eval_loss,
                    point.learning_rate,
                )
                for point in points
            ),
        )
        figures: list[FigureArtifact] = []
        loss_series: list[LineSeries] = []
        selectors: tuple[
            tuple[str, Callable[[TrainingHistoryPoint], float | None]], ...
        ] = (
            ("train_loss", lambda item: item.train_loss),
            ("eval_loss", lambda item: item.eval_loss),
        )
        for name, selector in selectors:
            filtered: list[tuple[float, float]] = []
            for point in points:
                value = selector(point)
                if value is not None:
                    filtered.append((float(point.step), value))
            if filtered:
                values = tuple(float(value) for _, value in filtered)
                _finite(values, name)
                loss_series.append(
                    LineSeries(
                        name=name,
                        x=tuple(step for step, _ in filtered),
                        y=values,
                    )
                )
        if loss_series:
            figures.append(
                line_figure(
                    name="loss_curves",
                    title="Training and validation loss",
                    x_label="Global step",
                    y_label="Loss",
                    series=tuple(loss_series),
                    figure_directory=self.figure_directory,
                    source_csv_path=source,
                )
            )
        learning_rate = tuple(
            (float(point.step), point.learning_rate)
            for point in points
            if point.learning_rate is not None
        )
        if learning_rate:
            values = tuple(float(value) for _, value in learning_rate)
            _finite(values, "learning_rate")
            figures.append(
                line_figure(
                    name="learning_rate",
                    title="Learning-rate schedule",
                    x_label="Global step",
                    y_label="Learning rate",
                    series=(
                        LineSeries(
                            "learning_rate",
                            tuple(step for step, _ in learning_rate),
                            values,
                        ),
                    ),
                    figure_directory=self.figure_directory,
                    source_csv_path=source,
                )
            )
        return tuple(figures)

    def checkpoint_metrics(
        self, points: tuple[CheckpointMetricPoint, ...]
    ) -> tuple[FigureArtifact, ...]:
        """Plot quality and loss across evaluated checkpoints."""

        if not points:
            return ()
        source = self._source(
            "checkpoint_metrics_source",
            (
                "checkpoint_id",
                "epoch",
                "top1_accuracy",
                "macro_f1",
                "balanced_accuracy",
                "eval_loss",
            ),
            tuple(
                (
                    point.checkpoint_id,
                    point.epoch,
                    point.top1_accuracy,
                    point.macro_f1,
                    point.balanced_accuracy,
                    point.eval_loss,
                )
                for point in points
            ),
        )
        epochs = tuple(point.epoch for point in points)
        quality = (
            LineSeries(
                "top1_accuracy",
                epochs,
                tuple(point.top1_accuracy for point in points),
            ),
            LineSeries(
                "macro_f1",
                epochs,
                tuple(point.macro_f1 for point in points),
            ),
            LineSeries(
                "balanced_accuracy",
                epochs,
                tuple(point.balanced_accuracy for point in points),
            ),
        )
        for item in quality:
            _finite(item.y, item.name)
        losses = tuple(point.eval_loss for point in points)
        _finite(losses, "checkpoint eval_loss")
        return (
            line_figure(
                name="checkpoint_quality",
                title="Validation quality by checkpoint",
                x_label="Epoch",
                y_label="Score",
                series=quality,
                figure_directory=self.figure_directory,
                source_csv_path=source,
                percentage_axis=True,
            ),
            line_figure(
                name="checkpoint_eval_loss",
                title="Validation loss by checkpoint",
                x_label="Epoch",
                y_label="Evaluation loss",
                series=(LineSeries("eval_loss", epochs, losses),),
                figure_directory=self.figure_directory,
                source_csv_path=source,
            ),
        )

    def per_class_metrics(self, metrics: ClassificationMetrics) -> FigureArtifact:
        """Plot precision, recall, and F1 for every canonical class."""

        source = self._source(
            "per_class_metrics_source",
            ("label", "support", "precision", "recall", "f1"),
            tuple(
                (
                    item.label,
                    item.support,
                    item.precision,
                    item.recall,
                    item.f1,
                )
                for item in metrics.per_class
            ),
        )
        return grouped_bar_figure(
            name="per_class_metrics",
            title="Per-class validation metrics",
            y_label="Score",
            categories=tuple(item.label for item in metrics.per_class),
            series=(
                (
                    "precision",
                    tuple(item.precision for item in metrics.per_class),
                ),
                ("recall", tuple(item.recall for item in metrics.per_class)),
                ("f1", tuple(item.f1 for item in metrics.per_class)),
            ),
            figure_directory=self.figure_directory,
            source_csv_path=source,
            percentage_axis=True,
        )

    def confusion_matrix(self, metrics: ClassificationMetrics) -> FigureArtifact:
        """Plot the canonical-label confusion matrix."""

        source = self._source(
            "confusion_matrix_source",
            ("true_label", "predicted_label", "count"),
            tuple(
                (true_label, predicted_label, metrics.confusion_matrix[i][j])
                for i, true_label in enumerate(metrics.labels)
                for j, predicted_label in enumerate(metrics.labels)
            ),
        )
        return heatmap_figure(
            name="confusion_matrix",
            title="Validation confusion matrix",
            labels=metrics.labels,
            matrix=metrics.confusion_matrix,
            figure_directory=self.figure_directory,
            source_csv_path=source,
        )

    def data_distribution(
        self, points: tuple[DistributionPoint, ...]
    ) -> FigureArtifact | None:
        """Plot class or source counts split by frozen dataset partition."""

        return plot_data_distribution(self.figure_directory, points)

    def class_distribution(
        self, points: tuple[DistributionPoint, ...]
    ) -> FigureArtifact | None:
        """Plot canonical-class counts for every frozen split."""

        return plot_data_distribution(
            self.figure_directory,
            points,
            name="class_distribution",
            title="Class distribution by frozen split",
        )

    def source_distribution(
        self, points: tuple[DistributionPoint, ...]
    ) -> FigureArtifact | None:
        """Plot source-dataset counts for every frozen split."""

        return plot_data_distribution(
            self.figure_directory,
            points,
            name="source_distribution",
            title="Source distribution by frozen split",
        )

    def trainable_parameters(
        self, points: tuple[TrainableParameterPoint, ...]
    ) -> FigureArtifact | None:
        """Plot trainable LoRA parameter counts by model component."""

        return plot_trainable_parameters(self.figure_directory, points)

    def resource_usage(
        self, points: tuple[ResourcePoint, ...]
    ) -> tuple[FigureArtifact, ...]:
        """Plot each available resource channel against elapsed wall time."""

        return plot_resource_usage(self.figure_directory, points)

    def quality_cost(
        self, points: tuple[QualityCostPoint, ...]
    ) -> FigureArtifact | None:
        """Plot macro-F1 against measured GPU-hours for compared runs."""

        return plot_quality_cost(self.figure_directory, points)
