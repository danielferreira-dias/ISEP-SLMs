"""Lazy Matplotlib primitives with atomic PNG and SVG output."""

from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .types import FigureArtifact


class PlottingUnavailableError(RuntimeError):
    """Raised when an optional plotting dependency is unavailable."""


class _Axes(Protocol):
    def plot(self, *args: object, **kwargs: object) -> object: ...
    def bar(self, *args: object, **kwargs: object) -> object: ...
    def scatter(self, *args: object, **kwargs: object) -> object: ...
    def imshow(self, *args: object, **kwargs: object) -> object: ...
    def text(self, *args: object, **kwargs: object) -> object: ...
    def set_title(self, value: str) -> object: ...
    def set_xlabel(self, value: str) -> object: ...
    def set_ylabel(self, value: str) -> object: ...
    def set_ylim(self, *args: object, **kwargs: object) -> object: ...
    def set_xticks(self, *args: object, **kwargs: object) -> object: ...
    def set_yticks(self, *args: object, **kwargs: object) -> object: ...
    def legend(self, *args: object, **kwargs: object) -> object: ...
    def grid(self, *args: object, **kwargs: object) -> object: ...


class _Figure(Protocol):
    def tight_layout(self) -> object: ...
    def savefig(self, *args: object, **kwargs: object) -> object: ...
    def colorbar(self, *args: object, **kwargs: object) -> object: ...


class _Pyplot(Protocol):
    def subplots(self, *args: object, **kwargs: object) -> tuple[_Figure, _Axes]: ...
    def close(self, figure: _Figure) -> object: ...


class _Matplotlib(Protocol):
    def use(self, backend: str, *, force: bool = True) -> object: ...


def _load_pyplot() -> _Pyplot:
    try:
        matplotlib = cast(_Matplotlib, importlib.import_module("matplotlib"))
        matplotlib.use("Agg", force=True)
        module = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("matplotlib"):
            raise PlottingUnavailableError(
                "Matplotlib is required to render thesis figures; install "
                "the project's training dependencies"
            ) from exc
        raise
    return cast(_Pyplot, module)


@dataclass(frozen=True, slots=True)
class LineSeries:
    """One named sequence for a line figure."""

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]


def _save_figure(
    pyplot: _Pyplot,
    figure: _Figure,
    *,
    name: str,
    figure_directory: Path,
    source_csv_path: Path,
) -> FigureArtifact:
    figure_directory.mkdir(parents=True, exist_ok=True)
    png_path = figure_directory / f"{name}.png"
    svg_path = figure_directory / f"{name}.svg"
    figure.tight_layout()
    try:
        for destination, file_format in (
            (png_path, "png"),
            (svg_path, "svg"),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=figure_directory,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                figure.savefig(
                    temporary,
                    format=file_format,
                    dpi=180,
                    bbox_inches="tight",
                )
                os.replace(temporary, destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    finally:
        pyplot.close(figure)
    return FigureArtifact(name, png_path, svg_path, source_csv_path)


def line_figure(
    *,
    name: str,
    title: str,
    x_label: str,
    y_label: str,
    series: tuple[LineSeries, ...],
    figure_directory: Path,
    source_csv_path: Path,
    percentage_axis: bool = False,
) -> FigureArtifact:
    """Render one or more line series with a shared numeric axis."""

    if not series or any(len(item.x) != len(item.y) for item in series):
        raise ValueError("Line figure requires non-empty rectangular series")
    pyplot = _load_pyplot()
    figure, axes = pyplot.subplots(figsize=(8.0, 4.8))
    for item in series:
        axes.plot(item.x, item.y, marker="o", linewidth=1.7, label=item.name)
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    if percentage_axis:
        axes.set_ylim(0.0, 1.0)
    if len(series) > 1:
        axes.legend()
    axes.grid(True, alpha=0.25)
    return _save_figure(
        pyplot,
        figure,
        name=name,
        figure_directory=figure_directory,
        source_csv_path=source_csv_path,
    )


def grouped_bar_figure(
    *,
    name: str,
    title: str,
    y_label: str,
    categories: tuple[str, ...],
    series: tuple[tuple[str, tuple[float, ...]], ...],
    figure_directory: Path,
    source_csv_path: Path,
    percentage_axis: bool = False,
) -> FigureArtifact:
    """Render side-by-side bars for one or more category series."""

    if not categories or not series:
        raise ValueError("Grouped bar figure requires categories and series")
    if any(len(values) != len(categories) for _, values in series):
        raise ValueError("Grouped bar series width does not match categories")
    pyplot = _load_pyplot()
    figure, axes = pyplot.subplots(figsize=(max(8.0, len(categories) * 0.45), 5.0))
    width = 0.8 / len(series)
    base = tuple(float(index) for index in range(len(categories)))
    for series_index, (label, values) in enumerate(series):
        positions = tuple(
            value - 0.4 + width / 2.0 + width * series_index for value in base
        )
        axes.bar(positions, values, width=width, label=label)
    axes.set_title(title)
    axes.set_ylabel(y_label)
    axes.set_xticks(base, labels=categories, rotation=45, ha="right")
    if percentage_axis:
        axes.set_ylim(0.0, 1.0)
    if len(series) > 1:
        axes.legend()
    axes.grid(True, axis="y", alpha=0.25)
    return _save_figure(
        pyplot,
        figure,
        name=name,
        figure_directory=figure_directory,
        source_csv_path=source_csv_path,
    )


def heatmap_figure(
    *,
    name: str,
    title: str,
    labels: tuple[str, ...],
    matrix: tuple[tuple[int, ...], ...],
    figure_directory: Path,
    source_csv_path: Path,
) -> FigureArtifact:
    """Render an annotated confusion-matrix heatmap."""

    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise ValueError("Heatmap matrix must be square and match labels")
    pyplot = _load_pyplot()
    size = max(7.0, len(labels) * 0.45)
    figure, axes = pyplot.subplots(figsize=(size, size))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)
    axes.set_title(title)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    ticks = tuple(range(len(labels)))
    axes.set_xticks(ticks, labels=labels, rotation=45, ha="right")
    axes.set_yticks(ticks, labels=labels)
    if len(labels) <= 25:
        maximum = max((max(row, default=0) for row in matrix), default=0)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axes.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if maximum and value > maximum / 2 else "black",
                    fontsize=7,
                )
    return _save_figure(
        pyplot,
        figure,
        name=name,
        figure_directory=figure_directory,
        source_csv_path=source_csv_path,
    )


def scatter_figure(
    *,
    name: str,
    title: str,
    x_label: str,
    y_label: str,
    labels: tuple[str, ...],
    x: tuple[float, ...],
    y: tuple[float, ...],
    figure_directory: Path,
    source_csv_path: Path,
) -> FigureArtifact:
    """Render a labelled quality-versus-cost scatter figure."""

    if not labels or not (len(labels) == len(x) == len(y)):
        raise ValueError("Scatter labels and coordinates must align")
    pyplot = _load_pyplot()
    figure, axes = pyplot.subplots(figsize=(7.0, 5.0))
    axes.scatter(x, y, s=64)
    for label, x_value, y_value in zip(labels, x, y, strict=True):
        axes.text(x_value, y_value, f" {label}", va="center")
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, alpha=0.25)
    return _save_figure(
        pyplot,
        figure,
        name=name,
        figure_directory=figure_directory,
        source_csv_path=source_csv_path,
    )
