"""Thesis-ready SKINCON quality figures with adjacent CSV sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.train.artifacts import ArtifactStore
from src.train.artifacts.plotting import LineSeries, grouped_bar_figure, line_figure
from src.train.artifacts.tables import write_csv_table
from src.train.artifacts.types import FigureArtifact


def render_morphology_plots(store: ArtifactStore) -> tuple[FigureArtifact, ...]:
    """Render checkpoint and per-concept morphology figures when available."""

    checkpoints = _checkpoint_rows(store.layout.metrics)
    if not checkpoints:
        return ()
    source = write_csv_table(
        store.layout.figures / "morphology_checkpoint_quality_source.csv",
        (
            "checkpoint_id",
            "epoch",
            "micro_f1",
            "macro_f1",
            "exact_match",
            "invalid_output_rate",
        ),
        checkpoints,
    )
    epochs = tuple(float(row[1]) for row in checkpoints)
    checkpoint_figure = line_figure(
        name="morphology_checkpoint_quality",
        title="SKINCON validation quality by checkpoint",
        x_label="Epoch",
        y_label="Score",
        series=(
            LineSeries("micro_f1", epochs, tuple(float(row[2]) for row in checkpoints)),
            LineSeries("macro_f1", epochs, tuple(float(row[3]) for row in checkpoints)),
            LineSeries(
                "exact_match", epochs, tuple(float(row[4]) for row in checkpoints)
            ),
        ),
        figure_directory=store.layout.figures,
        source_csv_path=source,
        percentage_axis=True,
    )
    per_concept = _best_per_concept(store)
    if not per_concept:
        return (checkpoint_figure,)
    concept_source = write_csv_table(
        store.layout.figures / "morphology_per_concept_source.csv",
        ("concept", "support", "precision", "recall", "f1"),
        per_concept,
    )
    concept_figure = grouped_bar_figure(
        name="morphology_per_concept",
        title="SKINCON per-concept validation metrics",
        y_label="Score",
        categories=tuple(str(row[0]) for row in per_concept),
        series=(
            ("precision", tuple(float(row[2]) for row in per_concept)),
            ("recall", tuple(float(row[3]) for row in per_concept)),
            ("f1", tuple(float(row[4]) for row in per_concept)),
        ),
        figure_directory=store.layout.figures,
        source_csv_path=concept_source,
        percentage_axis=True,
    )
    return checkpoint_figure, concept_figure


def _checkpoint_rows(metrics: Path) -> tuple[tuple[str | float, ...], ...]:
    rows: list[tuple[str | float, ...]] = []
    paths = sorted(metrics.glob("morphology_sft_dev__*.json"))
    for path in paths:
        payload = _object(_read_json(path), "morphology metrics")
        checkpoint = payload.get("checkpoint_id")
        epoch = _number(payload.get("epoch"))
        if not isinstance(checkpoint, str) or epoch is None:
            continue
        rows.append(
            (
                checkpoint,
                epoch,
                _required_number(payload, "micro_f1"),
                _required_number(payload, "macro_f1"),
                _required_number(payload, "exact_match"),
                _required_number(payload, "invalid_output_rate"),
            )
        )
    return tuple(sorted(rows, key=lambda row: float(row[1])))


def _best_per_concept(
    store: ArtifactStore,
) -> tuple[tuple[str | int | float, ...], ...]:
    best_path = store.path("manifests", "best_checkpoint.json")
    if not best_path.is_file():
        return ()
    best = _object(_read_json(best_path), "best checkpoint")
    checkpoint = best.get("checkpoint_id")
    if not isinstance(checkpoint, str):
        return ()
    path = store.path("metrics", f"morphology_sft_dev__{checkpoint}.json")
    if not path.is_file():
        return ()
    payload = _object(_read_json(path), "morphology metrics")
    raw = payload.get("per_concept")
    if not isinstance(raw, list):
        return ()
    rows: list[tuple[str | int | float, ...]] = []
    for item in raw:
        entry = _object(item, "per-concept metric")
        concept = entry.get("concept")
        support = entry.get("support")
        if (
            not isinstance(concept, str)
            or isinstance(support, bool)
            or not isinstance(support, int)
        ):
            raise ValueError("Invalid per-concept morphology metric")
        rows.append(
            (
                concept,
                support,
                _required_number(entry, "precision"),
                _required_number(entry, "recall"),
                _required_number(entry, "f1"),
            )
        )
    return tuple(rows)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Metric must be numeric or null")
    return float(value)


def _required_number(value: Mapping[object, object], key: str) -> float:
    result = _number(value.get(key))
    if result is None:
        raise ValueError(f"Morphology metric {key} is missing")
    return result
