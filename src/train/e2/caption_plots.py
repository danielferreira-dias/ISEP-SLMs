"""Thesis-ready SkinCAP and E2 multitask figures with CSV sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.train.artifacts import ArtifactStore
from src.train.artifacts.plotting import LineSeries, grouped_bar_figure, line_figure
from src.train.artifacts.tables import write_csv_table
from src.train.artifacts.types import FigureArtifact


def render_caption_and_multitask_plots(
    store: ArtifactStore,
) -> tuple[FigureArtifact, ...]:
    """Render caption checkpoint curves and the selected task-score panel."""

    figures: list[FigureArtifact] = []
    caption_rows = _caption_rows(store)
    if caption_rows:
        source = write_csv_table(
            store.layout.figures / "caption_checkpoint_quality_source.csv",
            (
                "checkpoint_id",
                "epoch",
                "caption_task_score",
                "concept_f1",
                "reference_similarity",
                "clinical_compliance",
            ),
            caption_rows,
        )
        epochs = tuple(float(row[1]) for row in caption_rows)
        figures.append(
            line_figure(
                name="caption_checkpoint_quality",
                title="SkinCAP validation quality by checkpoint",
                x_label="Epoch",
                y_label="Score",
                series=tuple(
                    LineSeries(
                        name,
                        epochs,
                        tuple(float(row[index]) for row in caption_rows),
                    )
                    for name, index in (
                        ("caption_task_score", 2),
                        ("concept_f1", 3),
                        ("reference_similarity", 4),
                        ("clinical_compliance", 5),
                    )
                ),
                figure_directory=store.layout.figures,
                source_csv_path=source,
                percentage_axis=True,
            )
        )
    task_rows = _selected_task_rows(store)
    if task_rows:
        source = write_csv_table(
            store.layout.figures / "e2_selected_task_scores_source.csv",
            ("task", "score", "metric"),
            task_rows,
        )
        figures.append(
            grouped_bar_figure(
                name="e2_selected_task_scores",
                title="Selected E2 checkpoint: disaggregated task quality",
                y_label="Score",
                categories=tuple(str(row[0]) for row in task_rows),
                series=(("score", tuple(float(row[1]) for row in task_rows)),),
                figure_directory=store.layout.figures,
                source_csv_path=source,
                percentage_axis=True,
            )
        )
    return tuple(figures)


def _caption_rows(store: ArtifactStore) -> tuple[tuple[str | float, ...], ...]:
    rows: list[tuple[str | float, ...]] = []
    for path in sorted(store.layout.metrics.glob("caption_sft_dev__*.json")):
        payload = _mapping(_read_json(path), "caption metrics")
        checkpoint = payload.get("checkpoint_id")
        epoch = _optional_number(payload.get("epoch"))
        if not isinstance(checkpoint, str) or epoch is None:
            continue
        rows.append(
            (
                checkpoint,
                epoch,
                _number(payload, "caption_task_score"),
                _number(payload, "concept_f1"),
                _number(payload, "reference_similarity_mean"),
                _number(payload, "clinical_compliance_rate"),
            )
        )
    return tuple(sorted(rows, key=lambda row: float(row[1])))


def _selected_task_rows(
    store: ArtifactStore,
) -> tuple[tuple[str | float, ...], ...]:
    best_path = store.path("manifests", "best_checkpoint.json")
    if not best_path.is_file():
        return ()
    best = _mapping(_read_json(best_path), "best checkpoint")
    checkpoint = best.get("checkpoint_id")
    if not isinstance(checkpoint, str):
        return ()
    metric_path = store.path("metrics", f"multitask_sft_dev__{checkpoint}.json")
    if not metric_path.is_file():
        return ()
    payload = _mapping(_read_json(metric_path), "multitask metrics")
    rows: list[tuple[str | float, ...]] = [
        ("Diagnosis", _number(payload, "diagnosis_macro_f1"), "macro_f1"),
        ("SKINCON", _number(payload, "morphology_macro_f1"), "macro_f1"),
    ]
    caption = _optional_number(payload.get("caption_task_score"))
    if caption is not None:
        rows.append(("SkinCAP", caption, "caption_task_score"))
    return tuple(rows)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Metric must be numeric or null")
    return float(value)


def _number(value: Mapping[object, object], key: str) -> float:
    result = _optional_number(value.get(key))
    if result is None:
        raise ValueError(f"Metric {key} is missing")
    return result
