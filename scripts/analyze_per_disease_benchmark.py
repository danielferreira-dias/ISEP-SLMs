#!/usr/bin/env python3
"""Create per-diagnosis metrics from completed ISEP benchmark predictions.

The script is deliberately read-only with respect to benchmark inputs. It
does not call a model or regenerate predictions; it derives diagnosis slices
from the JSONL artefacts produced by the deterministic ISEP tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TASKS: tuple[str, ...] = (
    "visual_top_k_closed_set",
    "visual_disease_confusion_sets",
    "evidence_grounded_diagnosis",
)
OPEN_ENDED_TASK = "open_ended_diagnosis"


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One benchmark condition and its model-facing display name."""

    condition: str
    model: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case reduced to ranked diagnosis information."""

    disease_id: str
    ranks: tuple[str, ...]

    @property
    def rank(self) -> int | None:
        """Return the one-indexed rank of the gold disease, if present."""

        try:
            return self.ranks.index(self.disease_id) + 1
        except ValueError:
            return None


def _one_prediction_file(root: Path, condition: str, task: str) -> Path:
    """Resolve exactly one prediction file for a condition/task pair."""

    matches = sorted(
        (
            path
            for path in (
                root / condition / task
            ).glob("*/*/predictions.jsonl")
            if path.is_file()
        ),
        key=lambda path: str(path),
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one predictions.jsonl for {condition}/{task}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _one_judge_metrics_file(root: Path, condition: str) -> Path:
    """Resolve the completed open-ended judge summary for a condition."""

    matches = sorted(
        path
        for path in (
            root / condition / OPEN_ENDED_TASK
        ).glob("*/*/judge_metrics.json")
        if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one judge_metrics.json for {condition}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _ranked_diseases(response: object) -> tuple[str, ...]:
    """Extract canonical ranked disease IDs from a benchmark response."""

    if not isinstance(response, dict):
        return ()
    output = response.get("canonical_output")
    if not isinstance(output, dict):
        output = response.get("parsed_output")
    if not isinstance(output, dict):
        return ()
    candidates = output.get("predictions")
    if not isinstance(candidates, list):
        candidates = output.get("differential")
    if not isinstance(candidates, list):
        return ()
    ranked: list[tuple[int, str]] = []
    for position, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        disease = item.get("disease_id")
        if not isinstance(disease, str) or not disease:
            continue
        raw_rank = item.get("rank", position)
        rank = raw_rank if isinstance(raw_rank, int) else position
        ranked.append((rank, disease))
    ranked.sort(key=lambda pair: pair[0])
    # A repeated ID is not a second independent answer. Keep first occurrence.
    unique: list[str] = []
    for _, disease in ranked:
        if disease not in unique:
            unique.append(disease)
    return tuple(unique)


def _load_cases(path: Path) -> tuple[CaseResult, ...]:
    """Load and reduce one predictions JSONL file."""

    cases: list[CaseResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            disease = payload.get("ground_truth_disease_id")
            if not isinstance(disease, str) or not disease:
                raise ValueError(f"Missing gold disease at {path}:{line_number}")
            cases.append(CaseResult(disease, _ranked_diseases(payload.get("response"))))
    if not cases:
        raise RuntimeError(f"No prediction rows found in {path}")
    return tuple(cases)


def _metrics(cases: Iterable[CaseResult], disease_id: str) -> dict[str, object]:
    """Compute per-diagnosis ranking metrics for one gold disease."""

    selected = tuple(case for case in cases if case.disease_id == disease_id)
    support = len(selected)
    ranks = tuple(case.rank for case in selected)
    valid = tuple(rank for rank in ranks if rank is not None)
    reciprocal = tuple(1.0 / rank for rank in valid)

    def rate(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / support

    return {
        "support": support,
        "valid_prediction_count": len(valid),
        "invalid_prediction_count": support - len(valid),
        "invalid_rate": (support - len(valid)) / support,
        "top_1_accuracy": rate(1),
        "top_3_accuracy": rate(3),
        "top_6_accuracy": rate(6),
        "mean_reciprocal_rank": sum(reciprocal) / support,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    """Write a deterministic UTF-8 CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    """Write canonical indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_taxonomy_names(path: Path) -> dict[str, str]:
    """Read stable disease IDs and display names from the benchmark taxonomy."""

    names: dict[str, str] = {}
    current_id: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        id_match = re.match(r"\s*- id: (D\d+)\s*$", line)
        if id_match:
            current_id = id_match.group(1)
            continue
        name_match = re.match(r"\s*display_name: (.+?)\s*$", line)
        if name_match and current_id is not None:
            names[current_id] = name_match.group(1).strip()
            current_id = None
    return names


def _fmt(value: object) -> str:
    """Format a metric for Markdown."""

    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if 0 <= value <= 1 else f"{value:.4f}"
    return str(value)


def _write_report(
    path: Path,
    root: Path,
    summary: list[dict[str, object]],
    per_disease: list[dict[str, object]],
    open_ended: list[dict[str, object]],
) -> None:
    """Write the thesis-ready Markdown summary and all diagnosis tables."""

    lines = [
        "# ISEPDermaBench — métricas por diagnóstico",
        "",
        "Este relatório foi derivado das predições existentes; não houve nova inferência.",
        "A comparação usa a campanha histórica `temperature=0.6`, `thinking off`, com",
        "o mesmo cohort para Frozen Vision e Vision LoRA.",
        "",
        "## Métricas globais",
        "",
        "| Modelo | Tarefa | N | Top-1 | Top-3 | Top-6 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {task} | {n} | {top1} | {top3} | {top6} | {mrr} |".format(
                model=row["model"],
                task=row["task"],
                n=row["sample_count"],
                top1=_fmt(row["top_1_accuracy"]),
                top3=_fmt(row["top_3_accuracy"]),
                top6=_fmt(row["top_6_accuracy"]),
                mrr=_fmt(row["mean_reciprocal_rank"]),
            )
        )
    lines.extend(
        [
            "",
            "## Open-ended",
            "",
            "Os scores open-ended abaixo vêm dos `judge_metrics.json` já preservados; não foi submetida uma nova avaliação.",
            "",
            "| Modelo | Cobertura | N julgados | Top-1 | Top-3 | MRR | Evidence 0–4 | Unsupported claims |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in open_ended:
        lines.append(
            "| {model} | {coverage} | {evaluated} | {top1} | {top3} | {mrr} | {evidence} | {unsupported} |".format(
                model=row["model"],
                coverage=_fmt(row["judge_coverage"]),
                evaluated=row["evaluated_total"],
                top1=_fmt(row["judge_top_1_accuracy"]),
                top3=_fmt(row["judge_top_3_accuracy"]),
                mrr=_fmt(row["judge_mean_reciprocal_rank"]),
                evidence=f"{float(row['mean_evidence_grounding']):.2f}",
                unsupported=_fmt(row["unsupported_claim_rate"]),
            )
        )

    for task in TASKS:
        task_rows = [row for row in per_disease if row["task"] == task]
        lines.extend(
            [
                "",
                f"## {task} — por diagnóstico",
                "",
                "`top_1_accuracy` é o recall da doença: casos dessa doença classificados em primeiro lugar.",
                "",
                "| Doença | ID | N | Frozen Top-1 | Vision Top-1 | Δ Top-1 | Frozen Top-3 | Vision Top-3 | Frozen Top-6 | Vision Top-6 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        grouped: dict[str, dict[str, dict[str, object]]] = {}
        for row in task_rows:
            grouped.setdefault(str(row["disease_id"]), {})[str(row["model"])] = row
        for disease in sorted(grouped):
            frozen = grouped[disease].get("Frozen Vision", {})
            vision = grouped[disease].get("Vision LoRA", {})
            f1 = frozen.get("top_1_accuracy")
            v1 = vision.get("top_1_accuracy")
            delta = v1 - f1 if isinstance(v1, float) and isinstance(f1, float) else None
            lines.append(
                "| {name} | {disease} | {n} | {f1} | {v1} | {delta} | {f3} | {v3} | {f6} | {v6} |".format(
                    name=frozen.get("disease_name", vision.get("disease_name", "—")),
                    disease=disease,
                    n=frozen.get("support", vision.get("support", "—")),
                    f1=_fmt(f1),
                    v1=_fmt(v1),
                    delta=(f"{delta * 100:+.2f} pp" if isinstance(delta, float) else "—"),
                    f3=_fmt(frozen.get("top_3_accuracy")),
                    v3=_fmt(vision.get("top_3_accuracy")),
                    f6=_fmt(frozen.get("top_6_accuracy")),
                    v6=_fmt(vision.get("top_6_accuracy")),
                )
            )
    lines.extend(
        [
            "",
            "## Proveniência",
            "",
            f"Input root: `{root}`",
            "",
            "Os resultados por diagnóstico são slices do mesmo conjunto de casos e não substituem a comparação global. Tarefas com conjuntos de candidatos ou protocolos diferentes devem permanecer separadas.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(output: Path, rows: list[dict[str, object]]) -> None:
    """Create per-task SVG/PNG diagnosis plots when Matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        diseases = sorted({str(row["disease_id"]) for row in task_rows})
        labels = {
            str(row["disease_id"]): str(row["disease_name"])
            for row in task_rows
        }
        frozen = {str(row["disease_id"]): float(row["top_1_accuracy"]) for row in task_rows if row["model"] == "Frozen Vision"}
        vision = {str(row["disease_id"]): float(row["top_1_accuracy"]) for row in task_rows if row["model"] == "Vision LoRA"}
        x = list(range(len(diseases)))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(10, len(diseases) * 0.65), 5.5))
        ax.bar([i - width / 2 for i in x], [frozen.get(d, math.nan) for d in diseases], width, label="Frozen Vision")
        ax.bar([i + width / 2 for i in x], [vision.get(d, math.nan) for d in diseases], width, label="Vision LoRA")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Top-1 por doença (recall)")
        ax.set_title(f"ISEPDermaBench — {task}")
        ax.set_xticks(x, [labels[d] for d in diseases], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        stem = figure_dir / f"per_disease_top1_{task}"
        fig.savefig(stem.with_suffix(".png"), dpi=180)
        fig.savefig(stem.with_suffix(".svg"))
        plt.close(fig)


def build_report(root: Path, output: Path, taxonomy_path: Path) -> None:
    """Build all per-diagnosis artefacts for the frozen historical campaign."""

    names = _load_taxonomy_names(taxonomy_path)
    specs = (
        RunSpec("frozen", "Frozen Vision"),
        RunSpec("vision", "Vision LoRA"),
    )
    summary: list[dict[str, object]] = []
    open_ended: list[dict[str, object]] = []
    per_disease: list[dict[str, object]] = []
    source_manifest: dict[str, object] = {}
    for spec in specs:
        for task in TASKS:
            prediction_path = _one_prediction_file(root, spec.condition, task)
            cases = _load_cases(prediction_path)
            metrics_path = prediction_path.with_name("metrics.json")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            source_manifest[f"{spec.condition}/{task}"] = {
                "predictions": str(prediction_path),
                "metrics": str(metrics_path),
                "sample_count": len(cases),
            }
            diseases = sorted({case.disease_id for case in cases})
            values = [_metrics(cases, disease) for disease in diseases]
            for disease, values_for_disease in zip(diseases, values, strict=True):
                per_disease.append(
                    {
                        "condition": spec.condition,
                        "model": spec.model,
                        "task": task,
                        "disease_id": disease,
                        "disease_name": names.get(disease, disease),
                        **values_for_disease,
                    }
                )
            top1 = metrics.get("canonical_top_1_accuracy", metrics.get("top_1_accuracy"))
            top3 = metrics.get("canonical_top_3_accuracy", metrics.get("top_3_accuracy"))
            top6 = metrics.get("canonical_top_6_accuracy", metrics.get("top_6_accuracy"))
            mrr = metrics.get("canonical_mean_reciprocal_rank", metrics.get("mean_reciprocal_rank"))
            summary.append(
                {
                    "condition": spec.condition,
                    "model": spec.model,
                    "task": task,
                    "sample_count": len(cases),
                    "top_1_accuracy": top1,
                    "top_3_accuracy": top3,
                    "top_6_accuracy": top6,
                    "mean_reciprocal_rank": mrr,
                }
            )

        open_prediction_path = _one_prediction_file(root, spec.condition, OPEN_ENDED_TASK)
        open_metrics_path = open_prediction_path.with_name("metrics.json")
        judge_metrics_path = _one_judge_metrics_file(root, spec.condition)
        judge_metrics = json.loads(judge_metrics_path.read_text(encoding="utf-8"))
        source_manifest[f"{spec.condition}/{OPEN_ENDED_TASK}"] = {
            "predictions": str(open_prediction_path),
            "metrics": str(open_metrics_path),
            "judge_metrics": str(judge_metrics_path),
            "sample_count": json.loads(open_metrics_path.read_text(encoding="utf-8")).get("total", 300),
        }
        open_ended.append(
            {
                "condition": spec.condition,
                "model": spec.model,
                "judge_coverage": judge_metrics["judge_coverage"],
                "evaluated_total": judge_metrics["evaluated_total"],
                "judge_top_1_accuracy": judge_metrics["judge_top_1_accuracy"],
                "judge_top_3_accuracy": judge_metrics["judge_top_3_accuracy"],
                "judge_mean_reciprocal_rank": judge_metrics["judge_mean_reciprocal_rank"],
                "mean_evidence_grounding": judge_metrics["mean_evidence_grounding"],
                "unsupported_claim_rate": judge_metrics["unsupported_claim_rate"],
            }
        )

    fields = (
        "condition",
        "model",
        "task",
        "disease_id",
        "disease_name",
        "support",
        "valid_prediction_count",
        "invalid_prediction_count",
        "invalid_rate",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_6_accuracy",
        "mean_reciprocal_rank",
    )
    _write_csv(output / "per_disease_metrics.csv", per_disease, fields)
    _write_csv(
        output / "summary.csv",
        summary,
        (
            "condition",
            "model",
            "task",
            "sample_count",
            "top_1_accuracy",
            "top_3_accuracy",
            "top_6_accuracy",
            "mean_reciprocal_rank",
        ),
    )

    comparison: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in per_disease:
        grouped.setdefault((str(row["task"]), str(row["disease_id"])), {})[str(row["model"])] = row
    for (task, disease), models in sorted(grouped.items()):
        frozen = models.get("Frozen Vision")
        vision = models.get("Vision LoRA")
        comparison.append(
            {
                "task": task,
                "disease_id": disease,
                "disease_name": (
                    frozen.get("disease_name") if frozen else vision.get("disease_name")
                ),
                "support_frozen": frozen.get("support") if frozen else None,
                "support_vision": vision.get("support") if vision else None,
                "frozen_top_1_accuracy": frozen.get("top_1_accuracy") if frozen else None,
                "vision_top_1_accuracy": vision.get("top_1_accuracy") if vision else None,
                "delta_top_1_accuracy": (
                    vision["top_1_accuracy"] - frozen["top_1_accuracy"]
                    if frozen and vision
                    else None
                ),
                "frozen_top_3_accuracy": frozen.get("top_3_accuracy") if frozen else None,
                "vision_top_3_accuracy": vision.get("top_3_accuracy") if vision else None,
                "frozen_top_6_accuracy": frozen.get("top_6_accuracy") if frozen else None,
                "vision_top_6_accuracy": vision.get("top_6_accuracy") if vision else None,
            }
        )
    _write_csv(
        output / "comparison.csv",
        comparison,
        tuple(comparison[0].keys()) if comparison else ("task", "disease_id"),
    )
    _write_csv(
        output / "open_ended_summary.csv",
        open_ended,
        (
            "condition",
            "model",
            "judge_coverage",
            "evaluated_total",
            "judge_top_1_accuracy",
            "judge_top_3_accuracy",
            "judge_mean_reciprocal_rank",
            "mean_evidence_grounding",
            "unsupported_claim_rate",
        ),
    )
    _write_report(output / "report.md", root, summary, per_disease, open_ended)
    _write_json(output / "source_manifest.json", source_manifest)
    _plot(output, per_disease)


def main() -> None:
    """Parse arguments and build the per-diagnosis report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/e1_epoch3_historical_t06_benchmarks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark_per_disease_v1/e1_epoch3_historical_t06"),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("data/benchmarks/ISEPDermaBench/artifacts/taxonomies/diseases.yaml"),
    )
    args = parser.parse_args()
    build_report(args.input_root, args.output_dir, args.taxonomy)
    print(args.output_dir)


if __name__ == "__main__":
    main()
