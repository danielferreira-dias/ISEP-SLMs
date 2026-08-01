"""Generate a self-contained HTML overview of the thesis datasets.

The report is intentionally read-only. It summarizes the materialized Parquet
manifests and release metadata without changing any split or training record.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


DEFAULT_OUTPUT = Path("data/reports/datasets_overview.html")
TRAINING_MANIFEST = Path(
    "data/training/dermatology_multimodal_v1/train_images.parquet"
)
TRAINING_RELEASE = Path(
    "data/training/dermatology_multimodal_v1/release/training_release_v1.json"
)
TAXONOMY = Path("configs/taxonomies/diseases.yaml")
VISUAL_TOP_K_DIR = Path(
    "data/benchmarks/derma_isep/visual_top_k_v1/datasets"
)

INTERNAL_SOURCE_IDS = {"fitzpatrick17k_c", "pad_ufes_20", "scin"}
SOURCE_LABELS = {
    "fitzpatrick17k_c": "Fitzpatrick17k-C",
    "pad_ufes_20": "PAD-UFES-20",
    "scin": "SCIN",
    "derm1m": "Derm1M",
    "hiba": "HIBA",
    "ddi": "DDI",
    "skindisnet": "SkinDisNet",
}
ROLE_LABELS = {
    "in_domain_diagnosis": "21-class diagnosis",
    "description_only": "Description only",
    "out_of_domain": "Outside the 21-class taxonomy",
}


def _read_parquet(root: Path, relative_path: Path) -> pd.DataFrame:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required dataset manifest does not exist: {path}")
    return pd.read_parquet(path)


def _format_count(value: int | float) -> str:
    return f"{int(value):,}"


def _display(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    return escape(str(value))


def _table(
    headers: Iterable[str],
    rows: Iterable[Iterable[Any]],
    *,
    table_id: str | None = None,
    numeric_columns: set[int] | None = None,
) -> str:
    numeric_columns = numeric_columns or set()
    identifier = f' id="{escape(table_id)}"' if table_id else ""
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            css = ' class="numeric"' if index in numeric_columns else ""
            cells.append(f"<td{css}>{value}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<div class="table-wrap"><table{identifier}>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _summary(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "images": len(frame),
        "groups": frame["leakage_group_id"].nunique(),
        "classes": frame["disease_id"].dropna().nunique(),
    }


def _bar(value: int, maximum: int, *, color: str = "blue") -> str:
    width = 0 if maximum == 0 else max(1.0, value / maximum * 100)
    return (
        '<div class="bar-track" aria-label="'
        f'{_format_count(value)}"><span class="bar {color}" '
        f'style="width:{width:.2f}%"></span></div>'
    )


def generate_dataset_overview(
    root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """Read the current releases and write a standalone HTML report."""

    root = root.resolve()
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)

    training = _read_parquet(root, TRAINING_MANIFEST)
    release = json.loads((root / TRAINING_RELEASE).read_text(encoding="utf-8"))[
        "release"
    ]
    taxonomy_data = yaml.safe_load((root / TAXONOMY).read_text(encoding="utf-8"))
    diseases = taxonomy_data["diseases"]
    disease_names = {item["id"]: item["display_name"] for item in diseases}

    split_paths = {
        "Original internal Train": VISUAL_TOP_K_DIR / "internal/train.parquet",
        "Benchmark Validation": VISUAL_TOP_K_DIR / "internal/validation.parquet",
        "Sealed Internal Test": VISUAL_TOP_K_DIR / "internal/internal_test.parquet",
        "Internal Benchmark 1000": (
            VISUAL_TOP_K_DIR / "internal/internal_benchmark_1000.parquet"
        ),
        "Internal Test reserve": (
            VISUAL_TOP_K_DIR / "internal/internal_test_reserve.parquet"
        ),
    }
    split_frames = {
        label: _read_parquet(root, path) for label, path in split_paths.items()
    }
    external_paths = {
        "DDI": VISUAL_TOP_K_DIR / "external/external_ddi.parquet",
        "SkinDisNet": VISUAL_TOP_K_DIR / "external/external_skindisnet.parquet",
    }
    external_frames = {
        label: _read_parquet(root, path)
        for label, path in external_paths.items()
    }

    in_domain = training[training["training_role"] == "in_domain_diagnosis"]
    role_summary = (
        training.groupby("training_role", dropna=False)
        .agg(
            images=("sample_id", "size"),
            groups=("leakage_group_id", "nunique"),
            sources=("dataset_id", "nunique"),
        )
        .reset_index()
    )
    source_role_summary = (
        training.groupby(["dataset_id", "training_role"], dropna=False)
        .agg(
            images=("sample_id", "size"),
            groups=("leakage_group_id", "nunique"),
        )
        .reset_index()
        .sort_values(["dataset_id", "training_role"])
    )

    image_pivot = in_domain.pivot_table(
        index="disease_id",
        columns="dataset_id",
        values="sample_id",
        aggfunc="size",
        fill_value=0,
    )
    class_groups = in_domain.groupby("disease_id")["leakage_group_id"].nunique()
    class_totals = in_domain.groupby("disease_id").size()
    source_columns = [
        "fitzpatrick17k_c",
        "pad_ufes_20",
        "scin",
        "derm1m",
        "hiba",
    ]
    max_class_count = int(class_totals.max())

    original_in_domain = int(
        in_domain["dataset_id"].isin(INTERNAL_SOURCE_IDS).shape[0]
    )
    derm1m_in_domain = int((in_domain["dataset_id"] == "derm1m").sum())
    hiba_in_domain = int((in_domain["dataset_id"] == "hiba").sum())
    total_in_domain = len(in_domain)

    confusion_development = _read_parquet(
        root,
        Path(
            "data/benchmarks/derma_isep/visual_confusion_sets_v1/"
            "datasets/development/validation_confusion_tasks.parquet"
        ),
    )
    confusion_final = _read_parquet(
        root,
        Path(
            "data/benchmarks/derma_isep/visual_confusion_sets_v1/"
            "datasets/internal/confusion_tasks.parquet"
        ),
    )
    evidence_development = _read_parquet(
        root,
        Path(
            "data/benchmarks/derma_isep/evidence_grounded_diagnosis_v1/"
            "datasets/development/evidence_grounded_validation.parquet"
        ),
    )
    evidence_external = _read_parquet(
        root,
        Path(
            "data/benchmarks/derma_isep/evidence_grounded_diagnosis_v1/"
            "datasets/external/evidence_grounded_ddi.parquet"
        ),
    )

    skincap = pd.read_csv(
        root
        / "configs/datasets/skincare/data/SkinCAP/skincap_v240623.csv"
    )
    skincon_fitzpatrick = pd.read_csv(
        root
        / "configs/datasets/skincon/data/annotations/annotations_fitzpatrick17k.csv"
    )
    skincon_ddi = pd.read_csv(
        root / "configs/datasets/skincon/data/annotations/annotations_ddi.csv"
    )
    skincon_unusable_column = "Do not consider this image"
    skincon_usable = int(
        (skincon_fitzpatrick[skincon_unusable_column] != 1).sum()
        + (skincon_ddi[skincon_unusable_column] != 1).sum()
    )
    skincot_image_root = root / "configs/datasets/skincare/data/SkinCoT/images"
    skincot_image_count = sum(
        path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        for path in skincot_image_root.rglob("*")
        if path.is_file()
    )

    split_rows = []
    split_uses = {
        "Original internal Train": "Original 21-class training source",
        "Benchmark Validation": "Teacher, prompt, checkpoint and threshold selection",
        "Sealed Internal Test": "Complete final internal audit",
        "Internal Benchmark 1000": "Primary paired before/after comparison",
        "Internal Test reserve": "Protected groups outside the primary 1,000",
    }
    for label, frame in split_frames.items():
        stats = _summary(frame)
        split_rows.append(
            (
                escape(label),
                _format_count(stats["images"]),
                _format_count(stats["groups"]),
                _format_count(stats["classes"]),
                escape(split_uses[label]),
            )
        )

    role_rows = []
    role_descriptions = {
        "in_domain_diagnosis": "May teach findings, differential and one of the 21 diagnoses.",
        "description_only": "May teach morphology and description; no diagnosis may be invented.",
        "out_of_domain": "May teach abstention; it is not a 21-class classification target.",
    }
    for row in role_summary.itertuples(index=False):
        role = str(row.training_role)
        role_rows.append(
            (
                escape(ROLE_LABELS.get(role, role)),
                _format_count(row.images),
                _format_count(row.groups),
                _format_count(row.sources),
                escape(role_descriptions[role]),
            )
        )

    source_role_rows = []
    for row in source_role_summary.itertuples(index=False):
        source_role_rows.append(
            (
                escape(SOURCE_LABELS.get(row.dataset_id, row.dataset_id)),
                escape(ROLE_LABELS.get(row.training_role, row.training_role)),
                _format_count(row.images),
                _format_count(row.groups),
            )
        )

    class_rows = []
    for disease in diseases:
        disease_id = disease["id"]
        total = int(class_totals.get(disease_id, 0))
        row = [
            f'<span class="mono">{escape(disease_id)}</span>',
            escape(disease["display_name"]),
            _format_count(total),
            _format_count(int(class_groups.get(disease_id, 0))),
        ]
        for source in source_columns:
            value = 0
            if disease_id in image_pivot.index and source in image_pivot.columns:
                value = int(image_pivot.loc[disease_id, source])
            row.append(_format_count(value))
        row.append(_bar(total, max_class_count))
        class_rows.append(row)

    external_rows = []
    external_class_rows = []
    for label, frame in external_frames.items():
        stats = _summary(frame)
        external_rows.append(
            (
                escape(label),
                _format_count(stats["images"]),
                _format_count(stats["groups"]),
                _format_count(stats["classes"]),
                "External generalization only",
            )
        )
        for disease_id, group in frame.groupby("disease_id"):
            external_class_rows.append(
                (
                    escape(label),
                    f'<span class="mono">{escape(str(disease_id))}</span>',
                    escape(disease_names.get(str(disease_id), str(disease_id))),
                    _format_count(len(group)),
                    _format_count(group["leakage_group_id"].nunique()),
                )
            )

    benchmark_rows = [
        (
            "Visual Top-K",
            "Development",
            _format_count(len(split_frames["Benchmark Validation"])),
            _format_count(
                split_frames["Benchmark Validation"]["leakage_group_id"].nunique()
            ),
            "21-class ranking",
        ),
        (
            "Visual Top-K",
            "Final internal",
            _format_count(len(split_frames["Internal Benchmark 1000"])),
            _format_count(
                split_frames["Internal Benchmark 1000"][
                    "leakage_group_id"
                ].nunique()
            ),
            "21-class ranking",
        ),
        (
            "Visual Confusion Sets",
            "Development",
            _format_count(len(confusion_development)),
            _format_count(confusion_development["sample_id"].nunique()),
            "834 tasks = 417 images × two difficulty conditions",
        ),
        (
            "Visual Confusion Sets",
            "Final internal",
            _format_count(len(confusion_final)),
            _format_count(confusion_final["sample_id"].nunique()),
            "828 tasks = 414 images × two difficulty conditions",
        ),
        (
            "Evidence-Grounded Diagnosis",
            "Development",
            _format_count(len(evidence_development)),
            _format_count(evidence_development["leakage_group_id"].nunique()),
            (
                f'{int(evidence_development["score_morphology"].sum())} morphology; '
                f'{int(evidence_development["score_description"].sum())} description; '
                f'{int(evidence_development["score_diagnosis"].sum())} diagnosis'
            ),
        ),
        (
            "Evidence-Grounded Diagnosis",
            "External DDI",
            _format_count(len(evidence_external)),
            _format_count(evidence_external["leakage_group_id"].nunique()),
            (
                f'{int(evidence_external["score_morphology"].sum())} morphology; '
                f'{int(evidence_external["score_description"].sum())} description; '
                f'{int(evidence_external["score_diagnosis"].sum())} diagnosis'
            ),
        ),
    ]

    auxiliary_rows = [
        (
            "SkinCAP",
            _format_count(len(skincap)),
            "Fitzpatrick17k + DDI image-caption overlay",
            "No — all images come from upstream datasets",
            "Clinical descriptions in Evidence-Grounded; candidate SFT caption targets",
            "Gated; upstream terms also apply",
        ),
        (
            "SkinCaRe",
            _format_count(len(skincap) + skincot_image_count),
            (
                f'{_format_count(len(skincap))} SkinCAP + '
                f'{_format_count(skincot_image_count)} SkinCoT'
            ),
            "No — SkinCAP and DermNet/SkinCoT lineages are reused",
            "Not in the current manifests; possible audited rationale source",
            "Gated; reasoning and licensing require review",
        ),
        (
            "SKINCON",
            (
                f'{_format_count(len(skincon_fitzpatrick) + len(skincon_ddi))} '
                f'annotations ({_format_count(skincon_usable)} usable)'
            ),
            "48 binary morphology concepts over Fitzpatrick17k + DDI",
            "No — annotation layer only",
            "Morphology gold labels in Evidence-Grounded",
            "Upstream image licences apply",
        ),
        (
            "DermNet Kaggle mirror",
            _format_count(19_559),
            "Clinical images in 23 broad directory categories",
            "Potentially, but case grouping is unavailable",
            "Excluded from current Train and disease coverage",
            "Provenance and CC BY-NC-ND restrictions require resolution",
        ),
        (
            "DermaVQA",
            "1,488 encounters",
            "Multilingual image questions and free-text answers",
            "Not assumed — IIYI/Reddit lineage may overlap other sources",
            "Not in current manifests; candidate future interactive task source",
            "Privacy, answer quality, and treatment advice require audit",
        ),
        (
            "DermoBench",
            "Task release",
            "Derived multimodal evaluation suite spanning several source datasets",
            "No — derived benchmark, not an independent source",
            "Separate benchmark suite; not part of current Derma-ISEP releases",
            "Gated and upstream licences apply",
        ),
    ]

    sample_rows = []
    sample_columns = [
        "training_role",
        "dataset_id",
        "sample_id",
        "disease_id",
        "caption",
        "image_uri",
    ]
    for _, group in training.groupby("training_role", sort=True):
        for record in group[sample_columns].head(3).to_dict(orient="records"):
            caption = record["caption"]
            caption_text = "—" if pd.isna(caption) else str(caption)
            if len(caption_text) > 180:
                caption_text = caption_text[:177].rstrip() + "…"
            uri = str(record["image_uri"])
            if len(uri) > 105:
                uri = uri[:102] + "…"
            disease_id = record["disease_id"]
            disease_text = "—" if pd.isna(disease_id) else str(disease_id)
            sample_rows.append(
                (
                    escape(ROLE_LABELS[str(record["training_role"])]),
                    escape(SOURCE_LABELS.get(record["dataset_id"], record["dataset_id"])),
                    f'<span class="mono">{escape(str(record["sample_id"]))}</span>',
                    escape(disease_text),
                    escape(caption_text),
                    f'<span class="mono small">{escape(uri)}</span>',
                )
            )

    artifact_rows = [
        (
            "Augmented training pool",
            '<span class="mono">data/training/dermatology_multimodal_v1/train_images.parquet</span>',
        ),
        (
            "Teacher annotation queue",
            '<span class="mono">data/training/dermatology_multimodal_v1/teacher_annotation_queue.parquet</span>',
        ),
        (
            "HF-ready 21-class image release",
            '<span class="mono">data/training/ISEPDermData/</span>',
        ),
        (
            "Original internal Train",
            '<span class="mono">data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/train.parquet</span>',
        ),
        (
            "Benchmark Validation",
            '<span class="mono">data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/validation.parquet</span>',
        ),
        (
            "Sealed Internal Test",
            '<span class="mono">data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/internal_test.parquet</span>',
        ),
        (
            "Primary Internal Benchmark",
            '<span class="mono">data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/internal_benchmark_1000.parquet</span>',
        ),
        (
            "External DDI",
            '<span class="mono">data/benchmarks/derma_isep/visual_top_k_v1/datasets/external/external_ddi.parquet</span>',
        ),
        (
            "External SkinDisNet",
            '<span class="mono">data/benchmarks/derma_isep/visual_top_k_v1/datasets/external/external_skindisnet.parquet</span>',
        ),
        (
            "SkinCAP captions and labels",
            '<span class="mono">configs/datasets/skincap/data/skincap_v240623.csv</span>',
        ),
        (
            "SKINCON morphology annotations",
            '<span class="mono">configs/datasets/skincon/data/annotations/</span>',
        ),
        (
            "SkinCaRe / SkinCoT reasoning data",
            '<span class="mono">configs/datasets/skincare/data/</span>',
        ),
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dermatology dataset overview</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #607080;
      --line: #dbe4ea;
      --panel: #ffffff;
      --page: #f4f7f9;
      --blue: #176b87;
      --blue-soft: #dff1f5;
      --green: #237a57;
      --green-soft: #e2f3eb;
      --amber: #a85e00;
      --amber-soft: #fff0d7;
      --red: #a33a3a;
      --red-soft: #fbe5e5;
      --shadow: 0 10px 28px rgba(31, 55, 73, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--page);
      font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      color: white;
      background: linear-gradient(130deg, #12394a, #176b87 58%, #2e8871);
      padding: 52px max(24px, calc((100vw - 1240px) / 2));
    }}
    header h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 50px); line-height: 1.08; }}
    header p {{ max-width: 820px; margin: 0; color: #d9eef4; font-size: 17px; }}
    nav {{
      position: sticky; top: 0; z-index: 10; overflow-x: auto;
      display: flex; gap: 8px; padding: 10px max(20px, calc((100vw - 1240px) / 2));
      border-bottom: 1px solid var(--line); background: rgba(255,255,255,.94); backdrop-filter: blur(10px);
    }}
    nav a {{ white-space: nowrap; color: var(--blue); text-decoration: none; font-weight: 650; padding: 7px 10px; border-radius: 8px; }}
    nav a:hover {{ background: var(--blue-soft); }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 72px; }}
    section {{ scroll-margin-top: 74px; margin: 0 0 32px; }}
    h2 {{ font-size: 25px; margin: 0 0 8px; }}
    h3 {{ margin: 24px 0 8px; }}
    .lede {{ color: var(--muted); max-width: 920px; margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 18px 0; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow); }}
    .card {{ padding: 18px; }}
    .card .value {{ display: block; font-size: 30px; font-weight: 780; line-height: 1.1; color: var(--blue); }}
    .card .label {{ display: block; margin-top: 6px; color: var(--muted); }}
    .panel {{ padding: 20px; margin-top: 14px; }}
    .callout {{ border-left: 5px solid var(--blue); background: var(--blue-soft); border-radius: 10px; padding: 16px 18px; margin: 16px 0; }}
    .callout strong {{ color: #104e63; }}
    .flow {{ display: grid; grid-template-columns: repeat(7, auto); align-items: stretch; gap: 10px; margin: 18px 0; }}
    .flow-box {{ display: grid; place-content: center; min-height: 92px; padding: 12px; text-align: center; border: 1px solid var(--line); border-radius: 12px; background: white; box-shadow: var(--shadow); }}
    .flow-box strong {{ display: block; font-size: 20px; color: var(--blue); }}
    .flow-arrow {{ display: grid; place-content: center; color: var(--muted); font-size: 22px; }}
    .role-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:16px; }}
    .role {{ padding:16px; border-radius:12px; border:1px solid var(--line); }}
    .role.classify {{ background:var(--green-soft); }}
    .role.describe {{ background:var(--blue-soft); }}
    .role.abstain {{ background:var(--amber-soft); }}
    .role b {{ display:block; margin-bottom:4px; }}
    .table-wrap {{ width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 11px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #edf3f6; color: #334653; font-size: 12px; letter-spacing: .035em; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    tbody tr:hover {{ background: #f7fafb; }}
    td.numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .small {{ font-size: 11px; color: var(--muted); }}
    .bar-track {{ width: 150px; height: 9px; margin-top: 6px; overflow: hidden; border-radius: 99px; background: #e8eef1; }}
    .bar {{ display: block; height: 100%; border-radius: inherit; background: var(--blue); }}
    .filter {{ width: min(420px, 100%); margin: 8px 0 12px; padding: 10px 12px; border: 1px solid #bdcbd3; border-radius: 9px; font: inherit; }}
    code {{ padding: 2px 5px; border-radius: 5px; background: #e9eff2; }}
    footer {{ color: var(--muted); margin-top: 38px; padding-top: 18px; border-top: 1px solid var(--line); }}
    @media (max-width: 800px) {{
      .flow {{ grid-template-columns: 1fr; }}
      .flow-arrow {{ transform: rotate(90deg); min-height: 24px; }}
      .role-grid {{ grid-template-columns: 1fr; }}
      th {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Dermatology dataset overview</h1>
    <p>A read-only view of the training pool, 21-class subset, auxiliary annotation datasets, development splits, sealed tests, external datasets, and specialized benchmark manifests.</p>
  </header>
  <nav>
    <a href="#map">Map</a>
    <a href="#clarification">6,417 → 18,914</a>
    <a href="#training">Training pool</a>
    <a href="#auxiliary">Auxiliary</a>
    <a href="#classes">21 classes</a>
    <a href="#evaluation">Evaluation</a>
    <a href="#external">External</a>
    <a href="#benchmarks">Benchmarks</a>
    <a href="#samples">Rows</a>
    <a href="#files">Files</a>
  </nav>
  <main>
    <section id="map">
      <h2>Current data map</h2>
      <p class="lede">The 81,787-image manifest is a multimodal training pool. Only 18,914 images have a diagnosis mapped to one of the 21 active classes.</p>
      <div class="cards">
        <div class="card"><span class="value">{_format_count(len(training))}</span><span class="label">images in the multimodal training pool</span></div>
        <div class="card"><span class="value">{_format_count(training['leakage_group_id'].nunique())}</span><span class="label">training leakage groups</span></div>
        <div class="card"><span class="value">{_format_count(total_in_domain)}</span><span class="label">images mapped to the 21 classes</span></div>
        <div class="card"><span class="value">21</span><span class="label">active closed-set diseases</span></div>
        <div class="card"><span class="value">{_format_count(release['excluded_new_candidates'])}</span><span class="label">new candidates excluded by duplicate checks</span></div>
      </div>
      <div class="flow">
        <div class="flow-box"><span>Original internal Train</span><strong>{_format_count(original_in_domain)}</strong><small>21-class images</small></div>
        <div class="flow-arrow">+</div>
        <div class="flow-box"><span>Derm1M mapped</span><strong>{_format_count(derm1m_in_domain)}</strong><small>21-class images</small></div>
        <div class="flow-arrow">+</div>
        <div class="flow-box"><span>HIBA mapped</span><strong>{_format_count(hiba_in_domain)}</strong><small>21-class images</small></div>
        <div class="flow-arrow">=</div>
        <div class="flow-box"><span>Augmented diagnosis subset</span><strong>{_format_count(total_in_domain)}</strong><small>21 classes</small></div>
      </div>
    </section>

    <section id="clarification">
      <h2>Why 6,417 became 18,914</h2>
      <div class="callout"><strong>Both numbers were correct at different stages.</strong> The 6,417 images are the frozen original Train from Fitzpatrick17k-C, PAD-UFES-20, and SCIN. After that split was frozen, eligible Derm1M and HIBA images were added to the training layer only. Of those additions, 12,497 map safely to the same 21-class taxonomy.</div>
      <div class="role-grid">
        <div class="role classify"><b>18,914 diagnosis images</b>Eligible for 21-class diagnosis, differential, evidence, and rationale targets.</div>
        <div class="role describe"><b>17,518 description-only images</b>Eligible for morphology and description, but not for a forced diagnosis.</div>
        <div class="role abstain"><b>45,355 out-of-domain images</b>Potential abstention examples; they must not be treated as one of the 21 classes.</div>
      </div>
    </section>

    <section id="training">
      <h2>Multimodal training pool</h2>
      <p class="lede">A <code>training_role</code> defines what each record may safely teach. It is not a conventional single-label classification dataset.</p>
      {_table(["Training role", "Images", "Groups", "Sources", "Safe use"], role_rows, numeric_columns={1, 2, 3})}
      <h3>Breakdown by source and role</h3>
      <input class="filter" type="search" placeholder="Filter source or role…" oninput="filterTable('source-role-table', this.value)">
      {_table(["Source", "Training role", "Images", "Groups"], source_role_rows, table_id="source-role-table", numeric_columns={2, 3})}
    </section>

    <section id="auxiliary">
      <h2>Auxiliary and configured datasets</h2>
      <p class="lede">These datasets are present in <code>configs/datasets/</code> but must not be added to the image totals as independent cases. Some annotate images already counted elsewhere; others are inactive because their labels, grouping, provenance, or licences do not yet satisfy the current release policy.</p>
      {_table(["Dataset", "Local size", "What it contains", "Independent images?", "Current use", "Main caution"], auxiliary_rows)}
      <div class="callout"><strong>Active overlays:</strong> SkinCAP supplies clinical descriptions and SKINCON supplies morphology concepts for Evidence-Grounded Diagnosis. They enrich Fitzpatrick17k/DDI records; they do not add new independent patients. <strong>Currently inactive:</strong> SkinCaRe, DermNet, and DermaVQA are not part of the 81,787-image training manifest, while DermoBench remains a separate derived evaluation suite.</div>
    </section>

    <section id="classes">
      <h2>21-class diagnosis subset</h2>
      <p class="lede">These are the 18,914 images that may receive a disease classification target. Counts remain imbalanced, so SFT sampling should be class-aware.</p>
      <input class="filter" type="search" placeholder="Filter disease or ID…" oninput="filterTable('class-table', this.value)">
      {_table(["ID", "Disease", "Images", "Groups", "Fitzpatrick17k-C", "PAD-UFES-20", "SCIN", "Derm1M", "HIBA", "Relative support"], class_rows, table_id="class-table", numeric_columns={2, 3, 4, 5, 6, 7, 8})}
    </section>

    <section id="evaluation">
      <h2>Internal development and final evaluation</h2>
      <p class="lede">These manifests remain protected from gradient-based training. Validation may influence development; Internal Test and Internal Benchmark may not.</p>
      {_table(["Manifest", "Images", "Groups", "Classes", "Purpose"], split_rows, numeric_columns={1, 2, 3})}
    </section>

    <section id="external">
      <h2>External datasets</h2>
      <p class="lede">External datasets test distribution shift. They cover subsets of the 21-class taxonomy and must not select the teacher, prompt, checkpoint, parser, or thresholds.</p>
      {_table(["Dataset", "Images", "Groups", "Supported classes", "Purpose"], external_rows, numeric_columns={1, 2, 3})}
      <h3>Supported external class distribution</h3>
      {_table(["Dataset", "ID", "Disease", "Images", "Groups"], external_class_rows, numeric_columns={3, 4})}
    </section>

    <section id="benchmarks">
      <h2>Benchmark views</h2>
      <p class="lede">A benchmark is a task contract over a manifest. The same image may support ranking, paired confusion, or evidence-grounding without becoming a new training image.</p>
      {_table(["Benchmark", "Role", "Rows/tasks", "Unique cases/groups", "What is scored"], benchmark_rows, numeric_columns={2, 3})}
    </section>

    <section id="samples">
      <h2>Example manifest rows</h2>
      <p class="lede">These examples show metadata only. A <code>zip://archive::member</code> URI identifies an image stored inside a source archive.</p>
      <input class="filter" type="search" placeholder="Filter sample rows…" oninput="filterTable('sample-table', this.value)">
      {_table(["Training role", "Source", "Sample ID", "Disease ID", "Caption excerpt", "Image URI"], sample_rows, table_id="sample-table")}
    </section>

    <section id="files">
      <h2>Where the data live</h2>
      {_table(["Artifact", "Repository-relative path"], artifact_rows)}
    </section>

    <footer>
      Generated from materialized Parquet manifests and release metadata by <code>python -m src.data_pipeline.dataset_overview_report</code>. This report does not modify datasets or splits.
    </footer>
  </main>
  <script>
    function filterTable(tableId, query) {{
      const needle = query.trim().toLowerCase();
      document.querySelectorAll(`#${{tableId}} tbody tr`).forEach((row) => {{
        row.hidden = needle && !row.textContent.toLowerCase().includes(needle);
      }});
    }}
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML overview of the datasets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing configs/, data/, and src/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Repository-relative output HTML path.",
    )
    args = parser.parse_args()
    output = generate_dataset_overview(args.project_root, output_path=args.output)
    print(output)


if __name__ == "__main__":
    main()
