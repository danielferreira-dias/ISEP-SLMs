"""Standalone HTML reports for completed multimodal benchmark runs."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import yaml

from src.benchmark.results import RunPaths, read_jsonl


ImageLoader = Callable[[str], bytes]


def generate_run_report(
    run_directory: Path,
    *,
    image_loader: ImageLoader | None = None,
    output_path: Path | None = None,
) -> Path:
    """Generate one self-contained, read-only HTML benchmark report."""

    paths = RunPaths.from_directory(run_directory.resolve())
    if not paths.predictions.is_file():
        raise FileNotFoundError(
            f"Predictions file is missing: {paths.predictions}"
        )
    destination = (output_path or paths.report).resolve()
    records = read_jsonl(paths.predictions)
    prompts = _prompt_index(paths.rendered_prompts)
    manifest = _load_yaml(paths.manifest)
    snapshot = _load_yaml(paths.config_snapshot)
    metrics = _load_json(paths.metrics)
    disease_names = _disease_names(snapshot)
    thumbnails: dict[str, tuple[str | None, str | None]] = {}

    report_records = [
        _report_record(
            record,
            prompts=prompts,
            disease_names=disease_names,
            image_loader=image_loader,
            thumbnails=thumbnails,
        )
        for record in records
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(paths.directory),
        "manifest": manifest,
        "metrics": metrics,
        "metric_cards": _metric_cards(report_records, metrics),
        "metric_breakdowns": _metric_breakdowns(metrics),
        "records": report_records,
    }
    encoded_payload = json.dumps(
        _json_compatible(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded_payload = (
        encoded_payload.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    _atomic_write(destination, _html_document(encoded_payload))
    return destination


def _report_record(
    record: dict[str, Any],
    *,
    prompts: dict[str, dict[str, Any]],
    disease_names: dict[str, str],
    image_loader: ImageLoader | None,
    thumbnails: dict[str, tuple[str | None, str | None]],
) -> dict[str, Any]:
    response = _mapping(record.get("response"))
    reasoning = _mapping(response.get("reasoning"))
    usage = _mapping(response.get("usage"))
    metadata = _mapping(record.get("metadata"))
    task_id = _text(record.get("task_id"))
    image_uri = _text(record.get("image_uri"))
    thumbnail, thumbnail_error = thumbnails.get(
        image_uri,
        (None, None),
    )
    if image_uri not in thumbnails and image_loader is not None:
        try:
            thumbnail = _thumbnail_data_url(image_loader(image_uri))
        except Exception as exc:
            thumbnail_error = f"{type(exc).__name__}: {exc}"
        thumbnails[image_uri] = (thumbnail, thumbnail_error)

    ground_truth_id = _text(record.get("ground_truth_disease_id"))
    reasoning_text = _text(reasoning.get("text"))
    reasoning_source = _text(reasoning.get("source"))
    reasoning_capture_mode = _text(reasoning.get("capture_mode"))
    reasoning_availability = _text(reasoning.get("availability"))
    reasoning_note = ""
    if (
        reasoning_capture_mode == "summary"
        and reasoning_source == "reasoning.summary"
        and reasoning_text in {"auto", "concise", "detailed"}
    ):
        reasoning_text = ""
        reasoning_availability = (
            "tokens_only"
            if reasoning.get("token_count") is not None
            else "none"
        )
        reasoning_note = (
            "Legacy run: the provider's requested summary mode was captured "
            "instead of generated summary text. The missing summary cannot "
            "be recovered from this run; the corrected parser records it as "
            "unavailable rather than storing this placeholder."
        )
    elif reasoning_capture_mode == "summary" and not reasoning_text:
        reasoning_note = (
            "A reasoning summary was requested, but the provider returned "
            "only reasoning token usage and no textual summary for this "
            "multimodal request."
        )
    status = _report_status(
        _text(record.get("status")),
        response=response,
    )
    return {
        "task_id": task_id,
        "sample_id": _text(record.get("sample_id")),
        "model_id": _text(record.get("model_id")),
        "status": status,
        "image_uri": image_uri,
        "thumbnail": thumbnail,
        "thumbnail_error": thumbnail_error,
        "ground_truth_id": ground_truth_id,
        "ground_truth_name": disease_names.get(
            ground_truth_id,
            ground_truth_id,
        ),
        "dataset_id": _text(metadata.get("dataset_id")),
        "skin_tone_system": _text(metadata.get("skin_tone_system")),
        "skin_tone": _text(metadata.get("skin_tone")),
        "diagnosis_basis": _text(metadata.get("diagnosis_basis")),
        "final_text": _text(response.get("final_text")),
        "parsed_output": response.get("parsed_output"),
        "canonical_output": response.get("canonical_output"),
        "json_valid": bool(response.get("json_valid")),
        "recoverable_json_valid": bool(
            response.get(
                "recoverable_json_valid",
                response.get("json_valid"),
            )
        ),
        "schema_valid": _report_schema_valid(response),
        "canonical_schema_valid": bool(
            response.get(
                "canonical_schema_valid",
                response.get("schema_valid"),
            )
        ),
        "canonicalization_rules": _string_list(
            response.get("canonicalization_rules")
        ),
        "validation_errors": _string_list(
            response.get("validation_errors")
        ),
        "reasoning": {
            "availability": reasoning_availability,
            "capture_mode": reasoning_capture_mode,
            "source": reasoning_source,
            "token_count": reasoning.get("token_count"),
            "text": reasoning_text,
            "note": reasoning_note,
        },
        "usage": usage,
        "finish_reason": _text(response.get("finish_reason")),
        "response_metadata": response.get("metadata"),
        "provider_metadata": response.get("provider_metadata"),
        "prompt": prompts.get(task_id, {}),
    }


def _thumbnail_data_url(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((320, 240), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=72,
        optimize=True,
        subsampling=2,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prompt_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        _text(record.get("task_id")): record
        for record in read_jsonl(path)
        if record.get("task_id") is not None
    }


def _disease_names(snapshot: dict[str, Any]) -> dict[str, str]:
    taxonomy = _mapping(snapshot.get("disease_taxonomy"))
    diseases = taxonomy.get("diseases")
    if not isinstance(diseases, list):
        return {}
    return {
        _text(item.get("id")): _text(
            item.get("display_name") or item.get("canonical_name")
        )
        for item in diseases
        if isinstance(item, dict) and item.get("id") is not None
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value]


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _metric_cards(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    status_cards = {
        "total": len(records),
        "ok": sum(record.get("status") == "ok" for record in records),
        "format_invalid": sum(
            record.get("status") == "format_invalid" for record in records
        ),
        "schema_invalid": sum(
            record.get("status") == "schema_invalid" for record in records
        ),
        "semantic_noncompliant": sum(
            record.get("status") == "semantic_noncompliant"
            for record in records
        ),
        "truncated": sum(
            record.get("status") == "truncated_output"
            for record in records
        ),
        "safety_refusals": sum(
            record.get("status") == "safety_refusal"
            for record in records
        ),
    }
    scalar_metrics = {
        str(name): value
        for name, value in metrics.items()
        if not isinstance(value, (dict, list, tuple))
    }
    return [
        {
            "name": name,
            "label": name.replace("_", " "),
            "value": _format_metric_value(value),
        }
        for name, value in {**status_cards, **scalar_metrics}.items()
    ]


def _report_status(
    status: str,
    *,
    response: dict[str, Any],
) -> str:
    """Map a legacy aggregate invalid status to its failed contract layer."""

    if status != "invalid_output":
        return status
    if not bool(response.get("json_valid")):
        return "format_invalid"
    if not _report_schema_valid(response):
        return "schema_invalid"
    metadata = _mapping(response.get("metadata"))
    if metadata.get("semantic_valid", True) is False:
        return "semantic_noncompliant"
    return status


def _report_schema_valid(response: dict[str, Any]) -> bool:
    """Expose schema validity independently for current and legacy runs."""

    metadata = _mapping(response.get("metadata"))
    schema_errors = metadata.get("schema_errors")
    if isinstance(schema_errors, (list, tuple)):
        return (
            isinstance(response.get("parsed_output"), dict)
            and len(schema_errors) == 0
        )
    return bool(response.get("schema_valid"))


def _metric_breakdowns(
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for name, value in metrics.items():
        if not isinstance(value, dict):
            continue
        nested = bool(value) and all(
            isinstance(item_value, dict)
            for item_value in value.values()
        )
        if nested:
            columns = list(
                dict.fromkeys(
                    str(column)
                    for item_value in value.values()
                    for column in item_value
                )
            )
            items = [
                {
                    "label": _metric_item_label(item_name),
                    "values": [
                        _format_metric_value(item_value.get(column))
                        for column in columns
                    ],
                }
                for item_name, item_value in value.items()
            ]
        else:
            columns = ["value"]
            items = [
                {
                    "label": _metric_item_label(item_name),
                    "values": [_format_metric_value(item_value)],
                }
                for item_name, item_value in value.items()
            ]
        sections.append(
            {
                "name": str(name),
                "label": str(name).replace("_", " "),
                "columns": [
                    column.replace("_", " ")
                    for column in columns
                ],
                "items": items,
            }
        )
    return sections


def _metric_item_label(value: Any) -> str:
    text = str(value)
    return text if ":" in text else text.replace("_", " ")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.1f}%"
        return f"{value:.4g}"
    if value is None:
        return "—"
    return str(value)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _html_document(payload: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark run report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #66727e;
      --line: #dce2e7;
      --accent: #136f63;
      --accent-soft: #e5f4f1;
      --danger: #9f2d2d;
      --warning: #8b5a00;
      --ok: #216e39;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); }}
    header {{
      padding: 28px clamp(18px, 5vw, 64px);
      color: white;
      background: linear-gradient(135deg, #123d4a, #136f63);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(24px, 4vw, 38px); }}
    h2, h3 {{ margin: 0; }}
    .subtle {{ color: var(--muted); }}
    header .subtle {{ color: #d6ebe7; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric-breakdowns {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric-breakdown {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(28, 39, 49, .05);
    }}
    .metric-breakdown h2 {{
      padding: 12px 14px;
      background: #f7f9fa;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
    }}
    .metric-breakdown table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .metric-breakdown th, .metric-breakdown td {{
      padding: 8px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .metric-breakdown tr:last-child th,
    .metric-breakdown tr:last-child td {{ border-bottom: 0; }}
    .metric-breakdown td {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .metric, .toolbar, .case {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(28, 39, 49, .05);
    }}
    .metric {{ padding: 14px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(4, minmax(130px, 1fr));
      gap: 10px;
      padding: 14px;
      margin-bottom: 18px;
    }}
    input, select, button {{
      width: 100%;
      min-height: 40px;
      border: 1px solid #b9c3cb;
      border-radius: 8px;
      background: white;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
    }}
    button {{ cursor: pointer; font-weight: 650; }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .result-line {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 12px 2px;
    }}
    .cases {{ display: grid; gap: 16px; }}
    .case {{ overflow: hidden; }}
    .case-head {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 14px;
      align-items: center;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .thumb {{
      width: 112px;
      height: 84px;
      object-fit: contain;
      background: #edf0f2;
      border-radius: 8px;
    }}
    .case-title {{ display: grid; gap: 5px; min-width: 0; }}
    .case-title code {{ overflow-wrap: anywhere; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 5px; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #edf0f2;
      color: #46525c;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.ok {{ background: #e6f4ea; color: var(--ok); }}
    .badge.invalid_output, .badge.format_invalid,
    .badge.schema_invalid, .badge.semantic_noncompliant,
    .badge.truncated_output {{
      background: #fff2d6; color: var(--warning);
    }}
    .badge.backend_error, .badge.image_error, .badge.safety_refusal {{
      background: #fde8e8; color: var(--danger);
    }}
    .case-body {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      padding: 16px;
    }}
    .pane {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 9px;
      overflow: hidden;
    }}
    .pane h3 {{
      padding: 10px 12px;
      background: #f7f9fa;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    pre {{
      margin: 0;
      min-height: 90px;
      max-height: 420px;
      overflow: auto;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    details {{ border-top: 1px solid var(--line); }}
    summary {{
      cursor: pointer;
      padding: 11px 16px;
      font-weight: 700;
      color: #33414d;
    }}
    .details-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      padding: 0 16px 16px;
    }}
    .errors {{ color: var(--danger); }}
    .pagination {{
      display: grid;
      grid-template-columns: 110px 1fr 110px;
      align-items: center;
      gap: 10px;
      margin: 18px 0 30px;
      text-align: center;
    }}
    .empty {{
      padding: 40px;
      text-align: center;
      background: white;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    @media (max-width: 800px) {{
      main {{ padding: 12px; }}
      .toolbar {{ position: static; grid-template-columns: 1fr 1fr; }}
      .case-head {{ grid-template-columns: 1fr; }}
      .thumb {{ width: 100%; height: 180px; }}
      .case-body {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1 id="title">Benchmark run report</h1>
    <div class="subtle" id="run-meta"></div>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <section class="metric-breakdowns" id="metric-breakdowns"></section>
    <section class="toolbar" aria-label="Report filters">
      <input id="search" type="search"
        placeholder="Search answer, reasoning, sample or task ID">
      <select id="status-filter" aria-label="Filter by status"></select>
      <select id="disease-filter" aria-label="Filter by disease"></select>
      <select id="dataset-filter" aria-label="Filter by dataset"></select>
      <select id="skin-tone-filter" aria-label="Filter by skin tone"></select>
    </section>
    <div class="result-line">
      <strong id="result-count"></strong>
      <label>Cases per page
        <select id="page-size">
          <option>10</option><option selected>25</option><option>50</option>
        </select>
      </label>
    </div>
    <section class="cases" id="cases"></section>
    <nav class="pagination" aria-label="Pagination">
      <button id="previous">Previous</button>
      <span id="page-label"></span>
      <button id="next">Next</button>
    </nav>
  </main>
  <script>
    const DATA = {payload};
    const records = DATA.records || [];
    const state = {{ page: 1, pageSize: 25 }};
    const esc = value => String(value ?? "").replace(
      /[&<>"']/g,
      char => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}})[char]
    );
    const pretty = value => value == null ? "" :
      typeof value === "string" ? value : JSON.stringify(value, null, 2);
    const option = (value, label) =>
      `<option value="${{esc(value)}}">${{esc(label)}}</option>`;
    const unique = key => [...new Set(records.map(row => row[key]).filter(Boolean))]
      .sort((a, b) => String(a).localeCompare(String(b)));
    const manifest = DATA.manifest || {{}};
    const model = manifest.model || {{}};
    const benchmark = manifest.benchmark || {{}};
    document.title = `${{model.id || "Model"}} · ${{benchmark.id || "Benchmark"}}`;
    document.getElementById("title").textContent =
      `${{model.display_name || model.id || "Model"}} · ${{benchmark.id || "Benchmark"}}`;
    document.getElementById("run-meta").textContent =
      `${{manifest.status || "unknown"}} · ${{records.length}} cases · ${{DATA.run_directory}}`;

    const metricRoot = document.getElementById("metrics");
    (DATA.metric_cards || []).forEach(metric => {{
      metricRoot.insertAdjacentHTML(
        "beforeend",
        `<div class="metric"><span class="subtle">${{esc(metric.label)}}</span>` +
        `<strong>${{esc(metric.value)}}</strong></div>`
      );
    }});
    const breakdownRoot = document.getElementById("metric-breakdowns");
    (DATA.metric_breakdowns || []).forEach(section => {{
      const columns = section.columns || ["value"];
      const head = `<thead><tr><th>group</th>${{
        columns.map(column => `<th>${{esc(column)}}</th>`).join("")
      }}</tr></thead>`;
      const rows = (section.items || []).map(item => {{
        const values = item.values || [item.value];
        return `<tr><th>${{esc(item.label)}}</th>${{
          values.map(value => `<td>${{esc(value)}}</td>`).join("")
        }}</tr>`;
      }}).join("");
      breakdownRoot.insertAdjacentHTML(
        "beforeend",
        `<section class="metric-breakdown"><h2>${{esc(section.label)}}</h2>` +
        `<table>${{head}}<tbody>${{rows}}</tbody></table></section>`
      );
    }});

    const statusFilter = document.getElementById("status-filter");
    statusFilter.innerHTML = option("", "All statuses") +
      unique("status").map(value => option(value, value)).join("");
    const diseaseFilter = document.getElementById("disease-filter");
    diseaseFilter.innerHTML = option("", "All diseases") +
      [...new Map(records.map(row => [
        row.ground_truth_id,
        `${{row.ground_truth_id}} · ${{row.ground_truth_name}}`
      ])).entries()].filter(([id]) => id).sort()
        .map(([id, label]) => option(id, label)).join("");
    const datasetFilter = document.getElementById("dataset-filter");
    datasetFilter.innerHTML = option("", "All datasets") +
      unique("dataset_id").map(value => option(value, value)).join("");
    const skinToneFilter = document.getElementById("skin-tone-filter");
    skinToneFilter.innerHTML = option("", "All skin tones") +
      unique("skin_tone").map(value => option(value, value)).join("");

    function filteredRecords() {{
      const query = document.getElementById("search").value.trim().toLowerCase();
      const status = statusFilter.value;
      const disease = diseaseFilter.value;
      const dataset = datasetFilter.value;
      const skinTone = skinToneFilter.value;
      return records.filter(row => {{
        if (status && row.status !== status) return false;
        if (disease && row.ground_truth_id !== disease) return false;
        if (dataset && row.dataset_id !== dataset) return false;
        if (skinTone && row.skin_tone !== skinTone) return false;
        if (!query) return true;
        return [
          row.task_id, row.sample_id, row.final_text,
          row.reasoning?.text, row.ground_truth_name, row.dataset_id,
          row.skin_tone_system, row.skin_tone
        ].join(" ").toLowerCase().includes(query);
      }});
    }}

    function pane(title, content, className = "") {{
      return `<section class="pane ${{className}}"><h3>${{esc(title)}}</h3>` +
        `<pre>${{esc(content || "Not available")}}</pre></section>`;
    }}

    function renderCase(row) {{
      const reasoning = row.reasoning || {{}};
      const usage = row.usage || {{}};
      const prompt = row.prompt || {{}};
      const reasoningTitle = (
        reasoning.capture_mode === "summary" &&
        reasoning.availability !== "summary"
      )
        ? "Reasoning summary unavailable"
        : reasoning.availability === "summary"
        ? "Reasoning summary"
        : reasoning.availability === "full"
          ? "Full reasoning"
          : "Reasoning";
      const reasoningDetails = [
        reasoning.source ? `source: ${{reasoning.source}}` : "",
        reasoning.token_count != null
          ? `${{reasoning.token_count}} reasoning tokens`
          : ""
      ].filter(Boolean).join(" · ");
      const thumb = row.thumbnail
        ? `<img class="thumb" src="${{row.thumbnail}}" alt="Benchmark image">`
        : `<div class="thumb"></div>`;
      const errors = (row.validation_errors || []).join("\\n");
      return `<article class="case">
        <div class="case-head">
          ${{thumb}}
          <div class="case-title">
            <h2>${{esc(row.ground_truth_name || row.ground_truth_id)}}</h2>
            <code>${{esc(row.sample_id)}}</code>
            <div class="badges">
              <span class="badge ${{esc(row.status)}}">${{esc(row.status)}}</span>
              <span class="badge">${{esc(row.ground_truth_id)}}</span>
              ${{row.dataset_id ? `<span class="badge">${{esc(row.dataset_id)}}</span>` : ""}}
              ${{row.skin_tone ? `<span class="badge">${{esc(row.skin_tone)}}</span>` : ""}}
            </div>
          </div>
          <div><strong>${{esc(usage.total_tokens ?? "—")}}</strong><br>
            <span class="subtle">total tokens</span></div>
        </div>
        <div class="case-body">
          ${{pane("Final answer", row.final_text)}}
          ${{pane(
            reasoningDetails
              ? `${{reasoningTitle}} · ${{reasoningDetails}}`
              : reasoningTitle,
            reasoning.text || reasoning.note || "No reasoning text returned"
          )}}
        </div>
        <details>
          <summary>Parsed output, validation and token usage</summary>
          <div class="details-grid">
            ${{pane("Parsed output", pretty(row.parsed_output))}}
            ${{pane("Canonical output", pretty(row.canonical_output))}}
            ${{pane(
              "Canonicalization rules",
              row.canonicalization_rules.join("\\n") || "None"
            )}}
            ${{pane("Validation errors", errors || "None", errors ? "errors" : "")}}
            ${{pane("Usage", pretty(usage))}}
            ${{pane("Reasoning metadata", pretty({{
              availability: reasoning.availability,
              capture_mode: reasoning.capture_mode,
              source: reasoning.source,
              token_count: reasoning.token_count,
              note: reasoning.note
            }}))}}
          </div>
        </details>
        <details>
          <summary>Prompt and provider metadata</summary>
          <div class="details-grid">
            ${{pane("System prompt", prompt.system_prompt)}}
            ${{pane("User prompt", prompt.user_prompt)}}
            ${{pane("Response metadata", pretty(row.response_metadata))}}
            ${{pane("Provider metadata", pretty(row.provider_metadata))}}
          </div>
        </details>
      </article>`;
    }}

    function render() {{
      const filtered = filteredRecords();
      const pages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
      state.page = Math.min(state.page, pages);
      const start = (state.page - 1) * state.pageSize;
      const pageRows = filtered.slice(start, start + state.pageSize);
      document.getElementById("result-count").textContent =
        `${{filtered.length}} of ${{records.length}} cases`;
      document.getElementById("page-label").textContent =
        `Page ${{state.page}} of ${{pages}}`;
      document.getElementById("previous").disabled = state.page <= 1;
      document.getElementById("next").disabled = state.page >= pages;
      document.getElementById("cases").innerHTML = pageRows.length
        ? pageRows.map(renderCase).join("")
        : `<div class="empty">No cases match the selected filters.</div>`;
    }}

    [
      "search", "status-filter", "disease-filter", "dataset-filter",
      "skin-tone-filter"
    ].forEach(id => {{
      document.getElementById(id).addEventListener("input", () => {{
        state.page = 1;
        render();
      }});
    }});
    document.getElementById("page-size").addEventListener("change", event => {{
      state.pageSize = Number(event.target.value);
      state.page = 1;
      render();
    }});
    document.getElementById("previous").addEventListener("click", () => {{
      state.page -= 1;
      render();
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }});
    document.getElementById("next").addEventListener("click", () => {{
      state.page += 1;
      render();
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }});
    render();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML report for a benchmark run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not resolve and embed image thumbnails.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_loader: ImageLoader | None = None
    resolver = None
    if not args.no_images:
        from src.data_pipeline.deduplication import ImageResolver

        resolver = ImageResolver(args.project_root.resolve())
        image_loader = resolver.read_bytes
    try:
        result = generate_run_report(
            args.run_directory,
            image_loader=image_loader,
            output_path=args.output,
        )
    finally:
        if resolver is not None:
            resolver.close()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
