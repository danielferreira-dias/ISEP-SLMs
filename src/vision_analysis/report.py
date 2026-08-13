"""Self-contained HTML report for qualitative attribution cases."""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image


def write_report(
    path: Path,
    *,
    analysis_id: str,
    model_name: str,
    cases: list[dict[str, Any]],
) -> None:
    """Write a compact, self-contained comparison report."""

    cards = "\n".join(_case_card(case) for case in cases)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(analysis_id)}</title>
<style>
body {{ margin: 0; background: #0b1020; color: #e7edf7; font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width: 1320px; margin: auto; padding: 32px 20px 56px; }}
h1 {{ margin-bottom: 4px; }} .subtitle {{ color: #9fb0c8; margin-top: 0; }}
.notice {{ background:#172038; border-left:4px solid #f1b44c; padding:14px 16px; border-radius:8px; margin:22px 0; }}
.case {{ background:#111a2e; border:1px solid #273451; border-radius:14px; padding:18px; margin:22px 0; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px; color:#b9c7dc; margin-bottom:14px; }}
.pill {{ background:#1d2943; border-radius:999px; padding:4px 9px; }}
.panels {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
.panel {{ background:#0d1527; border-radius:10px; padding:10px; }}
.panel img {{ width:100%; height:360px; object-fit:contain; background:#060a13; border-radius:7px; }}
.panel h3 {{ margin:8px 0 4px; font-size:15px; }}
.score {{ color:#9fb0c8; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
.legend {{ display:flex; gap:18px; color:#b9c7dc; font-size:13px; }} .red::before,.blue::before {{ content:""; display:inline-block; width:12px; height:12px; margin-right:5px; border-radius:2px; }}
.red::before {{ background:#ef4444; }} .blue::before {{ background:#3b82f6; }}
@media(max-width:850px) {{ .panels {{ grid-template-columns:1fr; }} .panel img {{ height:auto; }} }}
</style>
</head>
<body><main>
<h1>Student visual attribution pilot</h1>
<p class="subtitle">{_escape(model_name)} · {_escape(analysis_id)}</p>
<div class="notice"><strong>Interpretation:</strong> red regions support the selected diagnosis because blurring them lowers its score; blue regions suppress it because blurring them raises its score. These maps show sensitivity to perturbations, not private reasoning and not verified lesion segmentation.</div>
<div class="legend"><span class="red">supports target</span><span class="blue">suppresses target</span></div>
{cards}
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _case_card(case: dict[str, Any]) -> str:
    original = _data_url(case["original_image"])
    gold = _data_url(case["gold_overlay"])
    predicted = _data_url(case["predicted_overlay"])
    return f"""<section class="case">
<h2>{_escape(case['task_id'])}</h2>
<div class="meta"><span class="pill">{_escape(case['cohort'])}</span><span class="pill">source: {_escape(case['source'])}</span><span class="pill">skin tone: {_escape(case.get('skin_tone') or 'unavailable')}</span><span class="pill">age: {_escape(case.get('age_group') or 'unavailable')}</span></div>
<div class="panels">
<div class="panel"><img src="{original}" alt="original dermatology image"><h3>Original image</h3><div class="score">unaltered benchmark bytes</div></div>
<div class="panel"><img src="{gold}" alt="gold-label attribution"><h3>Gold target: {_escape(case['gold_label'])}</h3><div class="score">baseline mean log p = {case['gold_score']:.4f}</div></div>
<div class="panel"><img src="{predicted}" alt="predicted-label attribution"><h3>Benchmark-predicted target: {_escape(case['predicted_label'])}</h3><div class="score">baseline mean log p = {case['predicted_score']:.4f}</div></div>
</div></section>"""


def _data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + b64encode(buffer.getvalue()).decode("ascii")


def _escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
