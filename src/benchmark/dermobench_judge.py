"""Text-only DermoBench judges through OpenRouter's asynchronous Batch API."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.benchmark.dermobench import resolve_dermobench_spec
from src.benchmark.json_parsing import parse_json_output
from src.benchmark.results import read_jsonl


OPENROUTER_BATCH_URL = "https://openrouter.ai/api/beta/batches"
DEFAULT_BATCH_MODEL = "google/gemini-3.5-flash-lite:batch"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


SYSTEM_11 = (
    "You are a strict, no-nonsense clinical dermatology evaluator. "
    "You DO NOT see the image; evaluate ONLY by comparing the REFERENCE vs "
    "the CANDIDATE text. Use dermatology morphology standards. Avoid "
    "rewarding verbosity; penalize contradictions and invented findings. "
    "Focus on anatomical site, number/arrangement, primary lesion types, "
    "color, shape, borders, surface features, size/extent, distribution/"
    "pattern, and special/contextual features. Return STRICT JSON only."
)

USER_11 = """[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Evaluate as follows:
1) Decompose REFERENCE into at most 25 atomic CLAIMS.
2) For each CLAIM, label it with respect to CANDIDATE as Supported,
   PartiallySupported, Contradicted, Missing, or Vague.
3) Identify EXTRA INCORRECT statements in CANDIDATE.
4) Compute:
   recall_like = (Supported + 0.5*PartiallySupported) / max(1, total_ref_claims)
   precision_penalty = min(1.0, (Contradicted + ExtraIncorrect) / max(1, total_ref_claims))
   overall = round(100 * max(0, recall_like - 0.5*precision_penalty), 1)
Return JSON only:
{{
  "claims": [{{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}}],
  "counts": {{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},
  "rubric": {{"accuracy":0.0,"completeness":0.0,"consistency":0.0}},
  "overall": 0.0,
  "short_feedback": "at most 40 words"
}}"""

SYSTEM_12 = (
    "You are a strict dermatology evaluator for Task 1.2 (morph content + narrative). "
    "You DO NOT see the image. Focus on CONTENT, not formatting. "
    "Both REFERENCE and CANDIDATE may or may not wrap the morph JSON in <morph> tags. "
    "Do NOT penalize missing tags, extra whitespace, or minor ordering/format differences. "
    "If a JSON block is present anywhere, treat the FIRST JSON object as the morph content. "
    "If no JSON is present, infer the morph feature set from the surrounding text. "
    "Schemas you may encounter:\n"
    "  • SkinCon: {\"morphological_features_skincon\": [<feature strings>]} \n"
    "  • Derm7pt: {\"morphological_features_Derm7pt\": {pigment_network, blue_whitish_veil, vascular_structures, pigmentation, streaks, dots_and_globules, regression_structures}} \n"
    "For the narrative comparison, use dermatology morphology standards (site, number/arrangement, primary lesion types, color, shape, borders, surface features, size/extent, distribution/pattern, special/context). "
    "Also check CROSS-CONSISTENCY between the CANDIDATE morph content and CANDIDATE narrative. "
    "Return STRICT JSON only."
)

USER_12 = """You will be given REFERENCE and CANDIDATE texts.
Each may contain a morph JSON (SkinCon or Derm7pt) with or without <morph> tags,
possibly followed by a narrative paragraph. Do NOT penalize formatting.
Rules:
- If a JSON object appears anywhere, treat the FIRST JSON object as the morph content.
- If no JSON is found, infer the morph feature set from the surrounding text (best-effort).
- Use synonyms tolerance for semantic matching.

[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Your tasks:
1) MORPH SEMANTICS (content-first): Compare CANDIDATE-morph vs REFERENCE-morph semantically (synonyms allowed).
   Count supported/missing/contradicted/extra and give a semantic score in [0,1].
   If CANDIDATE has no explicit JSON, infer its morph set from the candidate text.

2) TEXT (NARRATIVE): Compare REFERENCE-narrative vs CANDIDATE-narrative using morphology standards.
   Extract <=25 atomic claims from the REFERENCE-narrative; for each, label CANDIDATE as Supported/PartiallySupported/Contradicted/Missing/Vague.
   Provide rubric sub-scores (accuracy, completeness, consistency) in [0,1] and overall [0,100] using:
   recall_like = (Supported + 0.5*PartiallySupported) / max(1, total_ref_claims)
   precision_penalty = min(1.0, (Contradicted + ExtraIncorrect) / max(1, total_ref_claims))
   overall = round(100 * max(0, recall_like - 0.5*precision_penalty), 1)

3) CROSS-CONSISTENCY: Judge if the CANDIDATE narrative contradicts the CANDIDATE morph content.
   Output a penalty in [0,1] (0=no issue, 1=severe) and short notes.

Output STRICT JSON:
{{
  "morph_semantic": {{
    "schema": "SkinCon" | "Derm7pt" | "Unknown",
    "supported": 0, "missing": 0, "contradicted": 0, "extra": 0,
    "score_semantic": 0.0,
    "notes": "≤60 words"
  }},
  "text_judge": {{
    "claims": [{{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}}],
    "counts": {{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},
    "rubric": {{"accuracy":0.0,"completeness":0.0,"consistency":0.0}},
    "overall": 0.0,
    "short_feedback": "≤40 words"
  }},
  "cross_consistency": {{"penalty": 0.0, "notes": "≤40 words"}}
}}"""

SYSTEM_31 = (
    "You are a strict dermatology evaluator for Task 3 (reasoning + final diagnosis). "
    "You DO NOT see the image; evaluate ONLY the textual content. Ignore formatting and tags. "
    "Goal: robustly extract (A) the candidate's reasoning and (B) the candidate's final diagnosis, "
    "then score (1) REASONING ALIGNMENT vs the GT reasoning and (2) DIAGNOSIS SIMILARITY vs the GT final diagnosis. "
    "Penalize contradictions and hallucinated findings. Do not reward verbosity. Return STRICT JSON only."
)

USER_31 = """[Task Prompt]
{task_prompt}

[GROUND_TRUTH_RAW]
{reference}

[CANDIDATE_RAW]
{candidate}

A) Extraction (be robust even if the candidate is unstructured):
   - From GROUND_TRUTH_RAW, extract reasoning from <reasoning> when present and final diagnosis from <final_diagnosis> when present; otherwise use best effort.
   - From CANDIDATE_RAW, extract the explanation and one most likely final diagnosis, preferring explicit Final/Diagnosis wording or the conclusive statement.

B) Reasoning Alignment:
   - Decompose gt_reasoning into ≤25 atomic claims (morphology, distribution, logic, differential cues).
   - For each claim, label cand_reasoning as Supported | PartiallySupported | Contradicted | Missing | Vague.
   - Compute:
   recall_like = (Supported + 0.5*PartiallySupported) / max(1, total_ref_claims)
   precision_penalty = min(1.0, (Contradicted + ExtraIncorrect) / max(1, total_ref_claims))
   reasoning_score = round(100 * max(0, recall_like - 0.5*precision_penalty), 1)

C) Diagnosis Similarity (graded, not binary):
   - Normalize synonyms and consider hierarchy and clinical proximity.
   - Relation: Exact | Synonym | Parent | Child | Sibling/CloseDifferential | SameSuperfamily | UnrelatedPlausible | WrongSystem | Nonsense/NoAnswer.
   - Similarity: Exact/Synonym=1.0; Parent/Child=0.85; Sibling/CloseDifferential=0.7; SameSuperfamily=0.5; UnrelatedPlausible=0.3; WrongSystem=0.1; Nonsense/NoAnswer=0.0.
   - diagnosis_score [0-100] = round(100 * similarity, 1).

D) Overall = round(0.5 * reasoning_score + 0.5 * diagnosis_score, 1).

STRICT JSON ONLY:
{{
  "extraction": {{"gt":{{"reasoning":"...","final_dx":"..."}},"cand":{{"reasoning":"...","final_dx":"..."}}}},
  "reasoning": {{"claims":[{{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}}],"counts":{{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},"score":0.0,"notes":"≤60 words"}},
  "diagnosis": {{"gt_dx":"...","cand_dx":"...","relation":"Exact|Synonym|Parent|Child|Sibling/CloseDifferential|SameSuperfamily|UnrelatedPlausible|WrongSystem|Nonsense/NoAnswer","similarity":0.0,"score":0.0,"notes":"≤40 words"}},
  "rubric": {{"reasoning_alignment":0.0,"diagnosis_similarity":0.0,"internal_consistency":0.0}},
  "overall": 0.0,
  "short_feedback": "≤50 words"
}}"""

SYSTEM_32 = (
    "You are a strict dermatology evaluator for Task 3.2 (reasoning + morph JSON + final diagnosis). "
    "You DO NOT see the image. Focus on CONTENT, not formatting. "
    "Both REFERENCE and CANDIDATE may or may not wrap the morph JSON in <morph> tags. "
    "Do NOT penalize missing tags, extra whitespace, or ordering differences. "
    "If a JSON object appears anywhere, treat the FIRST JSON object as the morph content. "
    "If no JSON is present, infer the morph feature set from the surrounding text. "
    "SCHEMA SELECTION RULE: Detect the schema used by REFERENCE. Compare and output using the SAME schema."
)

USER_32 = """You will be given REFERENCE and CANDIDATE texts containing three conceptual parts: <reasoning>, <morph> JSON, and <final_diagnosis>.
Be format-agnostic; extract content even when tags are missing or order differs.

Allowed schemas:
- Derm7pt (object with EXACT keys):
  pigment_network: absent|typical|atypical
  blue_whitish_veil: absent|present
  vascular_structures: absent|arborizing|comma|hairpin|within regression|wreath|dotted|linear irregular
  pigmentation: absent|diffuse regular|localized regular|diffuse irregular|localized irregular
  streaks: absent|regular|irregular
  dots_and_globules: absent|regular|irregular
  regression_structures: absent|blue areas|white areas|combinations

- SkinCon (array of strings only):
  {{"morphological_features_skincon": [ ... ]}} where each item is from this CLOSED set (case-sensitive):
  Abscess, Acuminate, Atrophy, Black, Blue, Brown(Hyperpigmentation), Bulla, Burrow, Comedo, Crust, Cyst, Dome-shaped, Erosion, Erythema, Excoriation, Exophytic/Fungating, Exudate, Fissure, Flat topped, Friable, Gray, Induration, Lichenification, Macule, Nodule, Papule, Patch, Pedunculated, Pigmented, Plaque, Poikiloderma, Purple, Purpura/Petechiae, Pustule, Salmon, Scale, Scar, Sclerosis, Telangiectasia, Translucent, Ulcer, Umbilicated, Vesicle, Warty/Papillomatous, Wheal, White(Hypopigmentation), Xerosis, Yellow.

SCHEMA SELECTION:
- Detect the schema used by REFERENCE (Derm7pt vs SkinCon). Use that schema for extraction/normalization and comparison. Do NOT switch schemas.

[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

A) EXTRACTION:
   Extract reasoning, normalized morph JSON in the SAME schema as REFERENCE, and one final diagnosis from both inputs. Derm7pt must contain all seven keys with one lowercase allowed value. SkinCon must contain the exact single key, closed-set items only, sorted alphabetically.

B) REASONING ALIGNMENT (lenient v2):
   Decompose GT reasoning into ≤25 atomic claims. Label each Supported | PartiallySupported | Vague | Missing | Contradicted. Count ExtraIncorrect only for specific, materially false statements. Let T=max(1,total_ref_claims):
   recall_like = (1.0*Supported + 0.6*PartiallySupported + 0.3*Vague) / T
   contrad_pen = 0.7*Contradicted / T
   missing_pen = 0.4*Missing / T
   extra_pen = 0.5*ExtraIncorrect / T
   reasoning_score = round(100 * max(0, recall_like - 0.5*contrad_pen - 0.2*missing_pen - 0.2*extra_pen), 1)

C) MORPH SEMANTICS:
   Compare candidate vs GT morph after normalization. Count supported/missing/contradicted/extra and score agreement in [0,1].

D) DIAGNOSIS SIMILARITY:
   Relation: Exact | Synonym | Parent | Child | Sibling/CloseDifferential | SameSuperfamily | UnrelatedPlausible | WrongSystem | Nonsense/NoAnswer.
   Similarity: Exact/Synonym=1.0; Parent/Child=0.85; Sibling/CloseDifferential=0.7; SameSuperfamily=0.5; UnrelatedPlausible=0.3; WrongSystem=0.1; Nonsense/NoAnswer=0.0.

E) CROSS-CONSISTENCY:
   Judge whether candidate reasoning contradicts candidate morph JSON; penalty [0,1].

STRICT JSON ONLY:
{{
  "extraction": {{"gt":{{"reasoning":"...","morph":{{}},"final_dx":"..."}},"cand":{{"reasoning":"...","morph":{{}},"final_dx":"..."}}}},
  "reasoning": {{"claims":[{{"text":"...","label":"Supported|PartiallySupported|Vague|Missing|Contradicted"}}],"counts":{{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},"score":0.0,"notes":"≤60 words"}},
  "morph_semantic": {{"schema":"SkinCon|Derm7pt","supported":0,"missing":0,"contradicted":0,"extra":0,"score_semantic":0.0,"notes":"≤60 words"}},
  "diagnosis": {{"gt_dx":"...","cand_dx":"...","relation":"Exact|Synonym|Parent|Child|Sibling/CloseDifferential|SameSuperfamily|UnrelatedPlausible|WrongSystem|Nonsense/NoAnswer","similarity":0.0,"score":0.0,"notes":"≤40 words"}},
  "cross_consistency": {{"penalty":0.0,"notes":"≤40 words"}},
  "short_feedback": "≤50 words"
}}"""


JUDGE_PROTOCOLS: dict[str, dict[str, Any]] = {
    "task_1_1_description_without_morphology": {
        "system": SYSTEM_11,
        "user": USER_11,
        "voters": 3,
        "max_tokens": 4096,
    },
    "task_1_2_description_with_morphology": {
        "system": SYSTEM_12,
        "user": USER_12,
        "voters": 3,
        "max_tokens": 8192,
    },
    "task_3_1_diagnostic_reasoning_without_morphology": {
        "system": SYSTEM_31,
        "user": USER_31,
        "voters": 1,
        "max_tokens": 8192,
    },
    "task_3_2_diagnostic_reasoning_with_morphology": {
        "system": SYSTEM_32,
        "user": USER_32,
        "voters": 1,
        "max_tokens": 10240,
    },
}


def prepare_batch(
    *,
    run_directory: Path,
    model: str = DEFAULT_BATCH_MODEL,
) -> dict[str, Any]:
    """Create an inline, text-only OpenRouter Batch request and local index."""

    run_directory = run_directory.resolve()
    manifest = _load_yaml_or_json(run_directory / "run_manifest.yaml")
    benchmark = manifest.get("benchmark", {})
    benchmark_id = (
        str(benchmark.get("id", ""))
        if isinstance(benchmark, dict)
        else ""
    )
    spec = resolve_dermobench_spec(benchmark_id)
    protocol = JUDGE_PROTOCOLS.get(spec.key)
    if protocol is None:
        raise ValueError(
            f"DermoBench task {spec.key!r} has deterministic scoring and "
            "does not require an LLM judge"
        )
    records = read_jsonl(run_directory / "predictions.jsonl")
    requests: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    skipped = 0
    for record_index, record in enumerate(records):
        response = record.get("response", {})
        metadata = record.get("metadata", {})
        candidate = (
            str(response.get("final_text", "")).strip()
            if isinstance(response, dict)
            else ""
        )
        if not candidate or not isinstance(metadata, dict):
            skipped += 1
            continue
        task_prompt = str(metadata.get("user_prompt", "")).strip()
        reference = str(metadata.get("reference_answer", "")).strip()
        if not task_prompt:
            prompts = _rendered_prompt_index(run_directory)
            task_prompt = str(
                prompts.get(str(record.get("task_id", "")), {}).get(
                    "user_prompt",
                    "",
                )
            ).strip()
        if not reference:
            raise ValueError(
                f"Prediction {record.get('task_id')!r} has no isolated "
                "DermoBench reference"
            )
        user = str(protocol["user"]).format(
            task_prompt=task_prompt,
            reference=reference,
            candidate=candidate,
        )
        for vote in range(1, int(protocol["voters"]) + 1):
            custom_id = f"dbj-{record_index:06d}-v{vote}"
            index[custom_id] = {
                "task_id": str(record.get("task_id", "")),
                "record_index": record_index,
                "vote": vote,
                "spec_key": spec.key,
            }
            requests.append(
                {
                    "custom_id": custom_id,
                    "body": {
                        "messages": [
                            {
                                "role": "system",
                                "content": str(protocol["system"]),
                            },
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.0,
                        "max_tokens": int(protocol["max_tokens"]),
                        "response_format": {"type": "json_object"},
                        "reasoning": {
                            "enabled": True,
                            "effort": "minimal",
                            "exclude": True,
                        },
                    },
                }
            )
    if not requests:
        raise ValueError("The run contains no judgeable DermoBench responses")
    output_dir = run_directory / "dermobench_judge"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": model,
        "requests": requests,
    }
    _write_json(output_dir / "batch_request.json", payload)
    _write_json(
        output_dir / "batch_index.json",
        {
            "schema_version": 1,
            "run_directory": str(run_directory),
            "benchmark_id": benchmark_id,
            "spec_key": spec.key,
            "judge_model": model,
            "judge_protocol": "upstream_text_only_flash_lite_batch_v1",
            "request_count": len(requests),
            "prediction_count": len(records),
            "skipped_prediction_count": skipped,
            "requests": index,
        },
    )
    return {
        "status": "prepared",
        "batch_request": str(output_dir / "batch_request.json"),
        "batch_index": str(output_dir / "batch_index.json"),
        "prediction_count": len(records),
        "request_count": len(requests),
        "voters_per_prediction": int(protocol["voters"]),
        "contains_multimodal_content": False,
        "model": model,
    }


def submit_batch(
    *,
    run_directory: Path,
    api_key_env: str = "OPENROUTER_API_KEY",
) -> dict[str, Any]:
    """Submit a prepared request without exposing the API key in artifacts."""

    output_dir = run_directory.resolve() / "dermobench_judge"
    payload = _load_json(output_dir / "batch_request.json")
    response = _request_json(
        OPENROUTER_BATCH_URL,
        method="POST",
        api_key=_required_api_key(api_key_env),
        payload=payload,
    )
    _write_json(output_dir / "batch_submission.json", response)
    return response


def fetch_batch(
    *,
    run_directory: Path,
    batch_id: str,
    api_key_env: str = "OPENROUTER_API_KEY",
) -> dict[str, Any]:
    """Fetch status and collect judgments automatically after completion."""

    response = _request_json(
        f"{OPENROUTER_BATCH_URL}/{batch_id}",
        method="GET",
        api_key=_required_api_key(api_key_env),
    )
    output_dir = run_directory.resolve() / "dermobench_judge"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "batch_status.json", response)
    if response.get("status") == "completed":
        return collect_batch(run_directory=run_directory, response=response)
    return {
        "id": response.get("id", batch_id),
        "status": response.get("status"),
        "request_counts": response.get("request_counts"),
        "terminal": response.get("status") in TERMINAL_BATCH_STATUSES,
        "collected": False,
    }


def collect_batch(
    *,
    run_directory: Path,
    response: dict[str, Any],
) -> dict[str, Any]:
    """Validate completed results, aggregate voters, and write judge metrics."""

    run_directory = run_directory.resolve()
    output_dir = run_directory / "dermobench_judge"
    index_document = _load_json(output_dir / "batch_index.json")
    request_index = index_document.get("requests")
    results = response.get("results")
    if response.get("status") != "completed" or not isinstance(results, list):
        raise ValueError("Only a completed OpenRouter batch can be collected")
    if not isinstance(request_index, dict):
        raise ValueError("DermoBench batch index is missing or invalid")
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_count = 0
    failed_count = 0
    for result in results:
        if not isinstance(result, dict):
            invalid_count += 1
            continue
        custom_id = str(result.get("custom_id", ""))
        local = request_index.get(custom_id)
        if not isinstance(local, dict):
            invalid_count += 1
            continue
        if result.get("error") is not None:
            failed_count += 1
            votes[str(local["task_id"])].append(
                {"status": "judge_error", "error": result.get("error")}
            )
            continue
        content = _batch_result_content(result)
        parsed = parse_json_output(content)
        if not parsed.recoverable_valid or not isinstance(parsed.decoded, dict):
            invalid_count += 1
            votes[str(local["task_id"])].append(
                {
                    "status": "judge_invalid",
                    "raw_text": content,
                    "error": parsed.error,
                }
            )
            continue
        votes[str(local["task_id"])].append(
            {
                "status": "ok",
                "vote": local.get("vote"),
                "judgment": parsed.decoded,
                "raw_json_valid": parsed.raw_valid,
                "recovery": parsed.recovery,
            }
        )
    spec_key = str(index_document.get("spec_key", ""))
    judgments: list[dict[str, Any]] = []
    for task_id in sorted(votes):
        valid = [
            vote["judgment"]
            for vote in votes[task_id]
            if vote.get("status") == "ok"
            and isinstance(vote.get("judgment"), dict)
        ]
        aggregate = _aggregate_votes(spec_key, valid) if valid else None
        judgments.append(
            {
                "task_id": task_id,
                "status": "ok" if aggregate is not None else "judge_invalid",
                "vote_count": len(votes[task_id]),
                "valid_vote_count": len(valid),
                "aggregate": aggregate,
                "votes": votes[task_id],
            }
        )
    _write_jsonl(output_dir / "judgments.jsonl", judgments)
    metrics = _judge_metrics(
        judgments,
        spec_key=spec_key,
        batch=response,
        invalid_result_count=invalid_count,
        failed_result_count=failed_count,
    )
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "batch_completed.json", response)
    return {
        "id": response.get("id"),
        "status": "completed",
        "collected": True,
        "judgments": str(output_dir / "judgments.jsonl"),
        "metrics": str(output_dir / "metrics.json"),
        "summary": metrics,
    }


def _aggregate_votes(
    spec_key: str,
    votes: list[dict[str, Any]],
) -> dict[str, Any]:
    if spec_key == "task_1_1_description_without_morphology":
        return {
            "overall": _mean_path(votes, "overall"),
            "accuracy": _mean_path(votes, "rubric", "accuracy"),
            "completeness": _mean_path(votes, "rubric", "completeness"),
            "consistency": _mean_path(votes, "rubric", "consistency"),
        }
    if spec_key == "task_1_2_description_with_morphology":
        text = _mean_path(votes, "text_judge", "overall")
        morph = _mean_path(votes, "morph_semantic", "score_semantic")
        cross = _mean_path(votes, "cross_consistency", "penalty")
        return {
            "text_overall": text,
            "morph_semantic": morph,
            "cross_consistency_penalty": cross,
            "final_overall": round(0.5 * text + 0.5 * 100.0 * morph, 2),
        }
    if spec_key == "task_3_1_diagnostic_reasoning_without_morphology":
        return {
            "reasoning_score": _mean_path(votes, "reasoning", "score"),
            "diagnosis_score": _mean_path(votes, "diagnosis", "score"),
            "diagnosis_similarity": _mean_path(
                votes,
                "diagnosis",
                "similarity",
            ),
            "overall": _mean_path(votes, "overall"),
        }
    if spec_key == "task_3_2_diagnostic_reasoning_with_morphology":
        reasoning = _mean_path(votes, "reasoning", "score")
        diagnosis = _mean_path(votes, "diagnosis", "score")
        morph = _mean_path(votes, "morph_semantic", "score_semantic")
        cross = _mean_path(votes, "cross_consistency", "penalty")
        text_block = 0.5 * reasoning + 0.5 * diagnosis
        final = 0.66 * text_block + 0.34 * 100.0 * morph
        return {
            "reasoning_score": reasoning,
            "diagnosis_score": diagnosis,
            "diagnosis_similarity": _mean_path(
                votes,
                "diagnosis",
                "similarity",
            ),
            "morph_semantic": morph,
            "cross_consistency_penalty": cross,
            "text_block": round(text_block, 2),
            "morph_block": round(100.0 * morph, 2),
            "final_overall": round(final, 2),
        }
    raise ValueError(f"Unsupported DermoBench judge protocol: {spec_key}")


def _judge_metrics(
    judgments: list[dict[str, Any]],
    *,
    spec_key: str,
    batch: dict[str, Any],
    invalid_result_count: int,
    failed_result_count: int,
) -> dict[str, Any]:
    valid = [
        item["aggregate"]
        for item in judgments
        if isinstance(item.get("aggregate"), dict)
    ]
    score_key = (
        "overall"
        if spec_key
        in {
            "task_1_1_description_without_morphology",
            "task_3_1_diagnostic_reasoning_without_morphology",
        }
        else "final_overall"
    )
    scores = [float(item.get(score_key, 0.0)) for item in valid]
    return {
        "protocol": "upstream_text_only_flash_lite_batch_v1",
        "comparability": (
            "Flash-Lite judge scores are not directly comparable with the "
            "paper's Gemini 2.5 Pro judge scores."
        ),
        "spec_key": spec_key,
        "sample_count": len(judgments),
        "valid_judgment_count": len(valid),
        "valid_judgment_rate": (
            len(valid) / len(judgments) if judgments else 0.0
        ),
        "mean_final_score": round(mean(scores), 4) if scores else 0.0,
        "invalid_batch_result_count": invalid_result_count,
        "failed_batch_result_count": failed_result_count,
        "batch_id": batch.get("id"),
        "batch_model": batch.get("model"),
        "batch_usage": batch.get("usage"),
    }


def _mean_path(values: list[dict[str, Any]], *path: str) -> float:
    numbers: list[float] = []
    for value in values:
        current: Any = value
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        try:
            numbers.append(float(current))
        except (TypeError, ValueError):
            numbers.append(0.0)
    return round(mean(numbers), 4) if numbers else 0.0


def _batch_result_content(result: dict[str, Any]) -> str:
    response = result.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    return str(message.get("content", "")) if isinstance(message, dict) else ""


def _rendered_prompt_index(run_directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("task_id", "")): record
        for record in read_jsonl(run_directory / "rendered_prompts.jsonl")
    }


def _required_api_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required API key environment variable is unset: {name}")
    return value


def _request_json(
    url: str,
    *,
    method: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter Batch API returned HTTP {exc.code}: {details}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter Batch API request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("OpenRouter Batch API returned a non-object response")
    return value


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        return _load_json(path)
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
