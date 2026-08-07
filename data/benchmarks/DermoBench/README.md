# DermoBench local release

This directory contains the byte-for-byte official DermoBench task annotations
and gated image archive plus a derived, training-leakage-filtered evaluation
view. It is an external benchmark, separate from the frozen ISEPDermaBench
protocols used by `src.benchmark.cli`.

## One-time local setup

The release is already present when the following exist:

- `release/task*/...json` and `.jsonl` annotation files;
- `release/dermobench_release_imgs.zip` (about 3.1 GB).

Create the local image tree and a reproducible path index with:

```bash
python -m src.data_pipeline.dermobench --extract
```

This extracts images to `release/images/` (about 3.1 GB more disk space) and
writes `release/image_index.json`. Both are ignored by Git. The operation can
be rerun safely after an interruption; files with the expected size are kept.

Run the same command without `--extract` to validate annotations and recreate
the index only:

```bash
python -m src.data_pipeline.dermobench
```

## Why the image index is required

The annotation field `image` is the canonical lookup key. Most paths match the
archive exactly; 3,396 Derm1M-EDU names differ only because the archive
normalizes punctuation and Unicode differently. `image_index.json` maps every
annotation path to the actual path below `release/images/`; consumers should
therefore resolve an annotation image as:

```python
from pathlib import Path
import json

release = Path("data/benchmarks/DermoBench/release")
index = json.loads((release / "image_index.json").read_text())
image_path = release / index["archive"]["image_root"] / index["image_paths"][row["image"]]
```

Do not rename the released annotations or images: that would make comparison
with the upstream task files non-reproducible.

## Scope and use

DermoBench contains 31,999 VQA tasks across morphology, diagnosis, reasoning,
and fairness. Its files use the conversation schema shown below; JSONL files
contain one object per line and JSON files contain an array.

```json
{
  "id": "sample_id",
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\\nQuestion"},
    {"from": "gpt", "value": "Reference answer"}
  ]
}
```

The official download was verified against the Hugging Face release: all 13
annotation files and the image ZIP match by SHA-256. The published package has
31,999 task rows and 12,371 referenced images. The paper's 33,999 count also
includes a 2,000-row hierarchical Task 2.3 that is not present in the public
Hub release; this is an upstream release discrepancy, not a local download
failure. Some official files also contain duplicate task IDs, so consumers
must use `(annotation_file, row_index)` when a globally unique execution key
is required.

Build and validate the thesis evaluation view with:

```bash
python -m src.data_pipeline.dermobench_evaluation
python -m src.data_pipeline.dermobench_evaluation --validate-only
```

The official `release/` remains immutable. The derived `evaluation/tasks/`
view contains 29,099 tasks after removing 2,900 rows associated with 863
training-overlapping images. Exclusion uses exact image SHA-256 plus available
SCIN case, PAD patient, Fitzpatrick source-image, and HIBA patient identities.

Open-ended Tasks 1.1, 1.2, 3.1, and 3.2 use Gemini 3.5 Flash-Lite through
`configs/models/gemini_3_5_flash_lite_openrouter.yaml` and the upstream
task-specific, text-only judge rubrics. These scores must be labeled as the
Flash-Lite judge protocol; they are not directly comparable to DermoBench's
published Gemini 2.5 Pro judge scores. MCQ tasks remain deterministic.

## Running the filtered tasks

The thesis CLI exposes all 13 released tasks through the dedicated adapter.
Use `list-benchmarks` to see the full IDs and run a local validation before
calling a model:

```bash
python -m src.benchmark.cli list-benchmarks

python -m src.benchmark.cli run \
  --model qwen_3_5_4b \
  --benchmark dermobench/task_2_1_diagnosis_mcq_4_choices \
  --evaluation-set filtered \
  --limit 10 \
  --dry-run
```

The adapter resolves images through `image_index.json`, preserves each
upstream user prompt and option set, appends the official MCQ answer-control
sentence, and creates a globally unique execution ID from the task file and
row number. MCQ accuracy is deterministic. Task 4 additionally reports
accuracy by DDI skin-tone group and the upstream fairness score.

Tasks 1.1, 1.2, 3.1, and 3.2 retain the evaluated model's free-text response
and remain judge-pending until the second stage. Prepare an auditable,
text-only OpenRouter batch after the model run:

```bash
python -m src.benchmark.cli dermobench-judge-batch \
  --run outputs/dermobench/<benchmark>/<model>/<run>
```

Inspect `dermobench_judge/batch_request.json`, then add `--submit` to send it.
Poll and collect a completed batch with the returned ID:

```bash
python -m src.benchmark.cli dermobench-judge-batch \
  --run outputs/dermobench/<benchmark>/<model>/<run> \
  --batch-id batch_123
```

Evaluated-model inference remains synchronous and multimodal. Only the
text-vs-text judge requests enter the Batch API; its payload contains no image,
audio, video, or file blocks. DermoBench must be used for research evaluation
only, not for clinical deployment or patient-facing diagnosis.
