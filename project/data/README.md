# SFT generation data

Manifest (one JSON object per line):

```json
{"sample_id": "s001", "image_path": "data/raw/images/s001.jpg", "gold_diagnosis": "melanoma"}
```

Paths in `image_path` are relative to `project/` unless absolute. Run CLIs from the repo root.

```bash
uv run python -m project.stages.stage_a \
  --manifest project/data/raw/samples.jsonl \
  --output project/data/morphology/stage_a.jsonl

uv run python -m project.stages.stage_b \
  --manifest project/data/raw/samples.jsonl \
  --stage-a project/data/morphology/stage_a.jsonl \
  --output project/data/reasoning/stage_b.jsonl
```

Stage A JSONL never stores `gold_diagnosis`. Stage B reads gold from the manifest and frozen Stage A from `--stage-a`.
