# SFT generation data

Manifest (one JSON object per line):

```json
{"sample_id": "s001", "image_path": "data/raw/images/s001.jpg", "gold_diagnosis": "melanoma"}
```

Default source is ISEPDistillDataset (`diagnosis` / `sft_train`) via `project.dataset`. Images come from the Hub `image` column; gold comes from `gold_diagnosis`. Stage A never sends gold to the teacher.

```bash
uv run python -m project.stages.stage_a --limit 5
uv run python -m project.stages.stage_b --limit 5
```

Outputs:

- `project/data/morphology/stage_a.jsonl`
- `project/data/reasoning/stage_b.jsonl`

Optional local-file JSONL (`--manifest`) still works for tests. Stage A JSONL never stores `gold_diagnosis`.
