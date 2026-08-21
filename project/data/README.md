# SFT generation data

The canonical source is the pinned private ISEPDistillDataset release
(`diagnosis` / `sft_train`) loaded through `project.dataset`. Images come from
the Hub `image` column and the human target comes from `gold_diagnosis`. Stage A
never sends that target to the teacher. The repository no longer ships a toy
`samples.jsonl`; an explicit local `--manifest` remains available only for
isolated tests or intentionally local inputs.

```bash
uv run python -m project.stages.stage_a --limit 5
uv run python -m project.stages.stage_b --limit 5

# Direct Vertex (Gemini Enterprise Agent Platform), after ADC is configured:
uv run python -m project.stages.stage_a \
  --config configs/teacher_configs/gemini_3_7_flash_vertex.yaml \
  --limit 5
```

For the normal E3 generation campaign, run both stages over the same ordered
cohort with one command:

```bash
uv run isep-generate-e3 \
  --config configs/teacher_configs/gemini_3_7_flash_vertex.yaml
```

The command displays a live progress bar for each stage with completed and
remaining images, accepted and failed records, ETA, the current sample, and—if
the provider config has pinned pricing—the cumulative estimated request cost. It
finishes all of Stage A before starting Stage B. Stage B is not launched if any
selected Stage A record is missing or failed, and neither stage automatically
retries a schema, safety, or clinical-validation failure. The Vertex client
does apply one bounded Tenacity policy to transient HTTP failures only: up to
six total attempts for 408, 429, 500, 502, 503, and 504, using exponential
backoff with jitter. The Google SDK retry loop is disabled to avoid nested
retries, and each durable row records the physical request-attempt count.

For Vertex, the checked-in Gemini 3.7 Flash config uses medium reasoning and
pins the published Standard/global USD token prices that apply through
2026-12-31. The estimate uses the response's input and total token counts, so
hidden thinking tokens are included as billed output. It is a list-price
estimate before credits, not the finalized Cloud Billing invoice or the
remaining promotional-credit balance.
An optional local soft ceiling stops the campaign after the current durable row
and before the next provider request:

```bash
uv run isep-generate-e3 \
  --config configs/teacher_configs/gemini_3_7_flash_vertex.yaml \
  --max-estimated-cost-usd 200
```

Use a USD ceiling comfortably below the absolute credit limit because the
published price is in USD, the credits are denominated in EUR, and billing or
exchange-rate adjustments are outside the generation response. For a provider-
enforced limit, configure a project- and service-scoped spend-cap budget in
Google Cloud Billing as well.

An interrupted command is safe to rerun: accepted JSONL records are detected
and skipped, while failed attempts remain in the audit trail. Use `--no-resume`
only for an intentional new attempt. Resume is rejected if the model, seed,
maximum output tokens, reasoning effort/exclusion policy, prompt, or schema hash
changed; use new output paths instead of mixing protocol versions. In
particular, do not resume a previous high-reasoning pilot with the current
medium-reasoning config. For an isolated five-image pilot, keep its outputs
separate from the canonical campaign:

```bash
uv run isep-generate-e3 \
  --config configs/teacher_configs/gemini_3_7_flash_vertex.yaml \
  --limit 5 \
  --stage-a-output project/data/dry_runs/e3_vertex_5/stage_a.jsonl \
  --stage-b-output project/data/dry_runs/e3_vertex_5/stage_b.jsonl
```

Example terminal state:

```text
Stage A [##################----------] 64/100 (64.00%) left=36 ok=64 failed=0 eta=05:12 current=sample-0064 est_cost=$2.1843/$200.00
```

Outputs:

- `project/data/morphology/stage_a.jsonl`
- `project/data/reasoning/stage_b.jsonl`

After both audit files have been validated, materialize the trainer-visible
multitask release:

```bash
uv run isep-materialize-e3 \
  --stage-a project/data/morphology/stage_a.jsonl \
  --stage-b project/data/reasoning/stage_b.jsonl \
  --hub-split sft_train
```

The default output is
`project/data/sft/e3_multitask/sft_train.parquet`, accompanied by
`sft_train.manifest.json` with row counts, task counts, byte size, and SHA-256.
The Parquet contains the image and one isolated `messages` conversation per
task:

- every source sample contributes the human-supervised `diagnosis` row;
- an accepted Stage A adds `morphology` and `caption` rows;
- an accepted Stage B adds exactly one conditional clinical behavior:
  `grounded_differential` for evaluable images or `request_new_image` for
  non-evaluable images.

A fully accepted source sample therefore produces four rows, not five
contradictory rows. The diagnosis prompt is replayed byte-for-byte from the
frozen source release. The teacher's `clinical_reasoning` is preserved verbatim
for the Stage B target; structured Stage A morphology is serialized as canonical
JSON. Missing generation attempts fail closed unless `--allow-partial` is
explicitly selected, and existing outputs are protected unless `--overwrite`
is supplied.

Stage A stores an image-assessment block and atomic observations (`obs_001`,
`obs_002`, ...). Stage B cites those immutable IDs, records whether the gold
anchor is visually supported, weak, or unsupported, and compares the gold
explicitly with plausible alternatives. The teacher also writes
`reasoning.clinical_reasoning`, a concise natural-language clinical
justification preserved verbatim as the Student's SFT target. Structured
evidence remains available in the same row for validation and audit; no local
template renderer rewrites the target.

Evaluable images always use `ANSWER_DIFFERENTIAL`; missing clinical information
is a limitation, not a question. `REQUEST_NEW_IMAGE` is reserved for inputs that
Stage A marked non-evaluable. Annotation conflicts and unsupported evaluable
anchors remain auditable but are excluded from training targets.

Rows also include hashes of the exact encoded image, prompt, schema, model
identity, seed, and attempt.

Both commands are fail-closed: exhausted API retries, schema errors, rejected
Stage B records, or missing accepted Stage A inputs produce a non-zero exit.
Only configured transient Vertex statuses are retried within a logical request;
rerunning a rejected clinical record remains an explicit new generation
attempt preserved in JSONL.

Optional local-file JSONL (`--manifest`) still works for tests. Stage A JSONL
never stores `gold_diagnosis`.
