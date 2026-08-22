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
remaining images, accepted, rejected, and failed records, ETA, the current
sample, and—if the provider config has pinned pricing—the cumulative estimated
request cost. It
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

An interrupted command is safe to rerun. Stage A successes are skipped. In
Stage B, both `ok` and `rejected` are terminal and skipped: `ok` may enter the
training release, while `rejected` remains an auditable quality-gate exclusion.
Only `error` remains retryable on a later invocation, and every failed attempt
stays in the JSONL audit trail. Use `--no-resume` only for an intentional new
attempt. Resume is rejected if the model, seed,
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
Stage A [##################----------] 64/100 (64.00%) left=36 ok=64 rejected=0 failed=0 eta=05:12 current=sample-0064 est_cost=$2.1843/$200.00
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
`sft_train.manifest.json` with row counts, task counts, byte size, SHA-256, and
per-sample Stage B coverage (`ok`, `rejected`, `error`, `missing_attempt`, or
`not_eligible_stage_a`). The manifest also records rejection reasons, retryable
errors, missing IDs, attempt IDs, and duplicate-attempt counts.
The Parquet contains the image and one isolated `messages` conversation per
task:

- every source sample contributes the human-supervised `diagnosis` row;
- an accepted Stage A adds `morphology` and `caption` rows;
- an accepted Stage B adds exactly one conditional clinical behavior:
  `grounded_differential` for evaluable images or `request_new_image` for
  non-evaluable images.

An image with accepted Stage A but no accepted Stage B still contributes its
human diagnosis, morphology, and caption rows. It contributes no Stage B target,
and its precise state is retained in the manifest: `rejected` when the clinical
gate excluded it, `error` after a provider/schema failure, or `missing_attempt`
when no Stage B call exists. Thus reduced coverage is visible rather than
silently dropping the image.

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

Stage A and transport/schema failures remain fail-closed. Stage B clinical
rejection is a successful terminal audit outcome: it is excluded from SFT and
is not sent to the provider again on resume. Only configured transient Vertex
statuses are retried within a logical request; an exhausted `error` row may be
retried by a later campaign invocation and every attempt remains in JSONL.

## Frozen E3 prompt protocol

The current teacher prompts are frozen as `e3_stage_a_v1` and
`e3_stage_b_v1`. Both OpenRouter and Vertex configs pin their exact SHA-256
digests, and config loading aborts before any request if the bytes drift. The
registry is in `configs/teacher_configs/prompts/README.md`. Any prompt change
requires explicit v2 files, new output paths, and a fresh validation pilot;
frozen v1 files must not be edited in place.

Optional local-file JSONL (`--manifest`) still works for tests. Stage A JSONL
never stores `gold_diagnosis`.

## Frozen E3 Stage A release

The final accepted-only Stage A release is:

`project/data/morphology/frozen/e3_stage_a_v1_20260822/`

It contains 6,312 ordered `ok` rows for the 6,312 unique source IDs, the exact
frozen prompt, response schema, Vertex teacher config, and
`freeze_manifest.json`. The accepted JSONL SHA-256 is
`1eefa665d791c5138ffc00d57c5d9161ab899985949d8d4c2f7e54d12db89bd2`.
The mutable audit log remains at `project/data/morphology/stage_a.jsonl` and
retains all 6,318 attempts. Batch/schema failures are stored outside the
accepted release, including the v3 quarantine sidecar.

The earlier `e3_stage_a_v1_20260821` freeze is superseded by the 2026-08-22
release: target content is unchanged, but the final release corrects the
canary's stored estimate from Standard to Batch list pricing. Full generation,
cost, integrity, leakage, and known prompt-adherence limitations are recorded
in `annotations/dataset_pipeline/14_e3_stage_a_teacher_dataset_freeze.md`.

## Vertex Batch transport for Stage B

`isep-stage-b-batch` changes only the transport of the frozen `e3_stage_b_v1`
protocol. It joins each source image with the accepted-only 2026-08-22 Stage A
release and the private gold diagnosis, and emits the same Stage B response
schema as the synchronous pipeline. Ingestion re-applies `parse_stage_b` and
the deterministic `validate_stage_b` gate. Accepted and clinically rejected
rows are terminal; provider or schema errors remain auditable and retryable.

Prepare a local one-record canary first:

```bash
uv run isep-stage-b-batch prepare \
  --stage-a project/data/morphology/frozen/e3_stage_a_v1_20260822/stage_a.jsonl \
  --stage-b-output project/data/reasoning/stage_b.jsonl \
  --work-dir project/data/batch_jobs/e3_stage_b_batch_canary_v1_20260822 \
  --gcs-prefix gs://BUCKET/isep/e3/stage-b-batch-v1/CAMPAIGN \
  --pending-limit 1
```

Preparation is local and performs no external transfer. Unlike Stage A, both
`requests.jsonl` and `items.jsonl` contain the private gold label. Upload is
therefore fail-closed unless the user has explicitly authorized this transfer
and the acknowledgement flag is supplied:

```bash
uv run isep-stage-b-batch upload \
  --work-dir project/data/batch_jobs/e3_stage_b_batch_canary_v1_20260822 \
  --authorize-private-gold-upload

uv run isep-stage-b-batch submit \
  --work-dir project/data/batch_jobs/e3_stage_b_batch_canary_v1_20260822

uv run isep-stage-b-batch status \
  --work-dir project/data/batch_jobs/e3_stage_b_batch_canary_v1_20260822

uv run isep-stage-b-batch download \
  --work-dir project/data/batch_jobs/e3_stage_b_batch_canary_v1_20260822

uv run isep-stage-b-batch ingest \
  --work-dir project/data/batch_jobs/e3_stage_b_batch_canary_v1_20260822 \
  --stage-a project/data/morphology/frozen/e3_stage_a_v1_20260822/stage_a.jsonl \
  --stage-b-output project/data/reasoning/stage_b.jsonl
```

Every prepared item binds the Stage A attempt ID, morphology SHA-256, exact
JPEG preprocessing manifest, source revision, gold diagnosis, prompt/schema
hashes, and Batch pricing. Ingestion refuses a different Stage A file or a
different joined morphology. A full Batch must use a new, empty work directory
and GCS prefix after the canary has passed; no command deletes remote objects.

## Frozen E3 Stage B release

The completed Stage B release is frozen at:

`project/data/reasoning/frozen/e3_stage_b_v1_20260822/`

It has exact terminal coverage of all 6,312 source IDs: 6,148 accepted rows in
`stage_b.jsonl`, 164 clinically rejected rows in `rejected.jsonl`, zero errors,
and zero missing attempts. The accepted JSONL SHA-256 is
`53d4e67ac6909aefea5f880c2aba9c77cc238301b95fa4b4b73ee32a27fc1ae7`;
the rejected JSONL SHA-256 is
`a697cc6614b2d059498c42872949a56afe90d2d442d89c8b947414ca175047b0`.
The directory also contains the exact prompt, response schema, Vertex teacher
config, normalization audit, and a hash-addressed `freeze_manifest.json`.

Thirty-five provider outputs violated one structural rule by citing the same
Stage A observation as evidence for both diagnoses in one comparison. The raw
responses and both failed attempts remain in the mutable audit trail. A frozen,
deterministic normalization removed each intersecting observation ID from both
evidence lists and reran the canonical parser and clinical gate. It changed
neither `diagnosis` nor `clinical_reasoning`; all 35 terminal content hashes are
bound to `normalization/01_overlap_normalization.jsonl` (SHA-256
`df37f514ba548cdd59cabcbbf2288a1d4a9129ebea804d3c0bb8e653eef612d3`).
The estimated Batch list-price cost represented by all physical Stage B
requests is USD 24.7412955 before credits; this is not a finalized Cloud
Billing invoice.

The final trainer-visible multitask release is
`project/data/sft/e3_multitask/sft_train.parquet`: 25,084 rows and
18,202,179,385 bytes, SHA-256
`d80868921439542f0f11f327c30be94cef3bc030330ed37eb5666c01531e80e6`.
Its task counts are 6,312 diagnosis, 6,312 morphology, 6,312 caption, 6,127
grounded differential, and 21 request-new-image rows. The 164 rejected Stage B
samples still contribute the three Stage A/human-supervised tasks but no Stage
B target; their exclusion and reason codes remain explicit in the materializer
manifest.
