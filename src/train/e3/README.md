# E3 hard-KD contracts

This package owns the teacher-derived Stage-A/Stage-B contracts and the five
task-isolated student renderings for E3. It remains deliberately separate from
E2, whose supervision is human-only.

E3 is **not registered as a runnable training phase**. Registration stays
blocked until an accepted, versioned teacher release exists. Importing this
package cannot call a teacher, publish data, or launch training.

## Canonical teacher records

`domain.py` separates two versioned generations:

- `StageATarget`: image assessment with explicit `clinical_photo`, `dermoscopy`,
  or `unknown` modality; dominant visual pattern; normalized atomic
  observations; non-assessable features; and one complete clinical caption;
- `StageBTarget`: explicit corrections plus two separate internal results:
  `diagnostic_assessment` (ranked differential, evidence, missing
  discriminators, confidence/risk and rationale) and `context_decision`
  (information sufficiency, one exclusive response policy, and explicit
  context questions);
- `TeacherTargetBundle`: an aggregate Stage-B state plus independent diagnostic
  and context-policy `accepted`, `rejected`, `not_applicable`, or
  `not_generated` reviews, together with immutable teacher, prompt, provider
  and call-outcome provenance.

Every attempted call has one typed `generation_status`: `succeeded`,
`provider_safety_refusal`, `transport_error`, `timeout`, `empty_response`, or
`invalid_schema`. Sanitized provider metadata may include response/request IDs,
finish reason, error code, and structured safety categories; raw prompts,
provider messages, headers, tokens, and secrets are not part of this contract.

Stage A must always have `gold_visible_to_teacher=false`. The active Stage-B v2
prompt has `gold_visible_to_teacher=true`: it receives the private normalized
gold as the required leading diagnostic anchor. Historical Stage-B v1 remains
answer-blind for controlled comparison. Any generated Stage B, whether later
accepted or rejected, requires accepted Stage A.
Evidence and correction IDs in B must resolve to
observations in A. This allows partial acceptance: a valid A remains usable for
perception and caption tasks even when B is rejected.

Scientific review applies only to `succeeded` calls. Failed calls use
`not_applicable`, cannot carry a target or review rejection reasons, and are
never recoded as clinical uncertainty. A Stage-A safety refusal prevents the
Stage-B call. A Stage-B safety refusal preserves accepted Stage-A tasks. The
generation pilot must not retry, switch provider, or weaken safety settings
silently.

The two Stage-B results are reviewed and materialized independently. Diagnostic
rendering requires compliance with the private-gold anchor and does not read
the context policy.
Context-policy rendering does not require private-gold agreement and does not
copy the diagnostic prose target; its local gate instead checks schema,
taxonomy, question uniqueness, and whether a context request separates the
leading hypothesis from at least one alternative. This is a structural and
actionability gate. Its prompt explicitly asks whether the image would be
sufficient at deployment independently of the supplied anchor. This is not
proof that the teacher's clinical policy is correct, so the release still
requires clinical audit.

## Frozen Stage-A terminology

The active Stage-A v2 prompt does not search the web at generation time. It
receives the complete, frozen `e3_dermatology_terminology_v1` catalogue from
`resources/dermatology_terminology_v1.yaml`. The catalogue contains 66
diagnosis-free concepts with definitions, modality constraints, observability
limits, source references, and an immutable resource hash. Its source hierarchy
is:

1. the [2016 ILDS revised glossary](https://www.ilds.org/what-we-do/project-and-programme/glossary-for-dermatology-terms/)
   for clinical lesion morphology, surface, shape, profile and distribution;
2. the [AAD morphology module](https://www.aad.org/member/education/residents/bdc/morphology)
   and [DermNet terminology](https://dermnetnz.org/topics/terminology) as
   systematic educational cross-checks;
3. the [2016 International Dermoscopy Society consensus](https://pubmed.ncbi.nlm.nih.gov/26896294/)
   only for images explicitly classified as dermoscopy.

Each Stage-A observation now uses the hybrid fields `concept_id`, exact
`concept_label`, and optional visible `concept_detail`. The provider schema
enumerates the 66 permitted IDs; local review independently rejects unknown
IDs, label mismatches, dermoscopy terms used for clinical/unknown images, and
all canonical diagnosis leakage. Palpation, symptoms, history and temporal
claims remain forbidden because they cannot be established from one image.

`stage_a_teacher_prompt.yaml` remains unchanged as the v1 prompt artefact.
`stage_a_teacher_prompt_v2.yaml` is the active prompt and contains four
contrastive field-level examples: a visible scale, absence limited to the shown
scope, non-assessable global distribution, and colour uncertainty. The examples
contain no disease association and are explicitly non-copyable unless visible.

## Student-visible tasks

`E3HardKDPhase` renders one row at a time with an unambiguous prompt and target:

| Variant | Required accepted source | Student target |
|---|---|---|
| `diagnosis` | private normalized gold only | exact canonical label |
| `morphology` | Stage A | canonical JSON excluding prose caption |
| `caption` | Stage A | complete answer-blind clinical caption |
| `grounded_differential` | Stage A and accepted Stage-B diagnostic review | complete deterministic description, leading diagnosis, alternatives, evidence, confidence, and missing discriminators |
| `context_policy` | Stage A and accepted Stage-B context-policy review | canonical sufficiency decision and either `ANSWER_DIFFERENTIAL` or `REQUEST_CONTEXT`; the latter includes explicit questions |

`target_text` is copied into the assistant message and is the field supervised
by assistant-only loss. `target_source_fields` records which canonical fields
created it. Gold metadata never enters non-diagnosis user prompts. Formal
follow-up questions enter only the isolated `context_policy` target and never
the diagnostic, morphology, caption, or grounded-differential targets.

## Quality gates

The strict models reject:

- surrounding whitespace, multiline captions, captions below five words, or
  captions that do not end at a sentence boundary;
- known incomplete clause endings such as `It is` or `which is`;
- duplicate observation IDs, non-contiguous differential ranks, duplicate
  diseases, contradictory evidence links, and unknown A-to-B references;
- inconsistent context policies, missing explicit questions, malformed request
  priorities, or request disease IDs outside the Stage-B differential;
- an accepted stage without provenance or with rejection reasons;
- any Stage-A call with gold visible to the teacher;
- a failed provider call presented as a scientific rejection, a successful call
  marked `not_applicable`, or any failed call carrying a teacher target;
- duplicate or misplaced provider safety categories;
- any generated Stage B when Stage A was not accepted;
- Stage A text containing a canonical diagnosis term;
- Stage-A terminology IDs outside the frozen catalogue, mismatched canonical
  labels, or dermoscopy-only concepts without confirmed dermoscopy;
- an accepted Stage-B leading diagnosis that disagrees with private gold for
  the `grounded_differential` task;
- duplicate context questions, request diseases outside the closed taxonomy,
  or a `REQUEST_CONTEXT` target that does not distinguish the leading
  hypothesis from at least one alternative.

Diagnosis replay is intentionally independent of teacher acceptance. If both
teacher stages fail, the human gold row may still train `diagnosis`; invalid
teacher rationale is never rescued by a correct label.

## Deterministic rendering

`rendering.py` creates byte-stable Stage-A JSON, canonical context-policy JSON,
and natural grounded differentials. The latter use one of twelve frozen surface templates selected
from `SHA-256(renderer_version + sample_id)`. All templates express the same
accepted diagnostic facts and never read `StageBTarget.context_decision` or the
raw teacher rationale. The context renderer reads only `context_decision`, maps
private disease IDs to canonical labels, and preserves the explicit questions.

Every formatted row records:

- phase and task identity;
- `target_text` and SHA-256;
- exact canonical source fields;
- Stage-A/Stage-B generation IDs when used;
- template and renderer versions when applicable;
- the two-message multimodal conversation consumed by the collator.

## Live generation progress

`progress.py` provides a backend-neutral event store used by the API or local
generation runner. The runner starts
one immutable `E3CampaignSpec`, then appends exactly one terminal
`E3ProgressEvent` for each attempted sample/stage. Duplicate terminal events
are rejected so that a retry cannot silently replace the first outcome.

The store publishes four local artifacts below the campaign output directory:

```text
campaign_manifest.json  immutable teacher, prompt, terminology and backend identity
generations.jsonl       append-only, fsync-ed terminal Stage-A/B events
campaign_status.json    atomic aggregate snapshot for automation
report.html             self-refreshing local progress dashboard
```

The progress record is deliberately sanitized. It cannot contain images,
prompts, raw teacher responses, provider messages, headers, secrets, or private
gold labels. After a successful Stage-B generation, the private evaluator must
record only the booleans `leading_label_match` and `gold_in_top3`. For Stage-B
v1 these are answer-blind diagnostic agreement metrics. For the active
gold-conditioned Stage-B v2 they are anchor-compliance checks and must not be
reported as independent diagnostic accuracy.

The generation runner will use the store as follows:

```python
from datetime import UTC, datetime
from pathlib import Path

from src.train.e3 import (
    E3CampaignSpec,
    E3CampaignState,
    E3GenerationStage,
    E3ProgressEvent,
    E3ProgressStore,
    StageReviewStatus,
    TeacherGenerationStatus,
)

store = E3ProgressStore.start(
    Path("outputs/training/e3_teacher_generation/pilot"),
    E3CampaignSpec(
        campaign_id="e3-pilot-qwen-3-6-27b",
        total_samples=100,
        provider="modal",
        backend="vllm_endpoint",
        teacher_model="Qwen/Qwen3.6-27B",
        teacher_revision="<pinned-commit>",
        stage_a_prompt_id="e3-stage-a-v1",
        stage_b_prompt_id="e3-stage-b-v1",
    ),
)

# Call immediately after one terminal Stage-A outcome.
store.record(
    E3ProgressEvent(
        event_id="<unique-generation-event-id>",
        recorded_at=datetime.now(UTC),
        sample_id="<private-stable-sample-id>",
        stage=E3GenerationStage.STAGE_A,
        generation_status=TeacherGenerationStatus.SUCCEEDED,
        review_status=StageReviewStatus.ACCEPTED,
    )
)

# Only after every sample has reached a terminal pipeline outcome.
store.finalize(E3CampaignState.COMPLETED)
```

Watch the atomic status from a second terminal without mutating the campaign:

```bash
uv run isep-e3-progress \
  outputs/training/e3_teacher_generation/pilot
```

For one machine-readable snapshot, use `--once --json`. Opening `report.html`
shows the same progress in a browser and refreshes every two seconds while the
campaign is running. The dashboard includes Stage-A/B generation and review
counts, typed provider failures, throughput, ETA, token use, response-policy
counts, separate Stage-B diagnostic/context-policy acceptance, private
leading-label/top-3 match rates, and currently materializable student rows per
task.

## Logical flow

```text
teacher Stage A (image, no gold)
  -> record typed generation_status
       provider failure -> not_applicable; Stage B not_generated
       succeeded        -> parse StageATarget -> automatic/human review
  -> accepted or rejected

accepted Stage A + original image (still no gold)
  -> teacher Stage B
  -> record typed generation_status
       provider failure -> not_applicable; preserve accepted Stage A
       succeeded        -> parse StageBTarget
       diagnostic_assessment -> ranked, evidence-linked differential
       context_decision      -> sufficiency + one exclusive response policy
          sufficient         -> ANSWER_DIFFERENTIAL, no requests
          insufficient       -> REQUEST_CONTEXT, explicit questions required
  -> independent local reviews
       diagnostic review
          validate taxonomy/evidence links
          compare lead with private gold only after generation
       context-policy review
          validate schema/taxonomy/question uniqueness/actionability
          do not require private-gold agreement
  -> aggregate Stage B accepted when at least one subtarget is accepted

TeacherTargetBundle
  -> E3HardKDPhase(task)
       diagnosis              <- gold only
       morphology             <- accepted A
       caption                <- accepted A
       grounded_differential  <- accepted A + accepted diagnostic subtarget
       context_policy         <- accepted A + accepted context-policy subtarget
  -> E3FormattedExample.as_record()
  -> messages[-1].assistant = target_text
  -> future assistant-only SFT collator
```

## Executable GPT-5.6 Sol pilot

The first end-to-end generation slice is frozen by
`configs/training/e3_teacher_generation_gpt_5_6_sol_medium.yaml`. It uses
`gpt-5.6-sol` through the Azure/OpenAI Responses API, strict JSON-schema output,
explicit `reasoning_effort=medium`, the Stage-A v2 prompt, and the frozen
terminology v1 resource. The provider-managed alias is recorded as such; it is
not misrepresented as an immutable model revision. Each response also records
the model identity reported by the provider when available.

The offline preflight reads no credentials and makes no external call. It
validates the model, prompts, terminology identity and source references, both
strict schemas, all release row metadata, the deterministic 100-case selection,
selected shard hashes, image hashes, image decoding, class/split balance, and
leakage-group uniqueness:

```bash
uv run isep-e3-teacher dry-run \
  outputs/training/e3_teacher_generation/gpt-5-6-sol-medium-v2-dry-run
```

For a smaller offline gate, `--limit` validates a deterministic prefix without
changing the frozen 100-case pilot or the one-case smoke contract:

```bash
uv run isep-e3-teacher dry-run \
  outputs/training/e3_teacher_generation/gpt-5-6-sol-medium-v2-dry-run-25 \
  --limit 25
```

The one-case smoke requires an explicit acknowledgement that one private image
will be sent to the configured provider:

```bash
uv run isep-e3-teacher smoke \
  outputs/training/e3_teacher_generation/gpt-5-6-sol-medium-v2-smoke \
  --confirm-external-image-upload
```

A bounded quality slice uses its own frozen configuration and treats its first
case as an in-campaign A+B schema/transport gate. A failed gate stops without a
retry; a successful gate continues without sending the first image again:

```bash
uv run isep-e3-teacher quality \
  outputs/training/e3_teacher_generation/gpt-5-6-sol-medium-v2-quality-25 \
  --config \
  configs/training/e3_teacher_generation_gpt_5_6_sol_medium_quality_25.yaml \
  --confirm-external-image-upload
```

Only a completed, identity-compatible smoke unlocks the 100-case pilot:

```bash
uv run isep-e3-teacher pilot \
  outputs/training/e3_teacher_generation/gpt-5-6-sol-medium-v2-pilot \
  --approved-smoke \
  outputs/training/e3_teacher_generation/gpt-5-6-sol-medium-v2-smoke \
  --confirm-external-image-upload
```

There is no automatic retry, provider reroute, answer repair, or schema repair.
`stage_results.jsonl` stores only parsed targets and sanitized provenance;
`teacher_bundles.jsonl` stores the final private A/B bundle. Raw prompts and raw
responses are never persisted. A deterministic generation ID and append-only
stores make explicit `--resume` safe: completed stages are reconciled into the
progress dashboard instead of being called again.

## Current boundary

Configuration loading, deterministic diagnosis-row selection, offline
integrity and terminology preflight, answer-blind Stage A plus gold-conditioned
GPT-5.6 Luna Stage B calls, two-stage validation,
guardrail classification, durable resume, private bundle output, and the live
progress interface are executable and CPU-tested. The real smoke and pilot are
deliberately not launched by tests. Release materialization, task mixing, E3
student training/evaluation, and phase-registry integration remain blocked
until the 100-case generation pilot passes its scientific and integrity gates.
