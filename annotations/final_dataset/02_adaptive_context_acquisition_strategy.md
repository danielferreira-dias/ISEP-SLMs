# Adaptive clinical-context acquisition strategy

## Material Passport

- Origin: thesis design discussion and targeted literature review.
- Date: 2026-08-15.
- Status: `PLANNED`; no training data, model checkpoint, or benchmark result is
  claimed by this document.
- Scope: a future interactive extension after E2 structured supervision, E3
  hard distillation, and final checkpoint selection are frozen.
- Version: `adaptive_context_acquisition_v1`.

## 1. Decision

Patient context can improve dermatology diagnosis, but a system that merely
receives all metadata at once does not demonstrate that it can recognise missing
information or request the most useful next fact. The thesis will therefore
separate two questions:

1. **Context utility:** how much does a fixed, authorised context bundle change
   diagnostic performance relative to image only?
2. **Context acquisition:** can the model decide whether to diagnose, request a
   context field, request a better image, or abstain, and can it select a useful
   next question?

The existing clinical-context ablation addresses the first question. The second
is a later agentic capability and must not be presented as an E2 or E3 result.

## 2. Why a prompt alone is insufficient

A system prompt can ask a model to request more information when uncertain. It
is a necessary baseline but not the main intervention. A prompt does not supply:

- positive examples of when a diagnosis is safe versus premature;
- a controlled vocabulary of permissible questions;
- supervision for choosing one discriminating field instead of several generic
  questions;
- ground truth to assess whether the chosen question reduced uncertainty.

MediQ directly motivates this distinction: it frames clinical reasoning as an
interactive task in which an expert should withhold a diagnosis under incomplete
information and ask follow-up questions. Its authors report that direct
prompting of strong LLMs to ask questions degraded performance, while explicit
abstention strategies improved diagnostic accuracy; this remains a preprint and
does not establish the result for dermatology specifically.

MedClarify proposes selecting questions by expected information gain over a
differential diagnosis and reports fewer errors than a single-shot baseline.
This is also a preprint and should be treated as design evidence, not as a
guarantee for the ISEP data.

## 3. Evidence that context can matter in dermatology

PAD-UFES-20 was designed with smartphone clinical images and patient clinical
data. It contains up to 21 patient features, including demographic, history and
lesion-related information. Multimodal skin-lesion work on this corpus reports
better internal classification metrics when image features are fused with such
metadata. For example, Khurshid et al. compare no-metadata, concatenation and
other multimodal models on PAD-UFES-20.

This evidence supports testing context, but it has important limits:

- it is mostly conventional image-classification work, not an interactive VLM;
- gains in a single corpus can contain source, prevalence, demographic, or
  label-proxy shortcuts;
- an image-plus-all-metadata result is not evidence that a model knows which
  question to ask;
- metadata must be available at inference and authorised for the intended use.

The ISEP thesis claim must consequently report image-only, full-context, and
interactive conditions separately and retain the same leakage-group split.

## 4. Scope of context

The first context ontology should be small, clinically meaningful, and based on
fields actually available with valid provenance. Candidate question families are:

| Family | Example field | Why it can discriminate |
|---|---|---|
| Onset and evolution | duration; recent enlargement or change | separates chronic stable lesions from evolving presentations. |
| Symptoms | itch; pain; burning; bleeding | can distinguish visually similar inflammatory or neoplastic conditions. |
| Distribution | body site; localised versus widespread | links morphology to anatomical pattern. |
| Exposure and triggers | contact; medication; sun exposure | can alter the differential without claiming a visible finding. |
| Prior course | treatment tried and response | distinguishes persistent disease from treatment-responsive alternatives. |
| Relevant history | immunosuppression; personal or family skin-cancer history | changes risk assessment and escalation, not necessarily the visible morphology. |
| Systemic context | fever or other documented systemic symptoms | supports a request for in-person assessment when clinically relevant. |

Genetic or genomic data are **not** part of v1. They are especially sensitive,
are not generally present in the current image sources, and risk becoming a
shortcut. They require a separate dataset, explicit consent/licensing,
indication-specific justification, and fairness review before being considered.

## 5. Data contract for an interactive case

The model must never receive a synthetic patient answer represented as a real
clinical history. A future `adaptive_context` row should instead store only
authorised source fields and their provenance:

```json
{
  "sample_id": "...",
  "leakage_group_id": "...",
  "initial_image_asset_id": "...",
  "initial_context": {},
  "hidden_context": {
    "duration_and_recent_change": {
      "answer": "...",
      "provenance": "source_clinical_metadata"
    }
  },
  "next_action": "REQUEST_CLINICAL_CONTEXT",
  "question_id": "duration_and_recent_change",
  "reason_code": "differential_uncertainty",
  "gold_diagnosis": "..."
}
```

The patient-facing answer is exposed only after a valid request for its closed
`question_id`. The original text, provenance, licence, source availability and
missingness must remain auditable. A teacher may help choose or phrase a
question, but must not invent the patient answer.

## 6. Action and question policy

The next action is a structured target rather than free prose:

```json
{
  "next_action": "REQUEST_CLINICAL_CONTEXT",
  "question_id": "duration_and_recent_change",
  "reason_code": "differential_uncertainty"
}
```

The initial closed action set is:

```text
DIAGNOSE_PROVISIONALLY
REQUEST_CLINICAL_CONTEXT
REQUEST_OVERVIEW_IMAGE
REQUEST_CLOSEUP_IMAGE
REQUEST_SCALE_OR_PROFILE
REQUEST_DERMOSCOPY
REQUEST_IN_PERSON_EXAM
ABSTAIN_POOR_QUALITY
ABSTAIN_OUT_OF_DOMAIN
```

`question_id` must be selected from the frozen context ontology. This makes
tool selection, argument validity, premature diagnosis, and question utility
measurable. Natural-language wording can be rendered deterministically after a
valid `question_id` is selected.

## 7. Training sequence and controls

This is a future `D4_adaptive_context` / agentic extension, not the first E2
run:

1. Train and evaluate E2 structured supervision.
2. Train and evaluate E3 hard distillation.
3. Freeze the selected student checkpoint and all model-level benchmarks.
4. Build an authorised, group-safe interactive context cohort from training
   sources only.
5. Train the action/question policy on `sft_train`; retain `sft_dev` for model
   selection; never use ISEPDermaBench or DermoBench for policy construction.
6. Compare a prompt-only baseline with supervised action selection under the
   same student, context ontology, answer simulator, maximum turns, decoding
   settings, and task IDs.

The comparison must include the 4B base model, the selected specialised 4B
checkpoint, and a relevant larger baseline whenever they share the same
interactive protocol. Quality and efficiency are jointly reported.

## 8. Evaluation protocol

For every starting image, evaluate diagnosis after zero, one, two, and a capped
number of context turns. Report at least:

| Outcome | Definition |
|---|---|
| Diagnostic quality by turn | Top-1, Top-k, macro-F1 and balanced accuracy after each turn count. |
| Action accuracy | Correct `DIAGNOSE`, `ASK`, image request, escalation, or abstention decision. |
| Question selection | Exact and Top-k `question_id` accuracy against an expert/source-derived target. |
| Executable request rate | Fraction of requests that use a valid closed action and argument. |
| Premature-diagnosis rate | Diagnoses emitted where the reference policy requires more information or abstention. |
| Appropriate-abstention rate | Correct abstentions for poor quality or out-of-domain cases. |
| Question efficiency | Mean turns, questions per successful diagnosis, and diagnostic gain per question. |
| Safety and grounding | Unsupported-claim, invalid-action and unrecoverable-loop rates. |
| Efficiency | Tokens, p50/p95 latency, cost and, for local models, VRAM, GPU-seconds and Wh per successful task. |

Use a fixed patient/context simulator, fixed distractor fields, matching task IDs,
and leakage-group bootstrap confidence intervals. Do not report a model that
asks more questions as better merely because it eventually sees more context;
turn caps and efficiency denominators are part of the protocol.

## 9. Decision gates

1. Audit which real and authorised context fields exist per source, including
   completeness and possible label leakage.
2. Define the minimal closed context ontology and actions.
3. Create a small, source-provenance pilot and validate parser/simulator logic.
4. Establish the prompt-only baseline.
5. Train the supervised policy only after E2/E3 and student selection are
   frozen.
6. Run the controlled comparison and tool-space scaling experiment described in
   [the efficiency and agentic benchmark strategy](../notes/18_efficiency_and_agentic_benchmark_strategy.md).

## 10. References

1. Pacheco AGC, et al. *PAD-UFES-20: A skin lesion dataset composed of patient
   data and clinical images collected from smartphones.* Data in Brief, 2020.
   https://doi.org/10.1016/j.dib.2020.106221
2. Khurshid M, Vatsa M, Singh R. *Optimizing Skin Lesion Classification via
   Multimodal Data and Auxiliary Task Integration.* 2024.
   https://arxiv.org/abs/2402.10454
3. Li SS, et al. *MediQ: Question-Asking LLMs and a Benchmark for Reliable
   Interactive Clinical Reasoning.* 2024. https://arxiv.org/abs/2406.00922
4. Wong HM, et al. *MedClarify: An information-seeking AI agent for medical
   diagnosis with case-specific follow-up questions.* 2026 preprint.
   https://arxiv.org/abs/2602.17308

## 11. Limitations

- Context can improve an internal dataset score through non-causal shortcuts;
  source, demographics and prevalence must be audited before interpreting gains.
- The model is not a diagnostic device and should not be trained to claim that
  unobserved history or examination findings are visible in an image.
- Interactive patient simulation is an approximation to real consultation.
- Genetic data and unrestricted free-text patient generation are outside this
  protocol.
