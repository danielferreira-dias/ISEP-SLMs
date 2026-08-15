# Title

This repository is about the development of ISEP's thesis about Small Language Models.

## Folder Structure

- @doc/ is the folder that contains the dissertation which includes different chapters
- @src/ is the source folder that consists of coding

## Permanent Thesis Comparison Principle

The central thesis hypothesis is that a smaller multimodal model, after
domain-specific training, can outperform larger generalist models on a
controlled dermatology domain while requiring less memory, latency, and cost.

All future benchmark analyses, reports, figures, and dissertation tables must:

- place the specialized small-model checkpoints directly alongside the larger
  comparison models whenever they share a task and evaluation protocol;
- include the small model before training, the selected specialized checkpoint,
  and the relevant larger-model baselines so that the specialization gain and
  size comparison are both visible;
- report quality and efficiency together, including parameter count and, when
  available, VRAM, latency, throughput, GPU-hours, and monetary cost;
- mark non-executed or non-comparable tasks with a dash and explain why, rather
  than silently omitting the specialized model from the table;
- distinguish strictly paired comparisons from contextual comparisons that use
  different datasets, prompts, releases, inference profiles, or judging
  protocols;
- avoid the universal claim that small models are always better: conclusions
  must remain scoped to the evaluated domain, task, data, and protocol.

## Permanent Efficiency and Agentic Benchmark Principle

Accuracy alone is insufficient for the thesis claim. Every future controlled
evaluation of local specialized and larger models should, whenever technically
available, collect inference efficiency on the same hardware and runtime:

- time to first token, end-to-end latency p50/p95/p99, tokens per second,
  requests per second, GPU-seconds per request, and peak GPU memory;
- sampled GPU power integrated over inference, Wh per request, and Wh per
  correct answer;
- quality-versus-latency, quality-versus-memory, quality-versus-energy, and
  quality-versus-cost Pareto-frontier plots.

API models may be compared on observed latency, token use, and billed cost, but
must have energy marked unavailable unless the provider exposes a verifiable
measurement. Never infer provider energy from a local GPU proxy.

Agentic evaluation begins only after the model fine-tuning and distillation
stage is frozen. It must report task success, tool-selection and argument
accuracy, executable and invalid call rates, steps and tool calls per task,
loop and recovery rates, and tokens, latency, cost, and energy per successful
task. Tool-space scaling must be evaluated with progressively larger tool sets
(for example 5, 10, 25, 50, and 100 tools), holding tasks and tool distractor
sampling controlled across compared models.

## E1 to E2 Thesis Rationale: From Classification to Morphological Competence

The thesis must explain E2 as a deliberate change in the target of learning,
not as an arbitrary increase in dataset size. E1 establishes the baseline
question:

> Can a 4B multimodal model learn to map a dermatology image to one of the
> canonical diagnoses?

That result is necessary, but it is not sufficient evidence that the model has
learned clinically useful visual representations. A diagnosis-only target can
be reached through shortcuts, dataset regularities, or coarse image-label
associations, while providing no direct evidence that the model can identify
the lesion attributes a clinician uses to describe and differentiate it.

E2 therefore asks a stronger and separately testable question:

> Does joint supervision with diagnosis and human-defined morphology concepts
> improve diagnostic generalization and produce a model that can expose
> clinically meaningful visual findings?

This motivation is grounded in four complementary lines of evidence:

- Dermatological diagnosis is normally preceded by a structured description
  of primary morphology, secondary surface changes, colour, configuration,
  distribution, and other clinical signs. The International League of
  Dermatological Societies describes a standardized lesion vocabulary as a
  foundation for communication and clinical practice ([Nast et al., 2016](https://doi.org/10.1111/bjd.14419)); a clinical reference similarly states
  that the primary lesion morphology is the basis of diagnostic categories
  ([NCBI Clinical Methods](https://www.ncbi.nlm.nih.gov/sites/books/NBK206/)).
- SkinCon was created precisely to provide dermatologist-annotated,
  human-interpretable concepts across diseases. Its 48 concepts include
  terms such as plaque, scale, and erosion, and its authors demonstrate their
  use for probing, concept-based explanations, debugging, and concept
  bottlenecks ([Daneshjou et al., 2023](https://arxiv.org/abs/2302.00785)).
- Multitask learning is justified when related supervision signals provide an
  inductive bias through a shared representation; this is the central claim
  of Caruana's foundational formulation ([Caruana, 1997](https://doi.org/10.1023/A:1007379606734)). In dermatology, a related study explicitly
  treats lesion characterization as a useful target alongside disease
  classification and argues that describing lesion types can facilitate
  diagnosis ([Liao & Luo, 2018](https://arxiv.org/abs/1812.03520)).
- Concept-bottleneck work shows why concept targets are scientifically useful:
  they make intermediate clinical variables inspectable and potentially
  actionable, while also warning that concept supervision does not guarantee
  superior end-task accuracy ([Koh et al., 2020](https://proceedings.mlr.press/v119/koh20a.html)).

The thesis must present these sources as motivation and methodological
precedent, not as proof that E2 will win. The causal claim is empirical:

> Holding the official base checkpoint, image preprocessing, split, optimizer,
> LoRA topology, update budget, seed protocol, and evaluation procedure fixed,
> adding morphology supervision changes the learned representation and may
> improve diagnosis, morphology recognition, calibration, or error
> interpretability.

The E2 comparison must preserve this attribution:

| Condition | Image target | Purpose |
| --- | --- | --- |
| E1 diagnosis-only | canonical diagnosis | baseline specialization |
| E2 diagnosis + SKINCON | diagnosis plus 48 human-defined concepts | primary morphology ablation |
| E2 + SkinCAP | diagnosis, SKINCON, filtered caption | separate caption-language ablation |

The SkinCAP arm is not the primary proof of morphological understanding. Its
captions were written with access to diagnosis and are therefore labelled
`human_caption_gold_conditioned_filtered`; they must be reported as an
additional ablation, not as answer-blind visual ground truth. The primary E2
claim is supported by the human-defined SKINCON concept targets.

E2 must be evaluated on both outputs, not only on diagnosis:

- diagnosis: Top-1/Top-3, macro-F1, balanced accuracy, per-class recall,
  calibration, and invalid-output rate;
- morphology: exact-match validity, micro/macro F1, per-concept precision,
  recall and F1, Hamming loss, and confusion/error slices where applicable;
- transfer and robustness: fixed external datasets and benchmarks that were
  not used to train or select E2;
- efficiency: trainable parameters, training GPU-hours, peak VRAM, inference
  latency, throughput, and energy metrics under the permanent efficiency
  principle above.

The interpretation should distinguish three possible outcomes:

1. **Diagnostic gain:** E2 improves diagnosis and morphology, supporting the
   hypothesis that related clinical supervision helped the shared visual
   representation.
2. **Representation gain without diagnostic gain:** E2 improves morphology or
   calibration but not Top-1; this is still a meaningful result because it
   shows that classification accuracy alone concealed a capability change.
3. **Negative transfer:** E2 improves morphology but harms diagnosis, or both
   tasks degrade; this is evidence that task relatedness, loss weighting,
   coverage, or capacity was insufficient, not a failed thesis.

Suggested thesis wording:

> A high diagnostic accuracy is an important endpoint, but it does not by
> itself establish that a multimodal model has acquired the intermediate
> visual vocabulary used in dermatological reasoning. Dermatological
> assessment conventionally describes lesion morphology before integrating
> those observations into a differential diagnosis. We therefore introduce
> E2 as a controlled multitask extension of E1: the same 4B model is trained
> from the same official base checkpoint and split, while an additional
> SKINCON objective supervises human-defined morphological concepts. E2 tests
> whether this clinically motivated auxiliary signal improves diagnosis and
> makes visual findings measurable, rather than assuming that a better label
> score proves understanding.

Do not write “the model understands dermatology” from E2 accuracy alone. Use
“learned measurable morphology concepts”, “improved concept-level prediction”,
“supports a clinically motivated representation hypothesis”, or “showed
evidence consistent with improved visual grounding” only when the corresponding
metrics and controls support that statement. Broader VLM benchmarks such as
DermaBench are relevant because they explicitly evaluate diagnosis, anatomy,
morphology, distribution, surface, colour, image quality, and open-ended
descriptions rather than classification alone ([DermaBench, 2026](https://arxiv.org/abs/2601.14084)); they remain external evaluation evidence,
not training or checkpoint-selection data.
