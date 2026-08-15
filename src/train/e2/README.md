# E2 human visual supervision

E2 trains from the same pinned `Qwen/Qwen3.5-4B` base used by E1, with a new
LoRA adapter. It does not resume the E1 adapter and does not contain teacher
answers.

The frozen v0.3 baseline contributes two tasks:

- `diagnosis`: 6,312 train and 1,229 dev rows with the exact E1 21-class target;
- `morphology`: 3,068 train and 527 dev rows with all eligible human SKINCON
  annotations over the frozen 48-concept ontology.

The additive v0.4 ablation contributes a third task:

- `caption`: 2,767 train and 483 dev rows with a short filtered SkinCAP visual
  observation and no trainer-visible diagnosis or unfiltered source caption.

Every row is used exactly once per epoch. A deterministic proportional schedule
interleaves two or three tasks without oversampling any source. The trainer may
then shuffle those fixed indices with its declared seed.

Before importing Unsloth or reserving CUDA, the pipeline validates the local
release manifest, ontology, row counts, byte sizes, and SHA-256 of every Parquet
shard. At access time it verifies the embedded image hash, prompt hash, target
source, schema, split and quality status. It reconstructs the multimodal chat
with the actual decoded image because the Parquet `messages` column deliberately
stores only an image placeholder.

Checkpoint selection remains based on diagnosis macro-F1 on the frozen E1 dev
split, preserving comparability with E1. E2 additionally reports SKINCON exact
match, micro/macro F1, per-concept precision/recall/F1, Hamming loss, and invalid
JSON rate for the human morphology dev split.

SkinCAP is evaluated as constrained free-text generation, so it has no honest
notion of exact label accuracy. Its deterministic metrics are clinical-format
compliance, prohibited-content rate, concept precision/recall/F1,
unsupported-concept rate, ROUGE-L, token F1 and reference similarity. The
report keeps the three tasks separate:

- diagnosis: Top-1 accuracy, macro-F1 and balanced accuracy;
- SKINCON: exact match, micro/macro F1 and per-concept results;
- SkinCAP: caption task score and all of its compliance/content/similarity
  components.

For overview figures only, `global_multitask_score` is the unweighted mean of
diagnosis macro-F1, morphology macro-F1 and caption task score. It is labelled
`macro_task_score_not_accuracy`, is comparable only for the same task set and
does not select the checkpoint.

At the real multimodal collator boundary, every unique diagnosis, morphology
or caption row records its split/group identity, original/resized dimensions,
pixel count and exact visual/prompt/target token counts. This permits cost and
quality analyses by task and annotation availability without retaining extra
clinical images in reports.

Teacher-derived Stage-A/Stage-B targets and the twelve open-response templates
belong to `src/train/e3/` and are not runnable until a future accepted E3 release
is materialized.

## Conditional SkinCAP caption transform

`src/train/e2/skincap/` implements the versioned
`skincap_observation_prefix_v1` transform. It retains only the caption prefix
before the first gold-diagnosis, diagnostic, testing, or management boundary,
then rejects short targets and any residual unsafe language. The default CLI
is aggregate-only and never writes source or derived clinical text:

```bash
python -m src.train.e2.skincap.cli
```

The resulting JSON records counts, provenance hashes, boundary frequencies,
rejection reasons, and target-length statistics. Written permission was
attested on 15 August 2026. The private additive release
`isep_distill_dataset_v0.4.1` admits 2,767 train and 483 dev caption rows with
zero cross-task train/dev group overlap. v0.4.0 is withdrawn.
The source caption, diagnosis, and removed suffix are absent from the
trainer-visible schema.

The original two-task `e2_skincon_unsloth_all` condition remains frozen. The
new three-task condition is a separate ablation so that any change can be
attributed to SkinCAP caption supervision rather than silently changing E2.
