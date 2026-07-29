# Paired visual disease confusion-set benchmark

## Objective

This stage adds a controlled benchmark that measures how disease-candidate
similarity affects visual classification. It complements the 21-class Visual
Top-K benchmark without replacing it.

The same image is evaluated under two three-candidate conditions. This paired
design controls for image content and isolates the effect of replacing
visually distant distractors with clinically motivated look-alike diseases.

## Source and scope

The source is the sealed
`internal_benchmark_1000.parquet`. No training, validation, reserve, or external
case is added. The selection uses one image per `leakage_group_id` and covers
15 active disease IDs.

Six active diseases remain outside version 1 because the current benchmark
does not support a non-overlapping, clinically coherent three-way set for
them: psoriasis, vitiligo, scabies, pityriasis rosea, prurigo nodularis, and
granuloma annulare.

## High-confusability sets

| Set | Disease IDs | Cases per disease |
| --- | --- | ---: |
| Melanocytic look-alike lesions | D001, D002, D006 | 30 |
| Keratinocytic lesions | D004, D005, D007 | 53 |
| Eczematous dermatitis | D009, D014, D015 | 17 |
| Acneiform and follicular disorders | D011, D012, D016 | 15 |
| Reactive eruptions | D017, D024, D025 | 23 |

Each set is downsampled deterministically to its rarest disease. Selection is
based on a seeded SHA-256 ordering of `sample_id`; no manual image-quality
selection is performed.

The resulting release contains:

- 414 unique images and leakage groups;
- 414 low/high pairs;
- 828 total tasks;
- five balanced high-confusability sets.

## Paired candidate construction

The high-confusability condition contains the three diseases declared for the
sample's set.

The low-confusability condition contains the same reference disease plus two
balanced distractors drawn from different construction-only appearance
partitions:

- lesion;
- inflammatory eruption;
- acneiform.

These partitions are not disease outputs, clinical routes, or information
shown to the model. They only prevent a low-confusability task from selecting
distractors from the same broad visual group. Candidate presentation order is
also randomized deterministically.

## Model contract

The model receives:

- one dermatological image;
- exactly three candidate disease IDs and English display names;
- an instruction to rank all candidates.

The response contains exactly three unique `{rank, disease_id}` objects with
consecutive ranks. The runtime schema narrows the disease enum to the current
task's candidates. Clinical context, explanations, confidence values, and
out-of-set diseases are forbidden.

## Evaluation

The primary fine-grained result is high-confusability Top-1 accuracy. The
benchmark additionally reports Top-2 accuracy, mean reciprocal rank,
class-balanced Top-1 F1, macro accuracy across sets, structural-output metrics,
and the paired confusability gap:

```text
low-confusability Top-1 accuracy
minus
high-confusability Top-1 accuracy
```

Low/high comparisons are paired by `pair_id`. Pre-training/post-training model
comparisons are paired by `task_id`. Confidence intervals use 10,000 paired
bootstrap resamples at the 95% level.

Top-3 accuracy is intentionally omitted because every valid response ranks all
three candidates and would therefore contain the reference by construction.

## Integrity and limitations

The release validator confirms source-subset membership, exact pair structure,
candidate uniqueness, reference inclusion, active disease IDs, within-set
balance, cross-partition low-confusability construction, checksums, and the
expected counts.

The confusion sets are clinically motivated but remain provisional pending
specialist review. The validation split may be used to audit whether the
intended confusions occur. Results from the sealed 1,000-case benchmark must
not be used to redefine candidate membership or tune the prompt.

Accuracy from a three-option task is not directly comparable to accuracy from
the 21-disease Visual Top-K task. The random Top-1 baseline for this benchmark
is 33.3%.

