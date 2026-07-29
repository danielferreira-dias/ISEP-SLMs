# SCIN

## Local status

`data/` contains the complete official `google/scin` Hugging Face snapshot at
revision `996257142f7517fb8991a28cfba46ec4e3f530a9`.

The snapshot contains 26 Parquet shards with embedded images, approximately
12.6 GB in total. It is a case-level dataset: do not split individual images
from the same case across train and evaluation sets.

Sources:

- <https://github.com/google-research-datasets/scin>
- <https://huggingface.co/datasets/google/scin>
- <https://doi.org/10.1001/jamanetworkopen.2024.46615>

## Dataset

SCIN contains 5,033 volunteer-contributed cases and 10,408 images of common
skin conditions. Each case may contain up to three images: close-up, angled,
and distance views. Contributors could provide age group, sex at birth,
self-reported Fitzpatrick type, race/ethnicity, texture, body location,
symptoms, related category, and duration.

Dermatologists retrospectively reviewed the submitted images. Only a subset of
cases has a usable condition differential, and these labels are not confirmed
in-person diagnoses.

## Labels

The primary diagnostic supervision is:

- `dermatologist_skin_condition_on_label_name`: dermatologist condition names.
- `dermatologist_skin_condition_confidence`: corresponding confidence values
  from 1 to 5.
- `weighted_skin_condition_label`: normalized condition-to-weight mapping
  aggregated across reviewers.

An illustrative target has the form:

```text
{
  "Eczema": 0.69,
  "Post-Inflammatory hyperpigmentation": 0.11,
  "Psoriasis": 0.08
}
```

Additional targets include dermatologist-estimated Fitzpatrick type and
trained-layperson Monk Skin Tone values. User-provided attributes include
raised/bumpy, flat, rough/flaky, fluid-filled, itching, bleeding, pain,
burning, increasing size, and affected body-part indicators.

Treat `weighted_skin_condition_label` as a ranked differential, not a certain
single-class ground truth.

## Licence and known issues

SCIN uses the custom SCIN Data Use License included in the official source. It
allows research use and adapted material with attribution and prohibits
re-identification.

The authors report 15 duplicate images appearing 42 times, 48 gradable cases
without a condition label, and one missing image. Account for these before
training.

