# SkinCAP

## Local status

`data/` contains the gated Hugging Face snapshot for
`joshuachou/SkinCAP`, revision
`4119044b3e14085d7439f88016d93376d433da5f`.

Access was obtained through the Hugging Face account already authenticated on
this machine. Do not redistribute the payload or assume another researcher is
covered by that approval.

The verified snapshot contains 4,000 main PNG images, 4,000 rows in each of
the CSV and XLSX metadata versions, and 346 additional images under
`skincap/not_include/`. Exclude that last directory from training.

Source: <https://huggingface.co/datasets/joshuachou/SkinCAP>.

## Dataset

SkinCAP pairs approximately 4,000 Fitzpatrick17k/DDI clinical images with
dermatologist-authored Chinese captions and English translations. Captions
describe anatomical site, primary and secondary morphology, colour,
distribution, and surface changes. The associated work reports 178 disease
types. The local table contains exactly 4,000 rows: 3,345 sourced from
Fitzpatrick17k and 655 from DDI. It has 187 distinct non-empty `disease`
strings, so normalize aliases and spelling variants before treating them as
classification classes.

The main tabular fields are:

- `id`, `skincap_file_path`, `ori_file_path`
- `disease`
- Chinese and English caption fields. In `skincap_v240715.xlsx` these are
  `caption_zh` and `caption_en`; in the older `skincap_v240623.csv` the
  polished English text is named `caption_zh_polish_en`.
- `remark`, `source`
- DDI fields such as `skin_tone` and `malignant`
- Fitzpatrick fields such as `fitzpatrick_scale`,
  `fitzpatrick_centaur`, `nine_partition_label`, and
  `three_partition_label`
- SKINCON morphology-concept columns

## Training targets

Unlike an image-class dataset, the central target is natural language:

```text
image -> caption_en / caption_zh
caption -> disease
image + morphology concepts -> disease
```

`disease` remains the diagnostic class label. The caption fields are
descriptive targets and should not be treated as treatment advice.
Example disease values include `psoriasis`, `squamous cell carcinoma`,
`melanocytic-nevi`, `lupus erythematosus`, `lichen planus`, and
`basal cell carcinoma`.

## Licence warning

The Hub metadata, dataset card, gated access agreement, and upstream image
licences are not fully aligned. The access agreement contains
non-commercial, no-redistribution, and derivative-work restrictions, while
the source images retain Fitzpatrick17k/DDI terms. Obtain written permission
before releasing adapted data or trained weights.
