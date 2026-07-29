# DermaVQA

## Local status

`data/osf/` contains the public OSF release:

- `project/`: OSF metadata, README, and CC BY 4.0 licence.
- `iiyi/`: IIYI metadata, multilingual validation/test annotations, mappings,
  split definitions, and `images_final.zip`.
- `reddit/`: dermatologist answer annotations and the public download helper.

Source: <https://osf.io/72rp3/>. Code and baseline models:
<https://github.com/velvinnn/DermaVQA>.

The public OSF listing does not include Reddit images directly; its
`download_data.py` helper retrieves permitted source material according to
the release workflow.

## Dataset

DermaVQA is a multilingual visual-question-answering dataset built from:

- 998 IIYI forum encounters with 2,945 images and multiple
  responses per encounter;
- 490 Reddit dermatology-question encounters. Dermatologists were hired to
  create answer targets because public replies cannot be assumed to be
  clinician-authored.

Questions cover open diagnosis, binary diagnosis, differential or
multiple-choice diagnosis, treatment, general advice, and progression or
healing.

## Labels and schema

Targets are free-text responses rather than a fixed disease-class column.
Core fields include:

- `encounter_id`, `author_id`, `image_ids`
- `query_title_zh`, `query_content_zh`
- `query_title_en`, `query_content_en`
- optional Spanish translations
- `responses`, containing response author IDs and multilingual answer text

Reddit files use split-specific `*_answersonly.json` records. IIYI validation
and test responses also contain annotation-quality fields such as
`completeness` and `contain_freq_ans`.

For a text-only model, exclude questions whose answer depends on an unseen
image or pair them with a validated image description. Treatment/advice
responses require clinician auditing before use.

## Licence and privacy

The OSF release includes a CC BY 4.0 licence and metadata identifying the IIYI
and Reddit sources. Preserve attribution and source provenance. Because these
are consumer-generated medical questions, avoid re-identification and inspect
all derived exports for residual personal information.
