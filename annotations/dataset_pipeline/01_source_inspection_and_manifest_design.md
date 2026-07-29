# Source inspection and manifest design

## Objective

The first stage established a common representation for heterogeneous
dermatology datasets before combining them. Direct concatenation was rejected
because each source uses different identifiers, grouping units, image storage
formats, label semantics, and evidence standards.

## Sources inspected

The first implementation covers four datasets:

| Dataset | Native row unit | Image storage | Diagnostic target | Strongest grouping key |
| --- | --- | --- | --- | --- |
| Fitzpatrick17k-C | Image | ZIP archive | Web-atlas disease label | Image identifier |
| PAD-UFES-20 | Image | Three ZIP archives | Six-class diagnostic code | Patient identifier |
| SCIN | Case | Embedded in 26 Parquet shards | Ranked dermatologist differential | Case identifier |
| DDI | Image | Image directory | Pathology-grounded disease label | Image identifier in the released metadata |

SkinCAP, SkinCaRe, SKINCON, and DermaVQA were not treated as independent
disease-classification sources. SkinCAP and part of SkinCaRe duplicate
Fitzpatrick17k/DDI lineage, SKINCON is an annotation overlay, and DermaVQA uses
free-text visual-question-answering targets.

## Shared row unit

The normalized manifest uses one row per image. Multi-image SCIN cases are
expanded into multiple rows that retain the same `group_id`. This permits
image-level inference while preserving case-level split integrity.

The common contract is defined in
`configs/datasets/manifest_schema.yaml`. It includes:

- stable global and source identifiers;
- a leakage-safe grouping identifier;
- an image locator;
- the original and normalized primary disease;
- all available ranked reference diagnoses;
- diagnosis evidence and gradability;
- skin-tone metadata when available;
- licence and source provenance;
- split and inclusion state;
- JSON-encoded source-specific metadata.

## Image locators

The normalization stage intentionally avoids extracting or duplicating image
bytes:

- Direct DDI files use repository-relative paths.
- Fitzpatrick17k-C and PAD-UFES-20 use `zip://` locators containing the archive
  path and member path.
- SCIN uses `parquet://` locators containing the source shard, row number, and
  embedded-image column.

An image resolver will be required before model inference. Keeping storage
references in the manifest separates data indexing from image materialization.

## Dataset roles

The selected roles are:

- Fitzpatrick17k-C, PAD-UFES-20, and SCIN are taxonomy contributors.
- DDI is reserved for external evaluation.
- Derived or task-incompatible datasets are excluded from disease-frequency
  calculations.

DDI remains independent of taxonomy selection and model development. This
preserves its value as a domain-shift evaluation source.
