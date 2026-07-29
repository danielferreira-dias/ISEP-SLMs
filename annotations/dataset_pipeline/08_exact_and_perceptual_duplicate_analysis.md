# Exact and perceptual duplicate analysis

## Objective

This stage implements reproducible image-content deduplication before any
train, validation, internal-test, or external-test split is created.

The objectives are:

1. calculate content fingerprints for every normalized image;
2. remove byte-identical redundancies conservatively;
3. identify visually equivalent images stored with different encodings;
4. detect cross-dataset overlap;
5. combine patient or case grouping with duplicate relationships into one
   leakage-safe split key;
6. recalculate disease and demographic support after exact exclusions and
   conservative perceptual grouping.

## Image resolution

The hashing implementation resolves all normalized image URI formats:

- direct repository-relative image paths;
- `zip://archive::member` locators;
- `parquet://shard::row=N::column=image_N_path` locators used by SCIN.

ZIP files are kept open during each run, and a bounded cache retains the active
embedded-image Parquet columns. This avoids extracting raw archives or
duplicating image payloads on disk.

All 46,023 normalized image rows were decoded successfully.

## Exact fingerprint

`image_sha256` is calculated over the encoded image bytes. Two rows have an
exact relationship only when their complete encoded byte sequence is
identical.

For a same-label exact group:

1. one deterministic canonical row is retained;
2. eligible redundant rows receive
   `deduplication_status: redundant_exact`;
3. redundant rows receive `include: false` and
   `exclusion_reason: exact_duplicate_redundant`.

Canonical selection prefers:

1. a currently eligible row;
2. a taxonomy contributor over an external or excluded dataset;
3. stronger diagnostic evidence;
4. the lexically first stable sample ID as a deterministic tie-breaker.

When identical bytes have more than one normalized disease ID, every eligible
row in the exact group is excluded with
`exact_duplicate_label_conflict`. This prevents an arbitrary label from being
selected as truth.

## Perceptual fingerprint

The perceptual algorithm is recorded as:

`phash_dct_32x32_low8x8_median_v1`

The procedure is:

1. apply EXIF orientation;
2. convert to grayscale;
3. resize to 32 by 32 pixels using Lanczos resampling;
4. calculate a two-dimensional type-II discrete cosine transform;
5. retain the 8 by 8 low-frequency block;
6. compare coefficients with the median to produce a 64-bit hash.

Candidate hashes are searched with a BK-tree using a maximum Hamming distance
of four bits. This avoids a quadratic comparison over 46,023 images.

Perceptual similarity is candidate evidence, not automatic confirmation.
Perceptual rows remain eligible but receive a shared duplicate and leakage
group. Their pair and component records are marked for review.

The observed perceptual distances were:

| Hamming distance | Candidate pairs |
| ---: | ---: |
| 0 | 552 |
| 2 | 450 |
| 4 | 249 |
| **Total** | **1,251** |

## Source-lineage evidence

The implementation can also compare stable `source_url` and
`upstream_image_uri` values preserved in `source_metadata`. No repeated
lineage key was found across the six current manifests. This does not prove
independent provenance because some datasets do not publish an original image
URL.

## Leakage-safe grouping

Manifest schema version 1.3.0 adds:

- `duplicate_group_id`;
- `duplicate_match_type`;
- `deduplication_status`;
- `leakage_group_id`.

Duplicate components are created from exact, perceptual, and source-lineage
edges. A second connected-component pass combines those edges with the
original `group_id`. Consequently, if one image from a patient matches an
image in another case or dataset, every image from both original groups
receives the same `leakage_group_id`.

Future split generation must use `leakage_group_id`. The original `group_id`
is retained for patient, case, lesion, or encounter auditability.

Coverage reports now count unique leakage groups. Exact exclusions are also
removed from benchmark image counts.

## Global results

The analysis produced:

| Result | Count |
| --- | ---: |
| Hashed images | 46,023 |
| Exact pair relations | 1,427 |
| Perceptual pair relations | 1,251 |
| All pair relations | 2,678 |
| Connected duplicate components | 2,099 |
| Samples in duplicate components | 4,450 |
| Cross-dataset components | 688 |
| Samples in cross-dataset components | 1,495 |
| Components requiring review | 1,020 |
| Pair relations requiring review | 1,263 |
| Newly excluded eligible rows | 34 |

The duplicate components comprise 1,091 exact-only components, 897
perceptual-only components, and 111 mixed components.

## Dataset results

| Dataset | Samples | Source groups | Leakage groups | Samples in duplicate components | Newly excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fitzpatrick17k-C | 11,394 | 11,394 | 11,367 | 732 | 0 |
| PAD-UFES-20 | 2,298 | 1,373 | 1,371 | 36 | 27 |
| SCIN | 10,406 | 5,033 | 5,010 | 73 | 7 |
| SkinDisNet | 1,710 | 416 | 414 | 71 | 0 |
| DDI | 656 | 656 | 652 | 8 | 0 |
| Dermnet Kaggle | 19,559 | 19,559 | 18,025 | 3,530 | 0 |

Dermnet was already excluded from benchmark use, so its 1,257 redundant exact
rows do not count as newly excluded eligible rows.

## Important findings

### SCIN

SCIN contains 15 canonical exact images and 27 redundant occurrences. This
reproduces the source documentation that 15 duplicated images appear 42 times
in total.

Only seven redundant SCIN rows were newly excluded because the other repeated
rows were already ineligible due to diagnosis gradability or taxonomy scope.

### PAD-UFES-20

Twelve exact image pairs, representing 24 rows, have conflicting normalized
disease IDs. All 24 rows were excluded. Three additional same-label exact
rows were excluded as redundant, producing 27 new exclusions.

The conflicts are retained in the duplicate reports for source-data review.

### Cross-dataset overlap

No byte-identical cross-dataset pair was observed. Perceptual comparison found:

| Dataset pair | Candidate pairs |
| --- | ---: |
| Fitzpatrick17k-C and Dermnet Kaggle | 800 |
| PAD-UFES-20 and Dermnet Kaggle | 5 |
| Fitzpatrick17k-C and SkinDisNet | 1 |
| SCIN and SkinDisNet | 1 |

The strong Fitzpatrick17k-C/Dermnet overlap is consistent with both datasets
containing web-atlas material, but every candidate still requires review.
Dermnet remains excluded regardless of these matches.

## Coverage impact

All twenty-one active diseases continue to pass the preliminary eligibility
thresholds after exact exclusion and leakage-safe grouping.

The largest unique-group reduction is for folliculitis, from 324 to 312
contributor groups. Other affected classes remain above their minimum support.
The lowest supported active class remains prurigo nodularis with 103 groups.

`drug_eruption` is unchanged at 215 groups and 286 images.

SkinDisNet retains 1,365 benchmark-mapped images. Its mapped cases now form
331 leakage-safe groups instead of 333 source patient groups because two
perceptual candidate relationships connect otherwise separate groups.

## Generated reports

- `data/reports/duplicate_pairs_v3.csv`
- `data/reports/duplicate_groups_v3.csv`
- `data/reports/duplicate_summary_v3.csv`
- updated version 3 source, disease, demographic, and subgroup reports

`duplicate_pairs_v3.csv` is the primary review queue. It records match
evidence, Hamming distance, source datasets and groups, normalized labels,
inclusion decisions, cross-dataset status, and whether review is required.

## Validation

Nineteen unit tests pass. They include:

- perceptual stability across PNG and JPEG encodings of the same pixels;
- deterministic canonical selection for exact duplicates;
- exclusion of exact duplicates with conflicting disease IDs;
- taxonomy and output-schema synchronization;
- SkinDisNet external-only role enforcement.

The manifest validator additionally checks:

- complete SHA-256 and perceptual-hash coverage;
- hash formats and algorithm provenance;
- manifest schema version 1.3.0;
- non-null leakage groups;
- consistency between source groups, duplicate groups, and leakage groups;
- exclusion of exact redundant and exact-conflict rows.

All six manifests pass validation with 46,023 globally unique sample IDs.

## Limitations and next gate

A pHash collision or visually similar clinical composition can create a false
candidate. Conversely, a large crop, rotation, annotation overlay, or severe
colour transformation may escape a 64-bit pHash threshold of four.

Before split generation:

1. review all label-conflicting perceptual components;
2. review all cross-dataset perceptual components;
3. sample within-dataset candidates at each Hamming distance;
4. approve or reject candidate edges;
5. rebuild final duplicate and leakage groups;
6. freeze the taxonomy and generate the final manifests.
