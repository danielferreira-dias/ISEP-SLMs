# DermoBench local release

This directory contains the DermoBench task annotations and the gated image
archive. It is an external benchmark, separate from the frozen
`ISEPDermaBench` protocols used by `src.benchmark.cli`.

## One-time local setup

The release is already present when the following exist:

- `release/task*/...json` and `.jsonl` annotation files;
- `release/dermobench_release_imgs.zip` (about 3.1 GB).

Create the local image tree and a reproducible path index with:

```bash
python -m src.data_pipeline.dermobench --extract
```

This extracts images to `release/images/` (about 3.1 GB more disk space) and
writes `release/image_index.json`. Both are ignored by Git. The operation can
be rerun safely after an interruption; files with the expected size are kept.

Run the same command without `--extract` to validate annotations and recreate
the index only:

```bash
python -m src.data_pipeline.dermobench
```

## Why the image index is required

The annotation field `image` is the canonical lookup key. Most paths match the
archive exactly; 3,396 Derm1M-EDU names differ only because the archive
normalizes punctuation and Unicode differently. `image_index.json` maps every
annotation path to the actual path below `release/images/`; consumers should
therefore resolve an annotation image as:

```python
from pathlib import Path
import json

release = Path("data/benchmarks/DermoBench/release")
index = json.loads((release / "image_index.json").read_text())
image_path = release / index["archive"]["image_root"] / index["image_paths"][row["image"]]
```

Do not rename the released annotations or images: that would make comparison
with the upstream task files non-reproducible.

## Scope and use

DermoBench contains 31,999 VQA tasks across morphology, diagnosis, reasoning,
and fairness. Its files use the conversation schema shown below; JSONL files
contain one object per line and JSON files contain an array.

```json
{
  "id": "sample_id",
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\\nQuestion"},
    {"from": "gpt", "value": "Reference answer"}
  ]
}
```

The current thesis CLI only executes the seven ISEPDermaBench task adapters.
To evaluate DermoBench through that CLI, the next implementation step is a
dedicated DermoBench adapter and task-specific scoring protocol. Until then,
use the extracted image path and original `conversations` data in a dedicated
evaluation script. DermoBench must be used for research evaluation only, not
for clinical deployment or patient-facing diagnosis.
