# DermoBench

## Local status

DermoBench access has been approved for this project. Download the gated
annotation suite into `data/`; it contains JSON/JSONL task files, not a
redistributable image archive.

```bash
hf download mendicant04/DermoBench --repo-type dataset \
  --local-dir configs/datasets/dermobench/data
```

The command uses the Hugging Face account already authenticated locally. If
needed, authenticate first with `hf auth login` and submit the dataset access
request while signed in. Do not commit downloaded data.

## Image access and resolution

Each record has a relative `image` path. The benchmark metadata does not
include the corresponding image files, so resolve every path to its upstream
provider and retain a local path-resolution audit. Existing local DDI images
can cover the DDI tasks (including Task 4 and the DDI subset of Task 2.1),
subject to a verified identifier/path match. SKINCON is an annotation overlay,
not an image source; its tasks need the underlying image datasets.

For Derm7pt, Derm1M-EDU, and SNU134, obtain the images directly from their
respective source providers under their own access terms. A successful
DermoBench request does not grant those image licences. Do not run a task
until all of its referenced images have been resolved and the matching rate is
recorded.

## Evaluation policy

`config.yaml` is the authoritative task inventory. Tasks 1.1, 1.2, 3.1, and
3.2 use LLM-as-a-judge in the upstream evaluation; all listed MCQ tasks use
deterministic exact-choice scoring. Report judge-based results separately,
with the judge model, prompt, temperature, and any adjudication procedure.

Source: <https://huggingface.co/datasets/mendicant04/DermoBench>.
