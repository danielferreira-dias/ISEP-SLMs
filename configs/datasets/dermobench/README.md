# DermoBench

## Local status

DermoBench is stored locally at `data/benchmarks/dermobench/release/`. This
release contains its JSON/JSONL task files and
`dermobench_release_imgs.zip` (about 3.2 GB), whose `imgs/` directory includes
the image payload referenced by the task records. Do not commit the clone or
the archive.

```bash
git clone https://huggingface.co/datasets/mendicant04/DermoBench
```

For a new clone, authenticate with the Hugging Face account approved for this
gated dataset before cloning. Do not commit downloaded data.

## Image access and resolution

Each record has a relative `image` path. Resolve it inside
`dermobench_release_imgs.zip` as `imgs/<image path>`; keep a path-resolution
audit and verify all referenced records before executing a task. The archive
contains source-labelled folders including DDI, Derm1M, Derm7pt, DermNet,
Fitzpatrick17k, ISIC, PAD-UFES-20, SCIN, SKINCON and SNU134.

The archive's presence does not replace upstream attribution or licensing
obligations. DermoBench remains a derived evaluation suite and is not an
independent image source.

## Evaluation policy

`config.yaml` is the authoritative task inventory. Tasks 1.1, 1.2, 3.1, and
3.2 use LLM-as-a-judge in the upstream evaluation; all listed MCQ tasks use
deterministic exact-choice scoring. Report judge-based results separately,
with the judge model, prompt, temperature, and any adjudication procedure.

Source: <https://huggingface.co/datasets/mendicant04/DermoBench>.
