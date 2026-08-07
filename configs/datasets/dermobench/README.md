# DermoBench

## Local status

DermoBench is stored locally at `data/benchmarks/DermoBench/release/`. This
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

The immutable upstream files contain 31,999 tasks. Before thesis evaluation,
build the leakage-filtered view with:

```bash
python -m src.data_pipeline.dermobench_evaluation
```

This writes `data/benchmarks/DermoBench/evaluation/tasks/` and excludes every
task whose image or source leakage group appears in ISEPDermData Train. The
current filtered view contains 29,099 tasks after removing 2,900 task rows
covering 863 unique overlapping images. The generated `evaluation/release.json`
records every checksum and aggregate exclusion reason.

`config.yaml` is the authoritative filtered task inventory. Tasks 1.1, 1.2,
3.1, and 3.2 use the upstream text-only judge prompts but run the judge through
`configs/models/gemini_3_5_flash_lite_openrouter.yaml`. MCQ tasks use
deterministic exact-choice scoring. Because the paper used Gemini 2.5 Pro,
Flash-Lite judge scores form a new protocol and are not directly comparable to
the paper's open-ended scores.

## Runtime adapter

`src.benchmark.dermobench` executes every filtered task through the common
multimodal pipeline. Select a task with
`--benchmark dermobench/<task_id>`. It preserves upstream options and prompts,
uses deterministic exact-choice scoring for MCQ tasks, and stages open-ended
answers without assigning a content score prematurely.

`src.benchmark.dermobench_judge` prepares the four open-ended task families for
Gemini 3.5 Flash-Lite using OpenRouter's text-only asynchronous Batch API.
Task 1.1/1.2 keep the upstream three-voter default; Task 3.1/3.2 use one voter.
The submission, request-to-task index, completed response, individual votes,
aggregates, usage, cost, and final judge metrics remain beside the original
run.

Source: <https://huggingface.co/datasets/mendicant04/DermoBench>.
