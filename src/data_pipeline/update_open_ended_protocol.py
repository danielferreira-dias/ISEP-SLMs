"""Apply a prompt-only open-ended protocol update to ISEPDermaBench."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from datasets import Dataset
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.open_ended_benchmark import TASK_FEATURES


RELEASE = Path("data/benchmarks/ISEPDermaBench")
RESOURCES = Path("src/benchmark/resources/open_ended_diagnosis")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def update_open_ended_protocol(root: Path) -> dict[str, object]:
    root = root.resolve()
    release_root = root / RELEASE
    resources = root / RESOURCES
    model_prompt = yaml.safe_load(
        (resources / "model_prompt.yaml").read_text(encoding="utf-8")
    )
    system_prompt = str(model_prompt["system_prompt"])
    user_prompt = str(model_prompt["user_template"])
    prompt_hash = _text_sha256(system_prompt + "\0" + user_prompt)

    updated_shards: dict[str, dict[str, object]] = {}
    for split in ("validation", "internal_benchmark"):
        paths = sorted(
            (release_root / "tasks/open_ended_diagnosis").glob(
                f"{split}-*.parquet"
            )
        )
        if len(paths) != 1:
            raise ValueError(f"Expected one open-ended {split} task shard")
        path = paths[0]
        records = pq.read_table(path).to_pylist()
        for record in records:
            record["system_prompt"] = system_prompt
            record["user_prompt"] = user_prompt
            record["prompt_id"] = str(model_prompt["id"])
            record["prompt_version"] = str(model_prompt["version"])
            record["prompt_sha256"] = prompt_hash
        temporary = path.with_suffix(".parquet.tmp")
        Dataset.from_list(records, features=TASK_FEATURES).to_parquet(temporary)
        os.replace(temporary, path)
        updated_shards[split] = {
            "path": path.relative_to(release_root).as_posix(),
            "rows": len(records),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    artifacts = {
        "artifacts/prompts/open_ended_diagnosis.yaml": "model_prompt.yaml",
        "artifacts/judges/open_ended_diagnosis_judge.yaml": "judge_prompt.yaml",
    }
    for destination, source in artifacts.items():
        shutil.copy2(resources / source, release_root / destination)

    release_path = release_root / "release.json"
    document = json.loads(release_path.read_text(encoding="utf-8"))
    release = document["release"]
    release["version"] = "1.5.0"
    for split in release["splits"]:
        if split["benchmark"] == "open_ended_diagnosis":
            split["task_shards"] = [updated_shards[split["split"]]]
    artifact_by_path = {item["path"]: item for item in release["artifacts"]}
    for destination in artifacts:
        artifact_by_path[destination]["sha256"] = _file_sha256(
            release_root / destination
        )
    release["open_ended_protocol_update"] = {
        "policy": "example_free_natural_ranked_prose_v1",
        "status": "frozen",
        "frozen_on": "2026-08-02",
        "model_prompt_version": str(model_prompt["version"]),
        "judge_prompt_version": str(
            yaml.safe_load(
                (resources / "judge_prompt.yaml").read_text(encoding="utf-8")
            )["version"]
        ),
        "model_prompt_sha256": _file_sha256(resources / "model_prompt.yaml"),
        "judge_prompt_sha256": _file_sha256(resources / "judge_prompt.yaml"),
        "acceptance_run": {
            "model_id": "gpt_5_6_luna",
            "split": "validation",
            "sample_size": 50,
            "seed": 42,
            "valid_response_rate": 0.94,
            "judge_coverage": 1.0,
            "safety_refusal_count": 3,
            "top_1_accuracy": 0.32,
            "top_3_accuracy": 0.44,
            "unsupported_claim_rate": 0.16,
            "mean_evidence_grounding": 3.42,
            "ab_winner": "model_prompt_v1.1.0",
            "ab_comparator": "model_prompt_v1.2.1",
        },
    }
    release_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_path = release_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    marker = "## Input schema\n"
    note = (
        "Release 1.5.0 freezes model prompt 1.1.0 after a paired 50-case A/B "
        "test against the more prescriptive prompt 1.2.1. The selected prompt "
        "retains natural clinical prose, explicit Top-3 ordering, visible-"
        "evidence grounding, and no prose example. Judge prompt 1.2.0 and its "
        "four-verdict rubric remain unchanged."
    )
    previous = (
        "Release 1.4.0 freezes model prompt 1.2.0 after a final acceptance "
        "run. It requires evidence-constrained visible findings and an "
        "explicitly labelled prose Top-3, distinguishes unassessable from "
        "clinically absent findings, and prohibits non-visual patient facts. "
        "Judge prompt 1.2.0 and its four-verdict rubric remain unchanged."
    )
    if previous in readme:
        readme = readme.replace(previous, note, 1)
    elif note not in readme:
        if marker not in readme:
            raise ValueError("ISEPDermaBench README input-schema marker is missing")
        readme = readme.replace(
            marker,
            "## Open-ended prompt protocol\n\n" + note + "\n\n" + marker,
            1,
        )
    readme_path.write_text(readme, encoding="utf-8")
    return release["open_ended_protocol_update"]


def main() -> None:
    result = update_open_ended_protocol(Path.cwd())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
