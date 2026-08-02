"""Contracts for the fixed nested Validation teacher-screening cohorts."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "data/benchmarks/ISEPDermaBench"
COHORT_ROOT = RELEASE_ROOT / "metadata/validation_screening_v1"


def _task_ids(filename: str) -> list[str]:
    return [
        line.strip()
        for line in (COHORT_ROOT / filename)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _validation_frame(kind: str, task: str) -> pd.DataFrame:
    paths = sorted(
        (RELEASE_ROOT / kind / task).glob("validation-*.parquet")
    )
    return pd.concat(
        [pq.read_table(path).to_pandas() for path in paths],
        ignore_index=True,
    )


class ValidationScreeningSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (COHORT_ROOT / "manifest.json").read_text(encoding="utf-8")
        )

    def test_initial_cohorts_have_one_hundred_selection_units(self) -> None:
        for task, entry in self.manifest["tasks"].items():
            initial = next(
                cohort
                for cohort in entry["cohorts"]
                if cohort["is_initial_screen"]
            )
            task_ids = _task_ids(initial["task_ids_file"])
            self.assertEqual(initial["unit_count"], 100, task)
            self.assertEqual(len(task_ids), initial["task_count"], task)
            expected_tasks = 200 if task == "visual_confusion_sets" else 100
            self.assertEqual(len(task_ids), expected_tasks, task)

    def test_expansions_are_exact_nested_prefixes(self) -> None:
        for task, entry in self.manifest["tasks"].items():
            cohorts = entry["cohorts"]
            if len(cohorts) < 2:
                continue
            first = _task_ids(cohorts[0]["task_ids_file"])
            expanded = _task_ids(cohorts[1]["task_ids_file"])
            self.assertEqual(expanded[: len(first)], first, task)

    def test_every_task_id_exists_in_its_validation_split(self) -> None:
        for task, entry in self.manifest["tasks"].items():
            available = set(
                _validation_frame("tasks", task)["task_id"].astype(str)
            )
            for cohort in entry["cohorts"]:
                selected = _task_ids(cohort["task_ids_file"])
                self.assertTrue(set(selected).issubset(available), task)
                self.assertEqual(len(selected), len(set(selected)), task)

    def test_confusion_cohort_preserves_complete_pairs(self) -> None:
        selected = set(
            _task_ids("visual_confusion_sets_100_pairs.task_ids.txt")
        )
        frame = _validation_frame("tasks", "visual_confusion_sets")
        frame = frame[frame["task_id"].astype(str).isin(selected)]
        self.assertEqual(frame["pair_id"].nunique(), 100)
        self.assertTrue((frame.groupby("pair_id").size() == 2).all())
        conditions = frame.groupby("pair_id")["condition"].agg(set)
        self.assertTrue(
            conditions.map(
                lambda value: value
                == {"low_confusability", "high_confusability"}
            ).all()
        )

    def test_visual_top_k_initial_cohort_covers_all_validation_classes(
        self,
    ) -> None:
        selected = set(_task_ids("visual_top_k_100_cases.task_ids.txt"))
        references = _validation_frame("references", "visual_top_k")
        selected_references = references[
            references["task_id"].astype(str).isin(selected)
        ]
        self.assertEqual(
            set(selected_references["reference_disease_id"]),
            set(references["reference_disease_id"]),
        )


if __name__ == "__main__":
    unittest.main()
