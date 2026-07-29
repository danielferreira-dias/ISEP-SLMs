"""Unit tests for leakage-safe benchmark split generation."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_pipeline.splitting import (
    assign_groups,
    select_fixed_case_benchmark,
)


class GroupSplitTests(unittest.TestCase):
    def test_complete_leakage_groups_are_assigned_once(self) -> None:
        frame = _split_frame()
        assignments = assign_groups(
            frame,
            ratios={
                "train": 0.6,
                "validation": 0.2,
                "internal_test": 0.2,
            },
            seed=42,
        )

        self.assertEqual(set(assignments), set(frame["leakage_group_id"]))
        frame["split"] = frame["leakage_group_id"].map(assignments)
        self.assertTrue(
            frame.groupby("leakage_group_id")["split"].nunique().eq(1).all()
        )
        self.assertEqual(
            set(frame["split"]),
            {"train", "validation", "internal_test"},
        )

    def test_assignment_is_reproducible(self) -> None:
        frame = _split_frame()
        ratios = {
            "train": 0.6,
            "validation": 0.2,
            "internal_test": 0.2,
        }
        first = assign_groups(frame, ratios=ratios, seed=42)
        second = assign_groups(
            frame.sample(frac=1.0, random_state=7),
            ratios=ratios,
            seed=42,
        )
        self.assertEqual(first, second)

    def test_fixed_case_benchmark_uses_one_image_per_group(self) -> None:
        frame = _split_frame()
        benchmark, reserve, assignments = select_fixed_case_benchmark(
            frame,
            sample_count=20,
            seed=1042,
            secondary_feature_weight=0.25,
        )

        self.assertEqual(len(benchmark), 20)
        self.assertEqual(benchmark["leakage_group_id"].nunique(), 20)
        self.assertEqual(len(reserve), 10)
        self.assertEqual(reserve["leakage_group_id"].nunique(), 10)
        self.assertFalse(
            set(benchmark["leakage_group_id"])
            & set(reserve["leakage_group_id"])
        )
        self.assertEqual(len(assignments), 30)


def _split_frame() -> pd.DataFrame:
    records = []
    diseases = ["D001", "D004", "D014"]
    datasets = ["fitzpatrick17k_c", "pad_ufes_20", "scin"]
    for index in range(30):
        records.append(
            {
                "sample_id": f"SAMPLE_{index}",
                "leakage_group_id": f"GROUP_{index}",
                "disease_id": diseases[index % len(diseases)],
                "dataset_id": datasets[index % len(datasets)],
                "diagnosis_basis": "pathology",
                "age_group_standardized": f"{index % 8}0_to_{index % 8}9",
                "race_ethnicity": None,
                "skin_tone_system": "fitzpatrick",
                "skin_tone": str(index % 6 + 1),
                "sex_or_gender_system": "source_reported",
                "sex_or_gender": "female" if index % 2 else "male",
            }
        )
    records.append(
        {
            **records[0],
            "sample_id": "SAMPLE_0_SECOND_IMAGE",
        }
    )
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    unittest.main()
