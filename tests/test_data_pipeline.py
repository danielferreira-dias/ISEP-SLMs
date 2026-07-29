"""Unit tests for disease normalization and coverage eligibility."""

from __future__ import annotations

from pathlib import Path
import json
import unittest

import pandas as pd
import yaml

from src.data_pipeline.adapters import (
    _fitzpatrick_value,
    _normalize_scin_race_ethnicity,
    _parse_scin_differential,
)
from src.data_pipeline.common import (
    DiseaseMapper,
    normalize_label,
    standardize_age_group,
)
from src.data_pipeline.reporting import (
    _informative_demographic_mask,
    _support_status,
)


ROOT = Path(__file__).resolve().parents[1]


class DiseaseMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapper = DiseaseMapper(
            ROOT / "configs/taxonomies/diseases.yaml",
            ROOT / "configs/taxonomies/source_disease_mappings.yaml",
        )

    def test_label_normalization(self) -> None:
        self.assertEqual(
            normalize_label("Seborrheic-Keratosis"),
            "seborrheic keratosis",
        )

    def test_dataset_specific_mapping(self) -> None:
        self.assertEqual(self.mapper.map("pad_ufes_20", "ACK"), "D007")

    def test_generic_eczema_maps_only_to_broad_class(self) -> None:
        self.assertEqual(self.mapper.map("scin", "Eczema"), "D014")

    def test_generic_tinea_remains_out_of_benchmark_scope(self) -> None:
        self.assertIsNone(self.mapper.map("scin", "Tinea"))

    def test_drug_rash_maps_to_drug_eruption(self) -> None:
        self.assertEqual(self.mapper.map("scin", "Drug Rash"), "D025")

    def test_every_source_label_has_a_canonical_counting_label(self) -> None:
        self.assertEqual(
            self.mapper.canonical_source_label("scin", "Rare Example Disease"),
            "rare_example_disease",
        )

    def test_scin_differential_is_ranked_by_weight(self) -> None:
        result = _parse_scin_differential(
            "{'Psoriasis': 0.2, 'Acne': 0.8}",
            self.mapper,
            "scin",
        )
        self.assertEqual([item["disease_id"] for item in result], ["D011", "D003"])
        self.assertEqual([item["rank"] for item in result], [1, 2])

    def test_invalid_fitzpatrick_value_is_omitted(self) -> None:
        self.assertIsNone(_fitzpatrick_value(0))
        self.assertEqual(_fitzpatrick_value(4.0), "4")

    def test_scin_two_or_more_races_uses_a_category_not_a_boolean(self) -> None:
        self.assertEqual(
            _normalize_scin_race_ethnicity("TWO_OR_MORE_AFTER_MITIGATION"),
            "two_or_more_races",
        )


class CoverageTests(unittest.TestCase):
    def test_support_passes_preliminary_thresholds(self) -> None:
        self.assertEqual(
            _support_status(
                total_groups=100,
                contributing_count=2,
                minimum_groups=100,
                minimum_datasets=2,
            ),
            ("pending_split_validation", None),
        )

    def test_support_reports_all_failed_thresholds(self) -> None:
        status, reason = _support_status(
            total_groups=20,
            contributing_count=1,
            minimum_groups=100,
            minimum_datasets=2,
        )
        self.assertEqual(status, "long_tail")
        self.assertIn("insufficient_unique_groups", reason)
        self.assertIn("insufficient_independent_datasets", reason)

    def test_age_standardization_preserves_source_granularity(self) -> None:
        self.assertEqual(
            standardize_age_group(source_group="AGE_30_TO_39"),
            "30_to_39",
        )
        self.assertEqual(
            standardize_age_group(age_years=72),
            "70_and_over",
        )
        self.assertEqual(
            standardize_age_group(source_group="AGE_UNKNOWN"),
            "unknown",
        )

    def test_missingness_categories_are_not_metric_subgroups(self) -> None:
        values = pd.Series(["female", "other_or_unspecified", None])
        self.assertEqual(
            _informative_demographic_mask(
                values,
                dimension="sex_or_gender",
            ).tolist(),
            [True, False, False],
        )


class TaxonomyContractTests(unittest.TestCase):
    def test_active_taxonomy_and_output_schema_are_synchronized(self) -> None:
        taxonomy = yaml.safe_load(
            (ROOT / "configs/taxonomies/diseases.yaml").read_text()
        )
        schema = json.loads(
            (ROOT / "schemas/visual_top_k.schema.json").read_text()
        )
        benchmark = yaml.safe_load(
            (ROOT / "configs/benchmarks/visual_top_k.yaml").read_text()
        )
        active_ids = [item["id"] for item in taxonomy["diseases"]]
        retired_ids = {item["id"] for item in taxonomy["retired_diseases"]}
        schema_ids = schema["properties"]["predictions"]["items"]["properties"][
            "disease_id"
        ]["enum"]

        self.assertEqual(len(active_ids), 21)
        self.assertEqual(active_ids, schema_ids)
        self.assertFalse(set(active_ids) & retired_ids)
        self.assertEqual(benchmark["taxonomy"]["expected_size"], 21)


if __name__ == "__main__":
    unittest.main()
