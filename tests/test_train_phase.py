"""Tests for closed-taxonomy E1 label-only rendering."""

from __future__ import annotations

import unittest

from PIL import Image

from src.train.domain import (
    LabeledImageSample,
    Taxonomy,
    TaxonomyClass,
    TrainingPhaseName,
)
from src.train.phases import LabelOnlyPhase, get_phase, registered_phases


class LabelOnlyPhaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = Taxonomy(
            taxonomy_id="test",
            classes=(
                TaxonomyClass("D001", "melanoma"),
                TaxonomyClass("D002", "melanocytic_nevus"),
            ),
        )

    def test_registry_exposes_only_e1(self) -> None:
        self.assertEqual(registered_phases(), (TrainingPhaseName.E1_LABEL,))
        self.assertIsInstance(
            get_phase(TrainingPhaseName.E1_LABEL, self.taxonomy),
            LabelOnlyPhase,
        )

    def test_example_contains_image_closed_prompt_and_exact_label(self) -> None:
        phase = LabelOnlyPhase(self.taxonomy)
        formatted = phase.format_example(_sample("D001", "melanoma"))
        record = formatted.as_record()
        messages = record["messages"]

        self.assertIsInstance(messages, list)
        assert isinstance(messages, list)
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant"],
        )
        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]
        self.assertEqual(user_content[0]["type"], "image")
        prompt = user_content[1]["text"]
        self.assertIn("- melanoma", prompt)
        self.assertIn("- melanocytic_nevus", prompt)
        self.assertEqual(assistant_content, [{"type": "text", "text": "melanoma"}])
        self.assertNotIn("source", record)
        self.assertNotIn("reasoning", record)

    def test_noncanonical_pair_is_rejected(self) -> None:
        phase = LabelOnlyPhase(self.taxonomy)
        with self.assertRaises(ValueError):
            phase.format_example(_sample("D001", "melanocytic_nevus"))


def _sample(disease_id: str, label: str) -> LabeledImageSample:
    return LabeledImageSample(
        sample_id="sample-1",
        leakage_group_id="group-1",
        disease_id=disease_id,
        label=label,
        source="test-source",
        image=Image.new("RGB", (32, 16), "red"),
    )


if __name__ == "__main__":
    unittest.main()
