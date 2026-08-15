"""CPU-only tests for human diagnosis and SKINCON E2 training."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from src.train.artifacts import ArtifactStore
from src.train.domain import ReleaseSubset, Taxonomy, TaxonomyClass
from src.train.e2 import (
    CaptionPredictionInput,
    DeterministicTaskInterleave,
    E2HumanDataset,
    E2HumanPhase,
    E2TaskName,
    MorphologyPredictionInput,
    MorphologyTarget,
    SkinConOntology,
    canonicalize_caption_predictions,
    canonicalize_morphology_predictions,
    caption_prompt,
    evaluate_caption_predictions,
    evaluate_morphology_predictions,
    morphology_prompt,
)
from src.train.e2.caption_plots import render_caption_and_multitask_plots
from src.train.e2.plots import render_morphology_plots
from src.train.phases.label_only import LabelOnlyPhase


class _Rows:
    def __init__(self, rows: tuple[Mapping[object, object], ...]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> object:
        return self.rows[index]


class E2HumanDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = Taxonomy(
            "test",
            (
                TaxonomyClass("D001", "melanoma"),
                TaxonomyClass("D002", "eczema"),
            ),
        )
        self.ontology = SkinConOntology("skincon_48_v1", ("Papule", "Plaque", "Scale"))
        self.phase = E2HumanPhase(self.taxonomy, self.ontology)

    def test_morphology_row_reconstructs_real_multimodal_message(self) -> None:
        row = _morphology_row(self.ontology)
        dataset = E2HumanDataset(
            backing=_Rows((row,)),
            task=E2TaskName.MORPHOLOGY,
            subset=ReleaseSubset.SFT_TRAIN,
            phase=self.phase,
            schema_version="0.3.0",
            max_edge_pixels=512,
            availability_by_image={str(row["image_sha256"]): (E2TaskName.MORPHOLOGY,)},
        )

        record = dataset[0]

        messages = record["messages"]
        self.assertIsInstance(messages, list)
        assert isinstance(messages, list)
        image = messages[0]["content"][0]["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(record["task_id"], "morphology")
        self.assertNotIn("lichen planus", str(messages[0]))

    def test_teacher_target_source_is_rejected(self) -> None:
        row = dict(_morphology_row(self.ontology))
        row["target_source"] = "teacher_generated"
        dataset = E2HumanDataset(
            backing=_Rows((row,)),
            task=E2TaskName.MORPHOLOGY,
            subset=ReleaseSubset.SFT_TRAIN,
            phase=self.phase,
            schema_version="0.3.0",
            max_edge_pixels=512,
            availability_by_image={str(row["image_sha256"]): (E2TaskName.MORPHOLOGY,)},
        )
        with self.assertRaisesRegex(ValueError, "target_source"):
            _ = dataset[0]

    def test_diagnosis_reuses_exact_e1_contract(self) -> None:
        row = _diagnosis_row(self.taxonomy)
        dataset = E2HumanDataset(
            backing=_Rows((row,)),
            task=E2TaskName.DIAGNOSIS,
            subset=ReleaseSubset.SFT_TRAIN,
            phase=self.phase,
            schema_version="0.3.0",
            max_edge_pixels=512,
            availability_by_image={str(row["image_sha256"]): (E2TaskName.DIAGNOSIS,)},
        )
        messages = dataset[0]["messages"]
        assert isinstance(messages, list)
        self.assertEqual(messages[1]["content"][0]["text"], "melanoma")
        self.assertEqual(
            messages[0]["content"][1]["text"],
            LabelOnlyPhase(self.taxonomy).prompt,
        )

    def test_caption_row_exposes_only_filtered_target_and_provenance_hash(self) -> None:
        row = _caption_row()
        dataset = E2HumanDataset(
            backing=_Rows((row,)),
            task=E2TaskName.CAPTION,
            subset=ReleaseSubset.SFT_TRAIN,
            phase=self.phase,
            schema_version="0.4.1",
            max_edge_pixels=512,
            availability_by_image={str(row["image_sha256"]): (E2TaskName.CAPTION,)},
        )

        record = dataset[0]

        messages = record["messages"]
        assert isinstance(messages, list)
        self.assertEqual(record["task_id"], "caption")
        self.assertEqual(
            messages[1]["content"][0]["text"],
            "A sharply demarcated erythematous plaque with fine scale is visible.",
        )
        self.assertNotIn("psoriasis", str(messages).lower())
        self.assertNotIn("source_caption_sha256", record)


class E2MixingTests(unittest.TestCase):
    def test_interleave_consumes_every_row_once_without_oversampling(self) -> None:
        diagnosis = tuple({"task_id": "diagnosis", "index": i} for i in range(7))
        morphology = tuple({"task_id": "morphology", "index": i} for i in range(3))
        mixed = DeterministicTaskInterleave(diagnosis, morphology)

        self.assertEqual(len(mixed), 10)
        self.assertEqual(mixed.task_counts, (7, 3))
        observed = tuple((mixed[i]["task_id"], mixed[i]["index"]) for i in range(10))
        self.assertEqual(len(set(observed)), 10)
        self.assertEqual(sum(task == "diagnosis" for task, _ in observed), 7)
        self.assertEqual(sum(task == "morphology" for task, _ in observed), 3)

    def test_three_task_interleave_is_exact_and_deterministic(self) -> None:
        diagnosis = tuple({"task_id": "diagnosis", "index": i} for i in range(7))
        morphology = tuple({"task_id": "morphology", "index": i} for i in range(3))
        captions = tuple({"task_id": "caption", "index": i} for i in range(4))

        mixed = DeterministicTaskInterleave(diagnosis, morphology, captions)
        observed = tuple((mixed[i]["task_id"], mixed[i]["index"]) for i in range(14))

        self.assertEqual(mixed.task_counts, (7, 3, 4))
        self.assertEqual(len(set(observed)), 14)
        self.assertEqual(sum(task == "caption" for task, _ in observed), 4)
        self.assertEqual(
            observed,
            tuple(
                (
                    DeterministicTaskInterleave(diagnosis, morphology, captions)[i][
                        "task_id"
                    ],
                    DeterministicTaskInterleave(diagnosis, morphology, captions)[i][
                        "index"
                    ],
                )
                for i in range(14)
            ),
        )


class E2MorphologyMetricTests(unittest.TestCase):
    def test_metrics_keep_invalid_outputs_in_denominator(self) -> None:
        ontology = SkinConOntology("skincon_48_v1", ("Papule", "Plaque", "Scale"))
        inputs = (
            MorphologyPredictionInput(
                "s1",
                "g1",
                ("Papule",),
                '{"positive_concepts":["Papule"],"all_concepts_annotated":true}',
                "checkpoint-1",
                3407,
            ),
            MorphologyPredictionInput(
                "s2", "g2", ("Scale",), "not json", "checkpoint-1", 3407
            ),
        )
        records = canonicalize_morphology_predictions(inputs, ontology)
        metrics = evaluate_morphology_predictions(records, ontology)

        self.assertEqual(metrics.sample_count, 2)
        self.assertEqual(metrics.exact_match, 0.5)
        self.assertEqual(metrics.invalid_output_rate, 0.5)
        self.assertAlmostEqual(metrics.micro_precision, 1.0)
        self.assertAlmostEqual(metrics.micro_recall, 0.5)
        self.assertAlmostEqual(metrics.micro_f1, 2 / 3)

    def test_unknown_or_duplicate_concepts_are_invalid(self) -> None:
        ontology = SkinConOntology("skincon_48_v1", ("Papule", "Scale"))
        for output in (
            '{"positive_concepts":["Unknown"],"all_concepts_annotated":true}',
            '{"positive_concepts":["Papule","Papule"],"all_concepts_annotated":true}',
        ):
            records = canonicalize_morphology_predictions(
                (
                    MorphologyPredictionInput(
                        "s", "g", ("Papule",), output, "base", 3407
                    ),
                ),
                ontology,
            )
            self.assertFalse(records[0].is_valid)


class E2CaptionMetricTests(unittest.TestCase):
    def test_caption_metrics_separate_compliance_reference_and_concepts(self) -> None:
        ontology = SkinConOntology("test", ("Plaque", "Scale", "Papule"))
        records = canonicalize_caption_predictions(
            (
                CaptionPredictionInput(
                    sample_id="a",
                    leakage_group_id="g1",
                    reference_text="A red plaque with fine scale is visible.",
                    true_concepts=("Plaque", "Scale"),
                    raw_output="A red plaque with fine scale is visible.",
                    checkpoint_id="checkpoint-1",
                    seed=42,
                ),
                CaptionPredictionInput(
                    sample_id="b",
                    leakage_group_id="g2",
                    reference_text="A solitary papule is visible on the skin.",
                    true_concepts=("Papule",),
                    raw_output="Diagnosis: melanoma; biopsy is recommended.",
                    checkpoint_id="checkpoint-1",
                    seed=42,
                ),
            ),
            ontology,
            ("melanoma", "eczema"),
        )

        metrics = evaluate_caption_predictions(records)

        self.assertEqual(metrics.sample_count, 2)
        self.assertEqual(metrics.clinical_compliance_rate, 0.5)
        self.assertEqual(metrics.prohibited_content_rate, 0.5)
        self.assertAlmostEqual(metrics.concept_recall, 2 / 3)
        self.assertGreater(metrics.reference_similarity_mean, 0.4)


class E2MorphologyPlotTests(unittest.TestCase):
    def test_thesis_figures_include_png_svg_and_csv_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore.create(Path(temporary), "e2", "run")
            store.write_json(
                "metrics",
                "morphology_sft_dev__checkpoint-1.json",
                {
                    "checkpoint_id": "checkpoint-1",
                    "epoch": 1.0,
                    "micro_f1": 0.7,
                    "macro_f1": 0.6,
                    "exact_match": 0.4,
                    "invalid_output_rate": 0.1,
                    "per_concept": [
                        {
                            "concept": "Papule",
                            "support": 5,
                            "precision": 0.8,
                            "recall": 0.75,
                            "f1": 0.774,
                        }
                    ],
                },
            )
            store.write_json(
                "manifests",
                "best_checkpoint.json",
                {"checkpoint_id": "checkpoint-1"},
            )

            figures = render_morphology_plots(store)

            self.assertEqual(len(figures), 2)
            for stem in (
                "morphology_checkpoint_quality",
                "morphology_per_concept",
            ):
                self.assertTrue((store.layout.figures / f"{stem}.png").is_file())
                self.assertTrue((store.layout.figures / f"{stem}.svg").is_file())
                self.assertTrue((store.layout.figures / f"{stem}_source.csv").is_file())

    def test_caption_and_task_figures_include_csv_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore.create(Path(temporary), "e2", "run")
            store.write_json(
                "metrics",
                "caption_sft_dev__checkpoint-1.json",
                {
                    "checkpoint_id": "checkpoint-1",
                    "epoch": 1.0,
                    "caption_task_score": 0.72,
                    "concept_f1": 0.68,
                    "reference_similarity_mean": 0.61,
                    "clinical_compliance_rate": 0.87,
                },
            )
            store.write_json(
                "metrics",
                "multitask_sft_dev__checkpoint-1.json",
                {
                    "checkpoint_id": "checkpoint-1",
                    "diagnosis_macro_f1": 0.75,
                    "morphology_macro_f1": 0.63,
                    "caption_task_score": 0.72,
                },
            )
            store.write_json(
                "manifests",
                "best_checkpoint.json",
                {"checkpoint_id": "checkpoint-1"},
            )

            figures = render_caption_and_multitask_plots(store)

            self.assertEqual(len(figures), 2)
            for stem in ("caption_checkpoint_quality", "e2_selected_task_scores"):
                self.assertTrue((store.layout.figures / f"{stem}.png").is_file())
                self.assertTrue((store.layout.figures / f"{stem}.svg").is_file())
                self.assertTrue((store.layout.figures / f"{stem}_source.csv").is_file())


def _encoded_image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 8), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _common_row() -> dict[str, object]:
    encoded = _encoded_image()
    return {
        "image": {"bytes": encoded, "path": "image.png"},
        "sample_id": "sample-1",
        "leakage_group_id": "group-1",
        "source_dataset": "fitzpatrick17k",
        "split": "sft_train",
        "schema_version": "0.3.0",
        "quality_status": "accepted",
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _morphology_row(ontology: SkinConOntology) -> dict[str, object]:
    prompt = morphology_prompt(ontology)
    target = MorphologyTarget(("Papule", "Scale"), True).canonical_json()
    return {
        **_common_row(),
        "task_id": "skincon_morphology_v1",
        "target_source": "human_annotated",
        "target_variant": "skincon_positive_concepts_v1",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_text": target,
        "gold_diagnosis": "lichen planus",
        "skincon": {
            "ontology_version": "skincon_48_v1",
            "annotation_source": "SKINCON",
            "positive_concepts": ["Papule", "Scale"],
            "all_concepts_annotated": True,
        },
    }


def _diagnosis_row(taxonomy: Taxonomy) -> dict[str, object]:
    prompt = LabelOnlyPhase(taxonomy).prompt
    return {
        **_common_row(),
        "task_id": "diagnosis_label_only_v1",
        "target_source": "human_source_join",
        "target_variant": "canonical_label_v1",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_text": "melanoma",
        "disease_id": "D001",
        "gold_diagnosis": "melanoma",
    }


def _caption_row() -> dict[str, object]:
    prompt = caption_prompt()
    return {
        **_common_row(),
        "schema_version": "0.4.1",
        "task_id": "skincap_observation_caption_v1",
        "target_source": "human_caption_gold_conditioned_filtered",
        "target_variant": "observation_only_single_sentence_v1",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_text": (
            "A sharply demarcated erythematous plaque with fine scale is visible."
        ),
        "source_caption_sha256": "a" * 64,
        "transform_version": "skincap_observation_prefix_v1",
    }


if __name__ == "__main__":
    unittest.main()
