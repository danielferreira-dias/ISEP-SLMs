"""Tests for training evaluation, scientific comparison, and artefacts."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq

from src.train.artifacts import (
    ArtifactStore,
    PlottingUnavailableError,
    ThesisPlotter,
    TrainingHistoryPoint,
    compare_runs,
    export_thesis_artifacts,
    generate_report,
    load_comparable_run,
    write_comparable_run_snapshot,
)
from src.train.evaluation import (
    CheckpointScore,
    ComparableRun,
    LabelAlias,
    LabelVocabulary,
    PredictionInput,
    PredictionRecord,
    RunContract,
    aggregate_seed_runs,
    canonicalize_predictions,
    compare_paired_runs,
    evaluate_predictions,
    select_best_checkpoint,
)
from src.train.reporting import build_run_report

LABELS = ("melanoma", "basal cell carcinoma")
CONTRACT = RunContract(
    dataset_revision="data-rev",
    split_hash="split-hash",
    prompt_hash="prompt-hash",
    model_revision="model-rev",
    label_contract_hash="labels-hash",
    training_contract_hash="training-hash",
)


def _records(
    outputs: tuple[str, ...], *, seed: int = 42
) -> tuple[PredictionRecord, ...]:
    gold = ("melanoma", "basal cell carcinoma", "melanoma", "basal cell carcinoma")
    groups = ("group-1", "group-1", "group-2", "group-3")
    vocabulary = LabelVocabulary(
        LABELS,
        (LabelAlias("BCC", "basal cell carcinoma"),),
    )
    return canonicalize_predictions(
        tuple(
            PredictionInput(
                sample_id=f"sample-{index}",
                leakage_group_id=groups[index],
                true_label=gold[index],
                raw_output=output,
                checkpoint_id="epoch-3",
                seed=seed,
            )
            for index, output in enumerate(outputs)
        ),
        vocabulary,
    )


def _run(
    experiment_id: str,
    run_id: str,
    outputs: tuple[str, ...],
    *,
    seed: int = 42,
    contract: RunContract = CONTRACT,
) -> ComparableRun:
    predictions = _records(outputs, seed=seed)
    return ComparableRun(
        experiment_id=experiment_id,
        run_id=run_id,
        seed=seed,
        contract=contract,
        predictions=predictions,
        metrics=evaluate_predictions(predictions, LABELS),
        gpu_hours=1.25,
        peak_vram_gib=18.5,
        trainable_parameters=123_456,
    )


class LabelAndMetricTests(unittest.TestCase):
    def test_parser_accepts_only_exact_canonical_or_alias_values(self) -> None:
        vocabulary = LabelVocabulary(
            LABELS,
            (LabelAlias("BCC", "basal cell carcinoma"),),
        )

        self.assertEqual(vocabulary.parse("  bCc\n"), "basal cell carcinoma")
        self.assertEqual(vocabulary.parse("MELANOMA"), "melanoma")
        self.assertIsNone(vocabulary.parse("melanoma because it is asymmetric"))
        self.assertIsNone(vocabulary.parse('"melanoma"'))
        with self.assertRaisesRegex(ValueError, "collides"):
            LabelVocabulary(
                LABELS,
                (LabelAlias("melanoma", "basal cell carcinoma"),),
            )

    def test_metrics_count_invalid_output_as_incorrect(self) -> None:
        predictions = _records(("melanoma", "BCC", "BCC", "I think this is BCC"))

        metrics = evaluate_predictions(predictions, LABELS)

        self.assertEqual(metrics.sample_count, 4)
        self.assertEqual(metrics.valid_count, 3)
        self.assertEqual(metrics.invalid_count, 1)
        self.assertAlmostEqual(metrics.top1_accuracy, 0.5)
        self.assertAlmostEqual(metrics.invalid_rate, 0.25)
        self.assertAlmostEqual(metrics.balanced_accuracy, 0.5)
        self.assertAlmostEqual(metrics.macro_f1, 7.0 / 12.0)
        self.assertEqual(metrics.confusion_matrix, ((1, 1), (0, 1)))

    def test_checkpoint_tie_breaking_is_fully_deterministic(self) -> None:
        metrics = evaluate_predictions(
            _records(("melanoma", "BCC", "melanoma", "BCC")), LABELS
        )
        tied_metrics = replace(metrics, macro_f1=0.9)
        checkpoints = (
            CheckpointScore("late", 3.0, 0.3, tied_metrics),
            CheckpointScore("high-loss", 1.0, 0.4, tied_metrics),
            CheckpointScore("winner", 1.0, 0.3, tied_metrics),
        )

        selected = select_best_checkpoint(checkpoints)

        self.assertEqual(selected.checkpoint_id, "winner")
        improved = replace(metrics, macro_f1=0.901)
        selected = select_best_checkpoint(
            (*checkpoints, CheckpointScore("better-f1", 4.0, 9.0, improved))
        )
        self.assertEqual(selected.checkpoint_id, "better-f1")


class ComparisonTests(unittest.TestCase):
    def test_group_bootstrap_and_exact_mcnemar_are_reproducible(self) -> None:
        baseline = _run("frozen", "frozen-42", ("melanoma", "BCC", "BCC", "melanoma"))
        candidate = _run("vision", "vision-42", ("melanoma", "BCC", "melanoma", "BCC"))

        first = compare_paired_runs(
            baseline, candidate, bootstrap_iterations=250, bootstrap_seed=9
        )
        second = compare_paired_runs(
            baseline, candidate, bootstrap_iterations=250, bootstrap_seed=9
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first.top1_delta, 0.5)
        self.assertEqual(first.baseline_only_correct, 0)
        self.assertEqual(first.candidate_only_correct, 2)
        self.assertAlmostEqual(first.mcnemar_exact_p, 0.5)
        self.assertEqual(first.group_count, 3)

    def test_comparison_rejects_different_training_contract(self) -> None:
        baseline = _run("frozen", "frozen-42", ("melanoma", "BCC", "BCC", "BCC"))
        changed_contract = replace(CONTRACT, training_contract_hash="other")
        candidate = _run(
            "vision",
            "vision-42",
            ("melanoma", "BCC", "melanoma", "BCC"),
            contract=changed_contract,
        )

        with self.assertRaisesRegex(ValueError, "scientific contract"):
            compare_paired_runs(baseline, candidate, bootstrap_iterations=10)

    def test_seed_aggregation_uses_sample_standard_deviation(self) -> None:
        first = _run("frozen", "frozen-42", ("melanoma", "BCC", "BCC", "BCC"))
        second = _run(
            "frozen",
            "frozen-3407",
            ("melanoma", "BCC", "melanoma", "BCC"),
            seed=3407,
        )

        aggregate = aggregate_seed_runs((first, second))[0]

        self.assertEqual(aggregate.seeds, (42, 3407))
        self.assertAlmostEqual(aggregate.top1_accuracy.mean, 0.875)
        self.assertGreater(aggregate.top1_accuracy.standard_deviation, 0.0)


class ArtifactTests(unittest.TestCase):
    def test_report_materializes_authoritative_training_history_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore.create(Path(directory), "experiment", "run-history")
            events = (
                {
                    "name": "custom_scalar",
                    "value": 0.75,
                    "step": 10,
                    "epoch": 1.0,
                    "timestamp_utc": "2026-08-12T10:30:00+00:00",
                },
                {
                    "name": "custom_scalar",
                    "value": 0.5,
                    "step": 20,
                    "epoch": None,
                    "timestamp_utc": "2026-08-12T10:31:00Z",
                },
            )
            source = "".join(json.dumps(event) + "\n" for event in events)
            jsonl_path = store.write_text("logs", "metrics.jsonl", source)

            build_run_report(store.layout.run_directory)

            self.assertEqual(jsonl_path.read_text(encoding="utf-8"), source)
            csv_path = store.path("metrics", "training_history.csv")
            parquet_path = store.path("metrics", "training_history.parquet")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    ["name", "value", "step", "epoch", "timestamp_utc"],
                )
                rows = list(reader)
            self.assertEqual([row["step"] for row in rows], ["10", "20"])
            self.assertEqual(rows[1]["epoch"], "")
            table = pq.read_table(parquet_path)
            self.assertEqual(
                table.column_names,
                ["name", "value", "step", "epoch", "timestamp_utc"],
            )
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(table.column("step").to_pylist(), [10, 20])

    def test_invalid_metric_log_does_not_replace_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore.create(Path(directory), "experiment", "run-atomic")
            valid = {
                "name": "custom_scalar",
                "value": 0.75,
                "step": 10,
                "epoch": 1.0,
                "timestamp_utc": "2026-08-12T10:30:00+00:00",
            }
            log_path = store.write_text(
                "logs", "metrics.jsonl", json.dumps(valid) + "\n"
            )
            build_run_report(store.layout.run_directory)
            csv_path = store.path("metrics", "training_history.csv")
            parquet_path = store.path("metrics", "training_history.parquet")
            original_csv = csv_path.read_bytes()
            original_parquet = parquet_path.read_bytes()

            invalid_lines = (
                "{",
                "\n",
                '{"name":"loss"}\n',
                (
                    '{"name":"loss","value":NaN,"step":1,"epoch":1.0,'
                    '"timestamp_utc":"2026-08-12T10:30:00+00:00"}\n'
                ),
                (
                    '{"name":"loss","value":0.5,"step":1,"epoch":1.0,'
                    '"timestamp_utc":"2026-08-12T10:30:00+01:00"}\n'
                ),
            )
            for invalid in invalid_lines:
                with self.subTest(invalid=invalid):
                    store.write_text("logs", "metrics.jsonl", invalid)
                    with self.assertRaisesRegex(ValueError, "Metric JSONL line 1"):
                        build_run_report(store.layout.run_directory)
                    self.assertEqual(log_path.read_text(encoding="utf-8"), invalid)
                    self.assertEqual(csv_path.read_bytes(), original_csv)
                    self.assertEqual(parquet_path.read_bytes(), original_parquet)

    def test_snapshot_roundtrip_reports_comparison_and_safe_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_store = ArtifactStore.create(root / "runs", "frozen", "seed-42")
            candidate_store = ArtifactStore.create(root / "runs", "vision", "seed-42")
            baseline = _run("frozen", "seed-42", ("melanoma", "BCC", "BCC", "BCC"))
            candidate = _run(
                "vision", "seed-42", ("melanoma", "BCC", "melanoma", "BCC")
            )
            write_comparable_run_snapshot(baseline_store, baseline)
            write_comparable_run_snapshot(candidate_store, candidate)

            restored = load_comparable_run(baseline_store.layout.run_directory)
            self.assertEqual(restored, baseline)
            report = generate_report(baseline_store.layout.run_directory)
            for path in (
                report.markdown_path,
                report.html_path,
                report.metrics_csv_path,
                report.metrics_latex_path,
            ):
                self.assertGreater(path.stat().st_size, 0)
            self.assertNotIn(
                "I think",
                report.html_path.read_text(encoding="utf-8"),
            )

            comparison = compare_runs(
                (
                    baseline_store.layout.run_directory,
                    candidate_store.layout.run_directory,
                ),
                root / "comparison",
                bootstrap_iterations=100,
                render_plots=False,
            )
            for path in (
                comparison.json_path,
                comparison.csv_path,
                comparison.latex_path,
                comparison.markdown_path,
                comparison.html_path,
            ):
                self.assertGreater(path.stat().st_size, 0)

            export_root = root / "doc" / "generated"
            first_export = export_thesis_artifacts(
                baseline_store.layout.run_directory, export_root
            )
            second_export = export_thesis_artifacts(
                baseline_store.layout.run_directory, export_root
            )
            self.assertTrue(first_export.copied)
            self.assertEqual(
                set(first_export.copied), set(second_export.skipped_existing)
            )

    def test_store_creates_complete_tree_and_rejects_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore.create(Path(directory), "experiment", "run-001")
            for section in (
                "manifests",
                "logs",
                "tensorboard",
                "checkpoints",
                "predictions",
                "metrics",
                "figures",
                "tables",
                "report",
            ):
                self.assertTrue(store.layout.section(section).is_dir())
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                store.path("metrics", "../escape.json")
            with self.assertRaises(FileExistsError):
                ArtifactStore.create(Path(directory), "experiment", "run-001")

    def test_plotting_is_lazy_and_preserves_source_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure_directory = Path(directory) / "figures"
            plotter = ThesisPlotter(figure_directory)
            points = (
                TrainingHistoryPoint(0, 0.0, 1.0, 1.1, 0.0),
                TrainingHistoryPoint(10, 0.2, 0.8, 0.9, 0.0002),
            )
            matplotlib_present = importlib.util.find_spec("matplotlib") is not None
            if matplotlib_present:
                figures = plotter.training_history(points)
                self.assertTrue(figures)
                for figure in figures:
                    self.assertGreater(figure.png_path.stat().st_size, 0)
                    self.assertGreater(figure.svg_path.stat().st_size, 0)
            else:
                with self.assertRaises(PlottingUnavailableError):
                    plotter.training_history(points)
            self.assertGreater(
                (figure_directory / "training_history_source.csv").stat().st_size,
                0,
            )


if __name__ == "__main__":
    unittest.main()
