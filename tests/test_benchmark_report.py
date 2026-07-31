"""Tests for standalone benchmark HTML reports."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from src.benchmark.report import generate_run_report


class BenchmarkReportTests(unittest.TestCase):
    def test_report_contains_safe_answers_reasoning_and_thumbnail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            malicious = "</script><script>alert('unsafe')</script>"
            prediction = {
                "task_id": "task-1",
                "sample_id": "sample-1",
                "model_id": "model-1",
                "status": "ok",
                "image_uri": "image.jpg",
                "ground_truth_disease_id": "D001",
                "metadata": {
                    "dataset_id": "dataset-1",
                    "skin_tone": "FST_3",
                },
                "response": {
                    "final_text": malicious,
                    "parsed_output": {"predictions": []},
                    "canonical_output": {"predictions": []},
                    "canonical_schema_valid": True,
                    "canonicalization_rules": [
                        "ranked_disease_id_list_to_objects"
                    ],
                    "json_valid": True,
                    "schema_valid": True,
                    "validation_errors": [],
                    "reasoning": {
                        "availability": "full",
                        "capture_mode": "full",
                        "source": "reasoning_content",
                        "token_count": 12,
                        "text": "Clinical reasoning",
                    },
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 30,
                        "total_tokens": 50,
                        "reasoning_tokens": 12,
                    },
                    "finish_reason": "stop",
                    "metadata": {},
                    "provider_metadata": {},
                },
            }
            legacy_prediction = json.loads(json.dumps(prediction))
            legacy_prediction.update(
                {
                    "task_id": "task-2",
                    "sample_id": "sample-2",
                    "status": "invalid_output",
                }
            )
            legacy_prediction["response"]["metadata"] = {
                "semantic_valid": False,
            }
            legacy_prediction["response"]["reasoning"].update(
                {
                    "availability": "summary",
                    "capture_mode": "summary",
                    "source": "reasoning.summary",
                    "text": "detailed",
                }
            )
            (run_directory / "predictions.jsonl").write_text(
                "\n".join(
                    (
                        json.dumps(prediction),
                        json.dumps(legacy_prediction),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (run_directory / "rendered_prompts.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "system_prompt": "System",
                        "user_prompt": "User",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_directory / "metrics.json").write_text(
                json.dumps(
                    {
                        "top_1_accuracy": 1.0,
                        "by_condition_top_1_accuracy": {
                            "low_confusability": 1.0,
                            "high_confusability": 0.5,
                        },
                        "by_skin_tone": {
                            "fitzpatrick:FST_3": {
                                "sample_count": 1,
                                "top_1_accuracy": 1.0,
                                "statistically_supported": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_directory / "run_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "status": "completed",
                        "model": {
                            "id": "model-1",
                            "display_name": "Model One",
                        },
                        "benchmark": {"id": "benchmark-1"},
                    }
                ),
                encoding="utf-8",
            )
            (run_directory / "config_snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "disease_taxonomy": {
                            "diseases": [
                                {
                                    "id": "D001",
                                    "display_name": "Melanoma",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            image = Image.new("RGB", (80, 60), "red")
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")

            report_path = generate_run_report(
                run_directory,
                image_loader=lambda uri: image_buffer.getvalue(),
            )

            content = report_path.read_text(encoding="utf-8")
            self.assertEqual(report_path.name, "report.html")
            self.assertIn("sample-1", content)
            self.assertIn("Clinical reasoning", content)
            self.assertIn("Canonical output", content)
            self.assertIn(
                "ranked_disease_id_list_to_objects",
                content,
            )
            self.assertIn("Melanoma", content)
            self.assertIn("data:image/jpeg;base64,", content)
            self.assertIn('"value":"100.0%"', content)
            self.assertIn('"label":"by condition top 1 accuracy"', content)
            self.assertIn(
                '"label":"low confusability","values":["100.0%"]',
                content,
            )
            self.assertIn(
                '"label":"fitzpatrick:FST_3",'
                '"values":["1","100.0%","no"]',
                content,
            )
            self.assertNotIn("[object Object]", content)
            self.assertIn(
                '"name":"semantic_noncompliant",'
                '"label":"semantic noncompliant","value":"1"',
                content,
            )
            self.assertIn(
                "Legacy run: the provider's requested summary mode",
                content,
            )
            self.assertNotIn(
                "</script><script>alert('unsafe')</script>",
                content,
            )
