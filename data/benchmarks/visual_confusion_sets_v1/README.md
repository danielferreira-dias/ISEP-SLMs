# Visual disease confusion-set benchmark release

This directory contains the provisional paired release used to compare visual
disease ranking under low- and high-confusability candidate conditions.

## Structure

```text
visual_confusion_sets_v1/
├── datasets/internal/confusion_tasks.parquet
├── reports/confusion_set_summary_v1.csv
└── release/
    ├── integrity_report_v1.yaml
    └── benchmark_release_v1.yaml
```

The task manifest contains 828 rows representing 414 unique images. Each image
has two rows with the same `pair_id`:

- `low_confusability`: the reference disease and two distractors from different
  appearance partitions;
- `high_confusability`: the three diseases in the clinically motivated
  confusion set.

Every task presents exactly three candidate diseases, and the model must rank
all three. The source images remain in the sealed Visual Top-K benchmark; this
release stores references and task definitions rather than duplicated image
bytes.

## Reproduce and validate

```bash
.venv/bin/python -m src.data_pipeline.confusion_sets
.venv/bin/python -m src.data_pipeline.confusion_sets --validate-only
.venv/bin/python -m src.benchmark.confusion_smoke_test
```

The release is provisional until the candidate sets and low-confusability
distractors receive clinical review. Test-set model outcomes must not be used
to revise set membership.

