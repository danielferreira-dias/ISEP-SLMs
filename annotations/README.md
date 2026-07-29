# Development annotations

This directory records how implementation and experimental stages were
designed, executed, validated, and revised. The notes are intended to preserve
the technical rationale needed when writing the dissertation.

Each annotation should identify:

- the objective and scope of the stage;
- the source inputs and assumptions;
- the implemented transformation;
- important design decisions;
- generated outputs;
- validation evidence;
- known limitations and the next decision gate.

## Dataset pipeline

1. [Source inspection and manifest design](dataset_pipeline/01_source_inspection_and_manifest_design.md)
2. [Normalization implementation](dataset_pipeline/02_normalization_implementation.md)
3. [Disease mapping and coverage analysis](dataset_pipeline/03_disease_mapping_and_coverage.md)
4. [Validation findings and current limitations](dataset_pipeline/04_validation_and_current_limitations.md)
5. [Twenty-class taxonomy expansion](dataset_pipeline/05_twenty_class_taxonomy_expansion.md)
6. [Demographic subgroups, drug eruption, and Dermnet audit integration](dataset_pipeline/06_demographic_subgroups_drug_eruption_and_dermnet.md)
7. [SkinDisNet external evaluation integration](dataset_pipeline/07_skindisnet_external_evaluation_integration.md)
8. [Exact and perceptual duplicate analysis](dataset_pipeline/08_exact_and_perceptual_duplicate_analysis.md)
9. [Duplicate review decisions](dataset_pipeline/09_duplicate_review_decisions.md)
10. [Leakage-safe benchmark release](dataset_pipeline/10_leakage_safe_benchmark_release.md)
11. [Benchmark execution smoke test](dataset_pipeline/11_benchmark_smoke_test.md)
12. [Fixed paired benchmark and demographic balance](dataset_pipeline/12_fixed_paired_benchmark_and_demographic_balance.md)
