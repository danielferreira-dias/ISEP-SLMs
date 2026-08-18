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
13. [ISEPDermaBench Hugging Face release](dataset_pipeline/13_isep_dermabench_huggingface_release.md)

## Benchmarks

1. [Paired visual disease confusion sets](benchmarks/01_visual_disease_confusion_sets.md)
2. [Open-ended diagnosis with a single blinded judge](benchmarks/02_open_ended_diagnosis_single_judge.md)
3. [Validation teacher screening: thinking-off results and full disclosure](benchmarks/03_validation_teacher_screening_thinking_off.md)
4. [Reasoning screening summary](benchmarks/04_reasoning_screening_summary.md)
5. [Teacher/student selection and final benchmark process](benchmarks/05_teacher_student_selection_and_final_benchmark_process.md)
6. [Visual-grounding no-image ablation](benchmarks/06_visual_grounding_no_image_ablation.md)
7. [Small visual hallucination audits](benchmarks/07_visual_hallucination_audits.md)
8. [Visual hallucination audit results](benchmarks/08_visual_hallucination_audit_results.md)
9. [Expanded visual hallucination audits](benchmarks/09_expanded_visual_hallucination_audits.md)
10. [Internal Benchmark: complete pre-training model comparison](benchmarks/10_internal_benchmark_qwen_3_5_vs_qwen_3_6.md)
11. [Internal Benchmark temperature sensitivity](benchmarks/11_internal_benchmark_temperature_sensitivity.md)
12. [DermoBench and Clinical Context comparison](benchmarks/12_dermobench_and_clinical_context_qwen_comparison.md)
13. [Same-hardware E1 quality-efficiency Pareto comparison](benchmarks/13_same_hardware_efficiency_pareto_e1.md)
14. [ISEPDermaBench: E1 label-only versus E2 multitask](benchmarks/14_e1_vs_e2_internal_benchmark.md)
15. [MedSigLIP zero-shot on the disease-classification Internal Benchmark](benchmarks/15_medsiglip_zero_shot_internal_benchmark.md)

## Training steps

1. [E1 label-only: frozen vision versus Vision LoRA](training_steps/01_e1_label_only_vision_lora_ablation.md)
2. [E1 continued fine-tuning: epochs 4 and 5](training_steps/02_e1_continued_fine_tuning_epochs_4_5.md)
3. [E2 learning-rate and visual-LoRA smoke pilots](training_steps/03_e2_learning_rate_and_vision_smoke_pilots.md)
4. [E2 full multitask campaign and E1 comparison](training_steps/04_e2_full_multitask_campaign_and_e1_comparison.md)

## Final dataset

1. [ISEPDistillDataset construction plan](final_dataset/01_isep_distill_dataset_construction_plan.md)
2. [Adaptive clinical-context acquisition strategy](final_dataset/02_adaptive_context_acquisition_strategy.md)

## Notes

1. [Interactive multimodal dermatology model strategy](notes/01_estrategia_modelo_multimodal_interativo.md)
2. [Teacher thinking and small-model distillation](notes/02_thinking_teacher_e_distillation.md)
3. [Output limitations, parsing, and metric policy](notes/03_limitacoes_output_parsing_e_metricas.md)
4. [Training and evaluation data organization](notes/04_organizacao_dos_dados_de_treino_e_avaliacao.md)
5. [Teacher selection, benchmark validation, and interactive synthetic data](notes/05_selecao_do_teacher_validacao_das_benchmarks_e_dados_interativos.md)
6. [Flow between Validation, Internal Benchmark, and fine-tuning](notes/06_fluxo_validation_internal_benchmark_e_fine_tuning.md)
7. [Open-ended prompt A/B test](notes/07_open_ended_prompt_ab_test.md)
8. [Open-ended prompt freeze](notes/08_open_ended_prompt_freeze.md)
9. [Final 50-case open-ended prompt A/B](notes/09_open_ended_prompt_final_ab.md)
10. [Dermatologist visual reasoning prompt research](notes/10_dermatologist_visual_reasoning_prompt_research.md)
11. [Dermatology MLLM and reasoning strategy research](notes/11_dermatology_mllm_reasoning_strategy_research.md)
12. [Open-ended model and judge prompt decision record](notes/12_open_ended_model_and_judge_prompt_decision_record.md)
13. [Textbook-grounded dermatology reasoning](notes/13_textbook_grounded_dermatology_reasoning.md)
14. [Synthetic dataset comparison and architecture decision](notes/14_synthetic_dataset_comparison_and_architecture_decision.md)
15. [Reasoning, vision, and the no-image grounding ablation](notes/15_reasoning_vision_and_no_image_grounding_ablation.md)
16. [SkinFlow and the visual-encoder strategy](notes/16_skinflow_architecture_and_visual_encoder_strategy.md)
17. [Visual attribution, lesion localization, and diagnostic accuracy](notes/17_visual_attribution_localization_vs_diagnostic_accuracy.md)
18. [Efficiency and agentic benchmark strategy](notes/18_efficiency_and_agentic_benchmark_strategy.md)
