# Withdrawn release

`isep_distill_dataset_v0.4.0` must not be used for training or checkpoint
selection. Its caption-only split audit passed, but a later cross-configuration
audit found 125 `leakage_group_id` values assigned to `sft_train` in one task
and `sft_dev` in another.

The immutable `v0.4.0` Hub tag is retained for provenance. Use corrected
release `isep_distill_dataset_v0.4.1`, which inherits the frozen E1 or
morphology split for every shared caption group and reports zero cross-task
train/dev overlap.
