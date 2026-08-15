# E3 hard-KD contracts

This package owns the future teacher-derived Stage-A/Stage-B target contract.
It is deliberately separate from E2, whose supervision is exclusively human:
canonical diagnosis labels plus SKINCON morphology annotations.

The E3 package validates accepted structured targets and renders a
deterministic open response using one of twelve frozen surface templates. It
does not call a teacher and is not yet registered as a runnable training phase.
Registration remains blocked until accepted teacher generations are
materialized in a versioned `ISEPDistillDataset` release.
