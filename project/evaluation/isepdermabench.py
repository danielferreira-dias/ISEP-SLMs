"""Planned ISEPDermaBench evaluation adapter.

https://huggingface.co/datasets/danielfdias98/ISEPDermaBench

The benchmark includes:

- grounded diagnosis and visual top-k classification;
- context ablation before and after additional context;
- visual confusion sets between similar diseases;
- unsupported-claim and hallucination analysis;
- open-ended diagnosis with a separately authorized LLM judge;
- evidence grounding with a separately authorized LLM judge.

This module must not select E3 checkpoints; the fixed benchmark is final-only.
"""
