# Open-ended diagnosis prompt freeze

> Superseded on 2026-08-02 by the paired 50-case decision documented in
> `09_open_ended_prompt_final_ab.md`. Release 1.4.0 is retained as protocol
> history; release 1.5.0 is the final frozen protocol.

Date: 2026-08-02

## Decision

Freeze evaluated-model prompt 1.2.0 and judge prompt 1.2.0 in
ISEPDermaBench release 1.4.0. Subsequent Validation comparisons must use these
exact prompt contents and hashes. A prompt change requires a new protocol and
release version; it must not silently replace this frozen version.

## Final acceptance run

- evaluated model: `gpt_5_6_luna`;
- judge: `gpt_5_6_luna`;
- split: Validation;
- sample: 20 deterministically selected cases;
- seed: 42;
- model prompt: 1.2.0;
- judge prompt: 1.2.0;
- reasoning effort: high;
- fallback judge: none.

All 20 responses used the required `Visible findings`, `Most likely
diagnosis`, `Second most likely diagnosis`, and `Third most likely diagnosis`
sections. Nineteen responses included an `Image limitations` sentence and no
case was declared not evaluable.

| Acceptance metric | Result |
| --- | ---: |
| Valid model responses | 20/20 |
| Format-invalid responses | 0 |
| Semantic-noncompliant responses | 0 |
| Truncated responses | 0 |
| Safety refusals | 0 |
| Judge coverage | 100% |
| Judge-invalid responses | 0 |
| Evidence grounding, 0–4 | 3.70 |
| Visual findings, 0–4 | 3.20 |
| Clinical rationale, 0–4 | 2.65 |
| Differential quality, 0–4 | 2.75 |
| Unsupported-claim rate | 15% |

The diagnostic results were Top-1 25%, Top-3 35%, and mean reciprocal rank
0.30. These values describe Luna on this small selected cohort and are not
prompt acceptance thresholds. The freeze decision is based on complete
protocol compliance, judge coverage, strong evidence grounding, and low
unsupported-claim frequency. Diagnostic accuracy remains a model outcome to
be measured on the complete Validation split.

## Known limitation

The `Image limitations` sentence appeared in 19/20 responses. This indicates a
conservative interpretation of photographic limitations and should be
reported when analysing response style. It does not invalidate the frozen
protocol, but it must not be silently removed after model comparisons begin.
