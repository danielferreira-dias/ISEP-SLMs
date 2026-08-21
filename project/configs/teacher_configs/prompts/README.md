# E3 teacher prompt registry

The E3 Stage A and Stage B prompts were frozen on 2026-08-21 after the paired
Vertex dry runs. Their exact bytes are part of the generation protocol:

| Prompt | Version | SHA-256 |
| --- | --- | --- |
| `stage_a.md` | `e3_stage_a_v1` | `c28f6ff4f9a47ba23bc02f2a6d14541ee5afeeaf134bca5cf48936f150121a4f` |
| `stage_b.md` | `e3_stage_b_v1` | `b8239b38c24eac6037c22bcfcbc3573deb37cd5011c97d234e4179f20718125e` |

Both teacher configs pin the version and digest. Config loading fails before a
provider call if either file drifts. Do not edit a frozen prompt in place. A
semantic or formatting change requires a new prompt file, an explicit `v2`
version, a new digest, new output paths, and a new validation pilot.

The freeze applies to prompt content. Orchestration, retry, audit tracking, and
materialization bugs may still be corrected as long as they do not alter the
teacher request protocol. Every generated row also retains the prompt digest in
its provenance.
