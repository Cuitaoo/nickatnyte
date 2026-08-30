# Ambiguity-Gated LLM: 200-Session Evaluation

> Superseded by `gated-llm-regression-root-cause.md`, which documents the
> selective-gate and canonicalization fixes plus their final full-set result.

Date: 2026-08-30

This run compares the combined Justin + Cui Tao pipeline with identical retrieval,
cross-encoder, business-rule, and evaluator settings. The only behavioral change
between the two runs is ambiguity-gated LLM state interpretation. OpenAI reranking
was disabled in both runs.

## Results

| Metric | Deterministic | Gated LLM | Delta |
| --- | ---: | ---: | ---: |
| Technical score | 0.887760 | 0.888631 | +0.000871 |
| Hit@10 | 0.990000 | 0.990000 | 0 |
| MRR | 0.810200 | 0.807770 | -0.002430 |
| MTTC | 3.515000 | 3.435000 | -0.080000 |
| Total API tokens | 0 | 318,473 | +318,473 |

The LLM produced a very small net score improvement by finding some targets earlier,
but ranking precision fell. At this scale, the gain is too small to establish that
the LLM path will transfer to the hidden set.

## Routing Diagnostics

- 230 turns routed to the LLM across all 200 sessions.
- All 200 sessions routed on turn 1.
- All 30 intent-override sessions routed a second time on the override turn.
- Turn distribution: turn 1 = 200, turn 3 = 12, turn 4 = 18.
- Scenario distribution: buying = 80, browsing = 80, boundary = 10,
  intent override = 60 calls across 30 sessions.
- Every routed call returned token usage; there were no zero-token API failures.
- Configured state model: `gpt-5.6-luna`.

Routing reasons overlap when one turn has multiple risk signals:

| Reason | Count |
| --- | ---: |
| `outside_safe_parse_allowlist` | 107 |
| `mixed_buying_browsing_signals` | 90 |
| `correction_or_override` | 32 |
| `unresolved_reference` | 31 |
| `category_looks_like_attribute` | 22 |

The current gate is therefore not selective in practice. It treats every initial
request as ambiguous, including templated buying and browsing requests that the
deterministic parser already handles reliably.

## Regressions

Eight sessions scored worse with the LLM. None became a complete miss.

| Sample | Scenario | Deterministic | Gated LLM | Routed turns |
| --- | --- | --- | --- | --- |
| `public_0189` | buying | T3 R1 | T5 R2 | T1 |
| `public_0072` | intent override | T4 R1 | T4 R2 | T1, T3 |
| `public_0080` | intent override | T5 R1 | T4 R2 | T1, T4 |
| `public_0183` | intent override | T5 R1 | T4 R2 | T1, T4 |
| `public_0042` | buying | T4 R1 | T2 R2 | T1 |
| `public_0002` | intent override | T5 R3 | T5 R6 | T1, T3 |
| `public_0001` | buying | T4 R2 | T3 R3 | T1 |
| `public_0133` | buying | T4 R1 | T5 R1 | T1 |

`public_0080`, `public_0183`, `public_0042`, and `public_0001` demonstrate the
evaluator's rank-versus-turn trade-off: an earlier target can still score worse if
the first observed rank falls from R1/R2 to R2/R3.

## Improvements

Fifteen sessions improved and 177 were unchanged. The largest improvements were:

| Sample | Scenario | Deterministic | Gated LLM |
| --- | --- | --- | --- |
| `public_0026` | buying | T5 R2 | T2 R1 |
| `public_0028` | buying | T5 R2 | T5 R1 |
| `public_0038` | intent override | T5 R8 | T5 R2 |
| `public_0022` | buying | T2 R2 | T4 R1 |
| `public_0103` | intent override | T4 R5 | T4 R2 |
| `public_0096` | intent override | T9 R4 | T4 R5 |
| `public_0198` | intent override | T5 R9 | T5 R3 |

The LLM helped several difficult override sessions, particularly `public_0038`,
`public_0096`, `public_0103`, and `public_0198`. This supports keeping an LLM path
for genuinely ambiguous state replacement, but not calling it on every first turn.

## Remaining Weak Sessions

The two complete misses are unchanged from the deterministic run:

- `public_0144`: intent override, miss after two LLM routes.
- `public_0175`: browsing, miss after one LLM route.

Other weak outcomes in the LLM run, defined as rank 7-10 or turn 8-10:

- `public_0076`: T5 R8
- `public_0081`: T5 R10
- `public_0083`: T7 R8
- `public_0087`: T5 R8
- `public_0126`: T8 R8
- `public_0137`: T5 R7
- `public_0178`: T5 R9

These cases were not created by the LLM; they were already weak under the
deterministic baseline.

## Recommendation

Do not ship the current gate unchanged. Keep deterministic interpretation for
structured initial buying/browsing templates and direct clarification answers.
Reserve the LLM for high-risk transitions such as ambiguous correction scope,
product-versus-preference replacement, unresolved references, and conflicting
state updates. This retains the production value of semantic state interpretation
without paying 318,473 tokens to alter only 23 of 200 outcomes.

Artifacts:

- `docs/evaluations/ct/results/public-deterministic.json`
- `docs/evaluations/ct/results/public-broad-gate-llm.json`
