# Iteration 2 Adaptive Lexical Agent Comparison

Date: 2026-08-27

## Outcome

The selective-evidence revision passes every offline acceptance gate. It improves
intent-override Hit@10 from `0.666667` to `0.833333` while also improving the
overall score; scenario Hit@10 either improved or held steady. No OpenAI API
calls were made for this revision; the paid comparison remains pending an
explicit spend decision.

### Offline comparison

| Branch | Score | Hit@10 | MRR | MTTC | Intent override Hit@10 |
|---|---:|---:|---:|---:|---:|
| Main baseline | 0.638829 | 0.805 | 0.440097 | 5.785 | 0.433333 |
| Claude adaptive branch | 0.655705 | 0.795 | 0.486016 | 5.380 | 0.366667 |
| Codex adaptive branch | 0.714761 | 0.870 | 0.507536 | 4.625 | 0.666667 |
| Verified pre-change branch | 0.719486 | 0.875 | 0.502952 | 4.445 | 0.666667 |
| Selective-evidence revision | **0.784771** | **0.940** | **0.559571** | **3.655** | **0.833333** |

### Offline scenario comparison

| Scenario | Pre-change Hit@10 | New Hit@10 | Pre-change MRR | New MRR | Pre-change MTTC | New MTTC |
|---|---:|---:|---:|---:|---:|---:|
| Boundary | 1.0000 | 1.0000 | 0.696111 | 0.612778 | 5.3000 | 4.5000 |
| Browsing | 0.9125 | 0.9625 | 0.523239 | 0.586830 | 3.6625 | 3.2000 |
| Buying | 0.9000 | 0.9500 | 0.497649 | 0.561409 | 4.0875 | 3.0250 |
| Intent override | 0.666667 | 0.833333 | 0.398611 | 0.464246 | 7.2000 | 6.266667 |

The five remaining intent misses are all in the hard clothing bucket
(`public_0071`, `public_0080`, `public_0096`, `public_0144`, and
`public_0177`). The gain therefore comes from stronger session evidence rather
than a global candidate-limit or ranking-weight increase.

### Latest verified OpenAI comparison (before selective evidence)

| Branch | Score | Hit@10 | MRR | MTTC | Intent override Hit@10 |
|---|---:|---:|---:|---:|---:|
| Main baseline | 0.669633 | 0.850 | 0.441111 | 5.385 | 0.566667 |
| Codex adaptive branch | 0.706291 | 0.860 | 0.473302 | 4.285 | 0.666667 |
| Verified pre-change branch | 0.728769 | 0.890 | 0.469897 | 3.860 | 0.633333 |

The selective-evidence revision has not been evaluated in paid mode. Its
offline run reported zero prompt and completion tokens.

The verified pre-change OpenAI artifact was generated on 2026-08-26, before
the selective-evidence implementation, using `gpt-5.6-luna` and the documented
OpenAI-enabled command below. It reported 320,930 prompt tokens and 85,587
completion tokens. It is retained only as the paid baseline; no paid request
was made while producing the 2026-08-27 selective-evidence result.

## Root causes found in the first adaptive implementation

1. The apparent 0.85 to 0.795 drop compared an OpenAI-enabled run with an offline run. In an offline-to-offline comparison, the first adaptive branch improved score and MRR, but lost recall.
2. Candidate admission stopped as soon as 500 unique IDs had been encountered. Because the latest-message route ran last, relevant products from that route could be discarded before fusion.
3. Tail diversity could replace products already ranked 7-10 with products ranked below the Top 10, directly reducing Hit@10.
4. Unstructured feature text created almost-unique signatures for every product, making the adaptive policy ask about features before more reliable material and color attributes.
5. Preference-override messages could replace the product category with an attribute such as `nylon` or `rubber sole`.
6. The LLM returned compound values such as `95% polyester, 5% spandex`, while deterministic parsing returned canonical atomic values. Literal compound matching made LLM retrieval less robust.

## Fixes on the Codex branch

- Fuse every route before retaining the best 500 candidates.
- Keep the strict reranked Top 10 instead of ejecting tail candidates for route diversity.
- Use controlled attribute lexicons for candidate diagnostics.
- Favor material and color when unstructured feature evidence is noisy, while still allowing strong feature evidence to win.
- Preserve the current category for preference-only overrides and clear stale question/recommendation context.
- Distinguish stale state from newly confirmed override values during reranking.
- Canonicalize compound material, color, and use-case values before storing them.
- Track normalized preference evidence as unsolicited, clarification-confirmed,
  or corrective session state.
- Preserve nonconflicting clarification evidence during same-product
  corrections while retiring superseded unsolicited evidence.
- Convert bounded evaluator-style clarification replies into structured values
  and suppress conversational boilerplate from fallback search terms.

## Reproduction

Offline:

```bash
OPENAI_ENABLED=false .venv/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output docs/evaluations/iteration2-offline-results.json
```

OpenAI enabled, after loading `OPENAI_API_KEY` from the ignored local `.env`:

```bash
OPENAI_ENABLED=true .venv/bin/python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output docs/evaluations/iteration2-openai-results.json
```
