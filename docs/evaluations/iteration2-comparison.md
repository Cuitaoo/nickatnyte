# Iteration 2 Adaptive Lexical Agent Comparison

Date: 2026-08-26

## Outcome

The Codex comparison branch improves the evaluator score in both deterministic fallback mode and OpenAI-enabled mode.

### Offline comparison

| Branch | Score | Hit@10 | MRR | MTTC | Intent override Hit@10 |
|---|---:|---:|---:|---:|---:|
| Main baseline | 0.638829 | 0.805 | 0.440097 | 5.785 | 0.433333 |
| Claude adaptive branch | 0.655705 | 0.795 | 0.486016 | 5.380 | 0.366667 |
| Codex adaptive branch | 0.714761 | 0.870 | 0.507536 | 4.625 | 0.666667 |

### OpenAI-enabled comparison

| Branch | Score | Hit@10 | MRR | MTTC | Intent override Hit@10 |
|---|---:|---:|---:|---:|---:|
| Main baseline | 0.669633 | 0.850 | 0.441111 | 5.385 | 0.566667 |
| Codex adaptive branch | 0.706291 | 0.860 | 0.473302 | 4.285 | 0.666667 |

The final OpenAI run reported 354,189 input tokens and 91,761 output tokens. Using the previously verified `gpt-5.6-luna` prices of $0.20 per million input tokens and $1.20 per million output tokens, its estimated direct cost is approximately $0.181.

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

