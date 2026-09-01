# Catalog-Grounded Scenario-Query Probe

## Scope

Five actual catalog products were selected without public-session labels. The
probe is not a purchased-item benchmark: a short broad request can match many
reasonable products. It tests that the optional LLM scenario route is safe and
adds useful catalog recall before it is ever enabled for submission.

The design keeps the three retrieval inputs separate:

```text
category_query  = explicit state category only
feature_query   = confirmed preferences and explicit search terms only
scenario_query  = temporary LLM-inferred functional vocabulary only
```

The scenario route searches feature/detail embeddings with a small separate RRF
signal. It never mutates active preferences, search terms, or profile memory. A
country or city alone is not actionable scenario evidence: it does not prove a
season, climate, or required feature.

Run with:

```bash
set -a
source .env.example
source ../fable-model-techjam/.env
set +a
OPENAI_ENABLED=true \
TECHJAM_QUERY_EXPANSION_ENABLED=true \
TECHJAM_QUERY_EXPANSION_MODE=recall \
../techjam-conversational-search/.venv/bin/python \
  -m tools.exercise_query_expansion_catalog_cases \
  --catalog ../techjam-conversational-search/data/catalog.jsonl
```

The latest machine-readable trace is
`results_query_expansion_catalog_cases_v4_live.json`.

## Results

| Request | Scenario behavior | Catalog outcome | Verdict |
| --- | --- | --- | --- |
| `good men's jacket for Canada` | No LLM scenario route | The selected jacket did not enter candidate recall | Correct safety behavior; country alone is under-specified, but no recall gain |
| `appropriate men's shoes for wet weather` | `waterproof traction` from basis `wet weather` | Intended Columbia shoe is rank 232 in the feature-vector index, beyond Top 30 | Correct temporary inference; insufficient vector recall |
| `good laptop backpack for commuting` | `padded sleeve comfortable carrying` from basis `commuting` | Intended Oakley backpack is rank 320 in the feature-vector index; candidate rank moved 67 to 70 | Correct temporary inference; no ranking lift |
| Detailed cotton shelf-bra camisole | No LLM scenario route | Returned rank 1 | Correct deterministic bypass |
| `model 5006715` | No LLM scenario route | Returned rank 1 | Correct exact-identifier bypass |

## What Passed

- Explicit actionable scenarios yielded clean, temporary LLM outputs:
  `waterproof traction` and `padded sleeve comfortable carrying`.
- The country-only request did not infer winter/weather features.
- Inferred scenario terms never entered confirmed preferences or `search_terms`.
- Detailed and exact-identifier requests bypassed the LLM and returned rank 1.

## What Did Not Pass

The feature did **not** demonstrate a product-level recall or ranking gain on
these broad catalog probes. Enlarging the scenario tail to reach feature-vector
ranks 232 and 320 would admit many weak products; it is not a defensible
submission optimization without a separately labeled scenario-relevance set.

## Decision

Keep `TECHJAM_QUERY_EXPANSION_ENABLED=false` for submission. The refactor made
the experimental route production-safer and inspectable, but it did not show a
recall or ranking lift. With expansion disabled, the full public offline run
reproduced `0.911818` exactly: Hit@10 `0.990`, MRR `0.893728`, MTTC `3.565`.
