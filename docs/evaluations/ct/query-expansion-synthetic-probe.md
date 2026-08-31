# Scenario Query-Expansion Synthetic Probe

## Purpose

This is an adversarial functionality probe, not a relevance benchmark. The
messages below have no hidden purchased item. It verifies that broad goals may
obtain temporary semantic recall hypotheses, while precise evidence remains on
the deterministic path.

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
  -m tools.exercise_query_expansion \
  --catalog ../techjam-conversational-search/data/catalog.jsonl
```

The full machine-readable trace is ignored at
`results_query_expansion_synthetic.json`.

## Observed Behavior

| Case | Expected | Observed | Result |
| --- | --- | --- | --- |
| `good jacket for Canada` | LLM plus temporary climate recall | One hypothesis: warmth and weather resistance | Pass; added 14 candidates |
| `appropriate shoes for wet weather` | LLM plus temporary functional recall | One hypothesis: water-resistant or waterproof shoes | Pass; added 15 candidates |
| `good backpack for commuting` | LLM plus temporary commuter recall | One hypothesis: commuter backpack | Pass; added 9 candidates |
| `black leather boots in size 10` | Deterministic exact-evidence path | No LLM call | Pass |
| Detailed cotton camisole request | Deterministic catalog-evidence path | No LLM call | Pass |
| `shirt for Russia` | LLM may choose a scenario expansion | LLM made a safe state update but emitted no hypothesis | Acceptable conservative abstention |
| `good pair of shoes for a rainy commute` | Scenario expansion would be useful | Reached the normal ambiguity LLM path, but did not receive the expansion prompt | Gate false-negative |
| `blue shirt for work under 30 dollars` | Deterministic structured parse preferred | Reached normal ambiguity LLM path | Existing parser coverage gap; unrelated to expansion |
| `model AB-1234` | Exact identifier fast path preferred | Recognized, normalized, and routed deterministically | Fixed: shorthand requires an ID-shaped token containing a digit |

## Safety Checks That Passed

- Hypotheses were stored only in the per-turn diagnostic and supplied only to
  the `vector_expansion` recall route.
- Generated terms such as `water-resistant`, `waterproof`, `warmth`, and
  `commuter` did **not** enter confirmed preferences or `search_terms`.
- The expansion route has zero direct RRF weight. It adds candidates; the
  existing ranking, local cross-encoder, and guardrails still choose the final
  order.
- No expansion was attempted for precise product evidence or lengthy catalog
  attribute descriptions.

## Conclusion

The mechanism works as intended for genuinely broad product goals: it adds
semantic recall without letting the LLM manufacture durable constraints or
catalog identifiers. It is intentionally disabled by default because the
public benchmark does not contain enough natural scenario language to validate
the effect on hidden targets.

Before enabling it in a production or demo configuration, improve the detector
to normalize common request prefixes such as `I need` and `I want`. Keep the
same strict first-turn, no-identifier, no-override, temporary-only boundary.
The exact-lookup grammar independently supports both `model number AB-1234`
and the natural shorthand `model AB-1234`, while rejecting ordinary phrases
such as `model shirt`.
