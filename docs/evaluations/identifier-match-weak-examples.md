# Exact Identifier Weak-Set Replay

This replay tests the exact identifier promotion added for labeled product IDs such as `Item model number: 5006715`, `model number`, `style number`, `part number`, `mpn`, and `sku`.

Run configuration:

```text
OPENAI_ENABLED=false
TECHJAM_VECTOR_ENABLED=true
TECHJAM_RERANK_ENABLED=true
TECHJAM_RERANK_WEIGHT=0.65
TECHJAM_RERANK_TOP_N=10
```

The replay uses the same 10 sessions listed in `weak-public-set-examples.md`, not the full 200-session public set.

## Summary

| Session | Scenario | Before `0.851142` | After Description + Identifier | Approx Delta |
|---|---:|---:|---:|---:|
| `public_0144` | intent_override | miss | miss | +0.0000 |
| `public_0154` | buying | miss | hit turn 2, rank 3 | +0.7800 |
| `public_0174` | buying | miss | miss | +0.0000 |
| `public_0175` | browsing | miss | miss | +0.0000 |
| `public_0198` | intent_override | hit turn 9, rank 7 | hit turn 9, rank 9 | -0.0095 |
| `public_0161` | buying | hit turn 9, rank 4 | hit turn 9, rank 4 | +0.0000 |
| `public_0126` | browsing | hit turn 6, rank 9 | hit turn 7, rank 9 | -0.0200 |
| `public_0035` | boundary | hit turn 5, rank 8 | hit turn 5, rank 8 | +0.0000 |
| `public_0087` | browsing | hit turn 5, rank 8 | hit turn 5, rank 8 | +0.0000 |
| `public_0137` | browsing | hit turn 6, rank 5 | hit turn 10, rank 1 | +0.1600 |

## Interpretation

The weak-set table is unchanged from the description-match replay. That is expected: the main affected case, `public_0154`, now hits on turn 2 from `cotton` + `white` description matching, before the simulator reveals `Item model number: 5006715` on turn 3.

The identifier logic is still worth keeping if the full eval is neutral or positive, because it handles a real production retrieval pattern:

```text
When a user supplies a labeled model/SKU/part number, exact catalog matches should be promoted above generic semantic matches.
```

Implementation behavior:

```text
1. Extract only labeled identifiers, not arbitrary numbers.
2. Add an `identifier` recall route over title/features/details/description.
3. Add a strong `exact_identifier` score component when the normalized ID appears in product corpus.
4. Keep other candidates available; this is a strong boost, not a hard filter.
```

Focused unit coverage verifies that `Item model number: 5006715` is extracted, ordinary budget/date numbers are ignored, and a matching model-number product outranks a generic cotton/white product.
