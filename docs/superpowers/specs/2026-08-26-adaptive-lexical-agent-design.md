# Iteration 2: Adaptive Lexical Shopping Agent Design

**Date:** 2026-08-26  
**Status:** Approved for implementation planning

## 1. Objective

Improve the conversational shopping agent's evaluator score without hardcoding public targets or depending on evaluator-specific behavior. The iteration keeps the existing one-call LangChain preference interpreter and replaces fixed clarification and overlapping retrieval behavior with an adaptive lexical pipeline.

The current OpenAI-enabled baseline is:

| Metric | Current | Iteration 2 target |
|---|---:|---:|
| Overall score | 0.6696 | >= 0.72 |
| Hit@10 | 0.850 | >= 0.90 |
| MRR | 0.441 | >= 0.48 |
| Mean turns to success | 5.385 | < 4.8 |
| Intent-override Hit@10 | 0.567 | >= 0.70 |
| Boundary Hit@10 | 0.700 | >= 0.70 |

## 2. Scope and constraints

### In scope

- Adaptive clarification-question selection based on the current candidate set.
- Several intentionally different BM25/FTS candidate-retrieval routes.
- Candidate fusion and attribute-aware reranking.
- Strict separation between current-session requirements and general user-profile tendencies.
- Complete product-state reset when the shopper changes intent.
- A final Top 10 that preserves high-confidence ordering while adding limited route diversity.
- Deterministic fallback behavior when the model is unavailable.

### Out of scope

- Embedding or vector search.
- A second model call for search, reranking, or question generation.
- Target-ID rules, public-test memorization, or special reliance on the evaluator's handling of `ask_attribute="other"`.
- A user-facing application or chat interface.
- Changes to the evaluator contract.

## 3. High-level flow

Each `respond()` turn follows this sequence:

1. The existing LangChain model call interprets only the conversation and produces a validated preference update.
2. The runtime state applies the update, including intent-reset semantics when requested.
3. The retriever runs several distinct lexical searches and merges their unique candidates.
4. The ranker scores candidates using current hard preferences, negative preferences, lexical evidence, and a small profile tie-breaker.
5. Candidate diagnostics estimate which unanswered attribute would most usefully narrow the results.
6. A deterministic policy chooses the next clarification attribute and renders a template question.
7. The agent returns the question and a catalog-valid Top 10.

The model does not read or search all 50,000 products. Product retrieval and ranking remain local and deterministic.

## 4. Conversation state and preference strength

State must distinguish these concepts:

### 4.1 Confirmed product preferences

Requirements stated or confirmed in the current shopping session, including category, features, use case, material, color, size, style, brand, and budget. These are strong ranking signals. Explicit negative preferences apply strong penalties.

Only current conversation messages may create confirmed product preferences. General profile text must never be included in model input in a way that lets the model promote it into a hard requirement.

### 4.2 General user profile

Long-term tendencies such as comfort, durability, or stylistic preferences. These remain separate from the active preference map and contribute only a small positive tie-breaker. A profile preference must not filter products, overpower a current-session requirement, or create a negative penalty.

### 4.3 Intent override

When the shopper abandons the current product intent, reset all product-specific state:

- category and confirmed preferences;
- rejected preferences and no-preference markers;
- accumulated search terms;
- asked attributes and the previous asked attribute;
- latest recommendations and product-specific retrieval context.

Preserve only the general user profile and conversation information needed for correct API operation. Retrieval on the reset turn gives additional weight to the latest user message so the previous product cannot dominate.

## 5. Diversified lexical candidate retrieval

The retriever runs routes that emphasize different evidence instead of issuing several near-identical broad queries. Applicable routes are:

1. **Category route:** category and product-type terms, weighted toward category/title fields.
2. **Feature/use-case route:** functional requirements and intended activity, weighted toward features, details, and description fields.
3. **Exact-phrase route:** quoted or tightly joined multiword phrases from the latest message and confirmed preferences.
4. **Attribute route:** material, color, size, style, brand, and other confirmed attribute values.
5. **Relaxed route:** removes low-confidence or overly restrictive terms to recover vocabulary-mismatched candidates.
6. **Latest-message route:** searches the newest shopper message independently, with a boost after an intent override.

Each useful route retrieves approximately 100-150 products. Duplicate product IDs are merged into a union capped at approximately 500 candidates. The implementation may tune these limits using latency and evaluator results, but should keep route identity for fusion and diagnostics.

Routes with no relevant terms are skipped. Search-expression failures fall back to safely escaped or reduced terms rather than failing the turn.

## 6. Candidate fusion and reranking

### 6.1 Fusion

Use reciprocal-rank-style fusion so that:

- a high position in any route is valuable;
- appearing in multiple independent routes increases confidence;
- raw FTS scores from differently shaped queries do not need direct comparison.

The fused result retains route membership and per-route rank for later diagnostics and final-list diversity.

### 6.2 Attribute-aware reranking

Reranking applies signals in this order of importance:

1. Exact category, product-type, and phrase agreement.
2. Confirmed feature and use-case agreement.
3. Confirmed material, color, size, style, brand, and budget agreement.
4. Strong penalties for explicitly rejected attributes or contradictory evidence.
5. Lexical fusion confidence.
6. A small general-profile tie-breaker.

Matches in structured or strongly associated fields should outrank incidental word occurrences. Numeric budget handling must compare available numeric price evidence when present and degrade safely when price data is absent.

Scoring must remain explainable through per-signal contributions in internal diagnostics, although those details do not need to be exposed through the evaluator API.

### 6.3 Final Top 10

Positions 1-6 are filled strictly by the highest reranked confidence. Positions 7-10 may select strong remaining candidates from underrepresented retrieval routes or product interpretations. Diversity selection must have a minimum relevance threshold so an unrelated product is never inserted merely to make the list different.

This balances MRR protection at the top with broader Hit@10 coverage at the bottom.

## 7. Adaptive clarification policy

The question policy is deterministic and consumes diagnostics from the merged candidate pool. It does not require another model call.

Eligible attributes exclude those that are already confirmed, explicitly declined, or already asked during the current product intent. For each eligible attribute, calculate a usefulness score from:

- **candidate disagreement:** whether plausible candidates differ meaningfully on the attribute;
- **coverage:** whether enough candidates contain evidence for the attribute;
- **intent relevance:** whether the attribute commonly distinguishes the current category or is suggested by current search evidence;
- **answer likelihood:** whether the shopper's wording or product context makes a meaningful answer plausible;
- **repeat/no-preference penalties:** attributes already handled receive an exclusion or strong penalty.

Policy rules:

- Ask category first only when product type is genuinely unclear and candidates span incompatible categories.
- Prefer feature, use case, or style when those answers can strongly divide the viable candidates.
- Ask material, color, or size when current candidates show those attributes are decision-relevant.
- Ask brand or budget only when evidence suggests they will narrow the results.
- Use `other` only when no specific eligible attribute has sufficient expected value.
- Always return recommendations alongside the question.

Question wording comes from stable templates so the returned `ask_attribute` and natural-language question cannot contradict one another.

## 8. Failure and fallback behavior

- If OpenAI is unavailable or returns an invalid update, retain the existing deterministic message parser and continue retrieval.
- If an individual search route fails or has no terms, continue with the remaining routes.
- If all focused routes are empty, use a safe broad lexical fallback based on the latest message, then a catalog-safe fallback if necessary.
- Never return an ID outside the frozen catalog.
- Never let stale recommendations or asked attributes survive an intent reset.
- Preserve model usage metadata when a model attempt occurred, including fallback turns.

## 9. Testing strategy

Implementation follows test-driven development. Tests should cover:

### State and model interpretation

- General profile values cannot become hard preferences without confirmation in conversation.
- Intent override clears every product-specific field, including question history and recommendations.
- General profile survives an intent override.
- Invalid or unavailable model output uses deterministic fallback safely.

### Retrieval and ranking

- Each route uses its intended fields and can contribute candidates absent from another route.
- Candidate fusion removes duplicates and rewards multi-route evidence.
- Exact and confirmed attributes outrank incidental lexical matches.
- Negative preferences penalize contradictions.
- Final-list diversity cannot add candidates below the relevance threshold.
- All returned IDs exist in the catalog.

### Clarification

- The policy chooses a high-value feature/use-case question over a low-value color question when candidate evidence supports that decision.
- Asked, answered, and declined attributes are not repeated.
- Category is not asked when already clear.
- `other` is used only as a fallback.
- Intent override makes previously asked attributes eligible for the new product intent.

### Evaluation

Run fast unit and integration tests first. Then run deterministic/public evaluator comparisons and record overall, per-scenario, and per-turn metrics. A full OpenAI-enabled evaluation is performed only after local evidence shows improvement. Compare results to the fixed baseline in Section 1 and inspect regressions even when the aggregate score rises.

## 10. Acceptance criteria

The iteration is ready to merge when:

- all existing and new tests pass;
- no public target IDs or evaluator-specific target rules are introduced;
- intent reset and profile-strength behavior are verified;
- evaluation results and configuration are recorded reproducibly;
- overall performance moves toward the Section 1 targets without dropping boundary Hit@10 below 0.70;
- any material tradeoff between Hit@10, MRR, and latency is documented before merge.

