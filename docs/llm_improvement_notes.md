# LLM Improvement Notes

These are production-oriented improvements to make when an LLM is available. The current lightweight path keeps deterministic rule-based fallback logic, but the LLM should eventually own the ambiguous interpretation work.

## Intent And Override Understanding

- Classify each turn into `buying`, `browsing`, `intent_override`, or `unknown`.
- For override turns, classify the override type:
  - `attribute_change`: user is changing one preference, such as material, color, budget, size, style, or use case.
  - `new_product_intent`: user is abandoning the old item and searching for a different product.
- Preserve category/context for attribute-only overrides.
  - Example: "Actually, what I need is wool" should preserve "leg warmers" and replace material with wool.
- Fully reset product-specific state for new product intent.
  - Clear category, preferences, removed preferences, no-preference attributes, search terms, asked attributes, previous asked attribute, and latest recommendations.
- Return structured output like:

```json
{
  "intent_mode": "buying",
  "override_type": "attribute_change",
  "changed_attributes": ["material"],
  "category": "unchanged",
  "set_preferences": [{"attribute": "material", "value": "wool"}],
  "remove_preferences": [{"attribute": "material"}],
  "reset_product_preferences": false,
  "search_terms": ["wool"]
}
```

## Preference Extraction

- Extract only confirmed preferences from the current shopping conversation.
- Do not convert general user-profile traits into hard product requirements.
- Keep profile preferences separate as soft ranking hints.
- Normalize values consistently:
  - material: `leather`, `cotton`, `wool`
  - color: `black`, `red`, `grey`
  - budget: `under $50`, `around $100`
  - use case: `running`, `hiking`, `work`
- Track negative preferences separately from positive preferences.
- Understand no-preference replies:
  - "I don't care about color" should clear active color preference and mark color as no-preference.
  - Do not penalize products for no-preference attributes.

## Clarifying Question Selection

- Use retrieval candidate diagnostics before asking the next question.
- Hard-filter question attributes that are:
  - already asked
  - already answered
  - marked no-preference
  - already known from category or preferences
- Score remaining attributes by:
  - candidate disagreement
  - evidence coverage
  - expected answer usefulness
  - category/scenario relevance
- Prefer specific high-value questions over generic ones.
- Avoid asking `feature` just because every product has feature text.
- Treat generic catalog metadata as low-value:
  - `Imported`
  - `Pull On closure`
  - `Button closure`
  - `Machine Wash`
  - `Rubber sole`
  - `100% Cotton` when common across many candidates
- Use broad "other" questions only as a fallback or late-stage recovery, not as a fixed early-turn rule.

## Retrieval And Reranking With LLM Support

- The LLM should not search all products directly.
- Use the LLM to produce structured query state, then use local retrieval:
  - BM25 / SQLite FTS
  - vector search
  - hybrid candidate fusion
- Prefer several targeted lexical routes over one broad query:
  - category-only route
  - title/category route
  - full-state route
  - feature/details route
  - latest-message route
  - exact-phrase route
  - relaxed route
- Use vector search as tail recall, not as a dominant ranking signal:
  - retrieve a small vector tail, such as top 30
  - merge those candidates into the rerank pool
  - keep the vector route weight low so lexical evidence remains primary
- Use LLM or a lightweight cross-encoder reranker only on a small candidate set.
- Reranker should distinguish:
  - hard confirmed requirements
  - soft profile preferences
  - generic metadata
  - exact title/style matches
  - explicit negative preferences
- When candidates are in the same category, boost distinctive title/style terms more than generic feature boilerplate.

## Evaluation Risks

- Do not hardcode public sample IDs or target product IDs.
- Avoid evaluator-specific rules such as fixed early `ask_attribute = "other"`.
- Prefer rules that would also make sense for real shoppers.
- Keep deterministic fallback behavior when the LLM is unavailable.
- Preserve token usage accounting when an LLM call succeeds or fails.
