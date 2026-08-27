# Selective Preference Evidence for Intent Corrections

**Date:** 2026-08-27  
**Status:** Approved concept; pending written-spec review

## 1. Objective

Improve intent-override retrieval without weakening ordinary browsing, buying, or boundary behavior. The agent must distinguish a correction to one preference within the same product search from a genuine switch to a different product intent.

The verified offline baseline is:

| Metric | Baseline |
|---|---:|
| Overall technical score | 0.719486 |
| Hit@10 | 0.875 |
| MRR | 0.502952 |
| MTTC | 4.445 |
| Intent-override Hit@10 | 0.666667 |
| Intent-override MRR | 0.398611 |
| Intent-override MTTC | 7.2 |

The current flat-state reset removes useful confirmed answers together with the preference being corrected. Broad remaining evidence such as `shirts + cotton` is insufficient to identify many products. Attempts to compensate through global route weighting or candidate recovery reduced the aggregate score and were reverted.

## 2. Scope and constraints

### In scope

- Track the source and lifecycle of product evidence within a session.
- Distinguish preference corrections from complete product changes.
- Preserve clarification-confirmed evidence during a preference correction.
- Replace conflicting evidence instead of appending incompatible values.
- Treat direct clarification answers as structured, strong evidence.
- Preserve asked/no-preference history for the same product intent.
- Keep the existing one-model-call-per-turn contract and deterministic fallback.

### Out of scope

- Embeddings, vector search, or a second model call.
- Public target IDs, target-specific rules, or evaluator changes.
- Global route-limit increases or global ranking-weight changes.
- Long-term storage of conversation evidence beyond the active session.
- Exposing evidence provenance through the public `Agent` API.

## 3. Evidence model

The runtime state will retain the current flattened preference maps for retrieval compatibility and add an internal ordered evidence ledger. Each evidence record contains:

- normalized attribute;
- one or more normalized values or lexical terms;
- source turn;
- source kind: `unsolicited`, `clarification`, or `correction`;
- active/inactive status or equivalent immutable replacement semantics.

`unsolicited` evidence comes from a shopper message that was not a direct answer to the agent's previous clarification question. `clarification` evidence comes from an answer to `previous_ask_attribute`. `correction` evidence comes from an explicit preference-only override.

The general user profile remains separate and never creates hard evidence.

The flattened `preferences`, `removed_preferences`, `no_preference_attributes`, and `search_terms` remain the inputs consumed by retrieval. They are derived or updated consistently with the active evidence ledger; the retriever does not need to understand conversational provenance.

## 4. Update classification

Each interpreted patch is classified before it mutates state:

### 4.1 Ordinary update

No explicit reset. New evidence is tagged as `clarification` when it answers `previous_ask_attribute`; otherwise it is `unsolicited`. Existing state remains active unless the patch explicitly removes or declines a value.

### 4.2 Preference correction

The message requests a reset but supplies an attribute value while retaining the current product category. Examples include “ignore that preference; I need cotton” and “actually, make it blue.”

For a correction:

- preserve the category;
- preserve evidence confirmed through clarification questions unless it conflicts with the correction;
- replace active values for every corrected attribute rather than append to them;
- retire earlier unsolicited evidence identified as the superseded preference;
- preserve asked attributes and no-preference markers for unaffected attributes;
- clear latest recommendations because they were ranked under stale evidence;
- keep the general profile unchanged.

If the wording does not identify a specific earlier value, the correction replaces the corrected attribute and retires the latest active unsolicited product-preference evidence. It does not erase later clarification-confirmed answers.

### 4.3 Product change

A reset with a new product category, or a bare reset without a usable attribute correction, is a complete product change. It clears category, all product evidence, preference maps, search terms, asked/no-preference history, and recommendations before applying the new patch. This preserves the complete-reset behavior required by the Iteration 2 design for genuine intent changes.

## 5. Clarification-answer handling

When the agent previously asked a specific attribute and the shopper provides a positive answer, the deterministic parser and the model tool path must attach the returned values to that attribute.

Examples:

- Asked `material`, reply “cotton with some polyester” → confirmed material values.
- Asked `feature`, reply “button closure and machine washable” → confirmed feature evidence.
- Asked `use_case`, reply “winter hiking” → confirmed use-case evidence.

Boilerplate such as “for that, what matters is” is not stored as search evidence. A no-preference reply records the attribute as declined and adds no lexical search terms.

The interpreter remains responsible for semantic extraction when available. The deterministic fallback must cover the evaluator-style direct-answer shape without attempting open-ended natural-language understanding.

## 6. Retrieval and ranking behavior

No new global retrieval route or ranking weight is introduced. Existing category, feature/use-case, phrase, attribute, relaxed, and latest-message routes continue unchanged.

The benefit comes from higher-quality active state:

- valid clarification evidence survives a preference correction;
- corrected attributes contain only the replacement values;
- generic conversational boilerplate does not dilute relaxed queries;
- confirmed feature phrases receive the existing strong preference and exact-phrase treatment.

This avoids the candidate displacement observed when a combined recovery route was added globally or persisted across override turns.

## 7. Failure and fallback behavior

- Invalid model output still uses deterministic parsing and preserves reported usage.
- Evidence updates are immutable with the rest of `ShoppingState`.
- Unknown attributes are ignored exactly as today.
- If provenance is absent in an older or synthetic state, it behaves as an empty ledger.
- A correction that cannot be classified safely falls back to the existing complete-reset behavior rather than retaining potentially stale product context.
- Retrieval failure continues to return deterministic catalog-valid fallback products.

## 8. Testing strategy

Implementation follows red-green-refactor cycles. Tests must cover:

- clarification evidence records its attribute, turn, and source kind;
- a preference correction preserves clarification-confirmed evidence;
- the corrected attribute replaces incompatible prior values;
- superseded unsolicited evidence is retired;
- asked and declined attributes remain stable for the same product intent;
- a genuine category change still clears every product-specific field;
- direct feature answers become structured feature evidence;
- no-preference replies add no search noise;
- model and fallback paths produce equivalent state semantics;
- existing API signatures, one-call behavior, usage accounting, and catalog-ID guarantees remain unchanged.

After the complete unit suite passes, run the deterministic 200-session evaluator. Do not run the OpenAI-backed evaluator unless the offline result satisfies all acceptance gates.

## 9. Acceptance gates

The change may proceed to the OpenAI evaluation only when:

- all tests pass and `git diff --check` is clean;
- intent-override Hit@10 is greater than 0.666667;
- intent-override MRR does not fall below 0.398611;
- overall technical score exceeds 0.719486;
- boundary Hit@10 remains at least 0.90;
- browsing and buying Hit@10 do not fall by more than one case each;
- no target-ID rule, evaluator modification, or secret is present.

The preferred intent-override target is at least 0.80 offline, with 0.867 as the stretch target. If the offline gate passes, run one full OpenAI-enabled evaluation and retain the change only if the final mode-for-mode comparison is an improvement.

## 10. Delivery

The provenance change is committed separately from evaluation artifacts. Final result JSON files and the comparison report are updated only after the accepted code revision is fixed. Task 7 then completes the required verification, independent code review, secret scan, and branch-finishing workflow.
