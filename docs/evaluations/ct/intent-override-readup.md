# Intent Override Readup

## What the Competition Scenario Means

The competition specification defines Intent Override as a session where an
earlier preference is replaced on turn 3 or 4. It accounts for 15% of both the
public and hidden sets.

In the supplied local evaluator, the target product does not change. The
simulator builds both values from the same target product:

- `old_value` is one of the target's soft preferences;
- `new_value` is the target's first hard constraint; and
- the override message is:
  `Actually, ignore my earlier preference. What I need is: <new_value>.`

The evaluator refuses to score a target recommendation before this override is
sent. Once the override has been sent, ordinary hit/rank scoring begins.

Therefore, the supplied Intent Override scenario is primarily a **same-product
preference correction**, not a complete change to another product category.

## Why the Old Preference Can Still Match the Product

Both old and new values are generated from the target's metadata. For
`public_0003`, the target is `Casio Men's Wrist Watch AQ-800E-7A` and both of
these are real target features:

- old preference: `Stainless Steel Band`;
- replacement preference: `Water Resistant`.

The simulator is testing whether the agent follows the customer's latest stated
intent, not whether the old value is factually false of the target. Retaining
the old value may accidentally help retrieve this particular target, but it is
incorrect conversational state and could hurt hidden sessions by keeping stale
constraints active.

## The Three Required State Transitions

### 1. Ordinary Merge

The shopper adds information without superseding prior state.

```text
Current: women's boots, black
Message: "Leather, please."
Result: women's boots, black, leather
```

Keep category, active preferences, question history, and search context. Add
only the new supported evidence.

### 2. Same-Product Preference Correction

The shopper rejects or replaces an earlier preference but continues shopping
for the same product type.

```text
Current: watches; Stainless Steel Band
Message: "Ignore my earlier preference. I need Water Resistant."
Result: watches; Water Resistant
```

Expected behavior:

- preserve the current product category;
- add the replacement value under its canonical attribute;
- retire the explicitly rejected value;
- when the old value is referenced only as "earlier preference", retire the
  latest active unsolicited preference evidence;
- preserve unrelated clarification-confirmed evidence unless it conflicts;
- preserve the aggregate user profile;
- invalidate recommendations ranked with stale evidence.

This transition is represented by `replace_preferences` with one of two explicit
correction scopes:

- `corrected_attributes`: replace only attributes named in the patch;
- `latest_unsolicited`: also retire exactly the latest active unsolicited
  preference evidence when the shopper refers to it indirectly.

### 3. Complete Product Change

The shopper names a genuinely different product type.

```text
Current: shirts; cotton; blue
Message: "Actually, show me waterproof hiking boots instead."
Result: hiking boots; waterproof; hiking
```

Clear all product-specific state before applying the new intent:

- category;
- active and removed preferences;
- no-preference markers;
- search terms and evidence ledger;
- asked-attribute history;
- prior recommendations.

Keep only the aggregate user profile and runtime/session information. This
transition is represented by `product_change`.

## Practical Classification Rules

Use deterministic evidence before asking an LLM:

```text
No correction cue
    -> merge

Correction cue + explicit new product noun incompatible with current category
    -> product_change

Correction cue + attribute/feature value, no conflicting product noun
    -> replace_preferences

Correction cue but unresolved reference and no safe replacement
    -> constrained LLM choice or clarification
```

Words such as `actually`, `instead`, and `ignore` are not enough by themselves
to prove a product change. The new product noun is the decisive evidence.

## Correct Handling of `public_0003`

Before the override:

```json
{
  "category": "watches wrist watches",
  "preferences": {
    "material": ["stainless steel band"]
  },
  "no_preference_attributes": ["color"],
  "previous_ask_attribute": "feature"
}
```

Message:

```text
Actually, ignore my earlier preference. What I need is: Water Resistant.
```

Correct transition:

```text
replace_preferences
+ add feature = water resistant
+ retire latest unsolicited preference = stainless steel band
+ keep category = watches wrist watches
+ keep no-preference color unless the correction conflicts with it
```

Result:

```json
{
  "category": "watches wrist watches",
  "preferences": {
    "feature": ["water resistant"]
  },
  "no_preference_attributes": ["color"]
}
```

Before the scoped transition was added, the LLM added `Water Resistant` but
preserved `Stainless Steel Band`. The tool now emits the typed
`correction_scope`, validates `latest_unsolicited` against explicit indirect
correction wording, and deterministically retires the corresponding evidence.
The model chooses the transition; it does not directly mutate retrieval state.

## Recommended Production Architecture

```text
message
  -> deterministic parser and transition candidates
  -> optional constrained LLM for unresolved correction scope
  -> deterministic evidence validation
  -> deterministic transition application
  -> canonical retrieval state
```

The LLM should choose among safe transition candidates. It should not directly
decide how retrieval fields are populated or whether unsupported state is
deleted.

## Competition-Specific Cautions

1. A target returned before the override is not scored as a hit.
2. The first scored hit ends the session, so a low-ranked hit cannot improve on
   a later turn.
3. Do not treat every correction phrase as a complete product reset.
4. Do not retain stale evidence merely because it also happens to match the
   target product.
5. Do not hardcode the evaluator's sentence template; hidden sessions may use
   paraphrases even though exact product IDs remain the scoring truth.

## Relevant Files

- `docs/competition_specification.md`: scenario mix and session protocol.
- `evaluator/local_evaluator.py`: exact public simulator behavior.
- `starter/preference_tool.py`: transition classification and application.
- `starter/state.py`: `merge`, `replace_preferences`, and `product_change` state
  types.
- `docs/superpowers/specs/2026-08-27-selective-preference-evidence-design.md`:
  intended evidence-retirement semantics for same-product corrections.
