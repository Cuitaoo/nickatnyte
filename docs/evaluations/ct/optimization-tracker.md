# TechJam Optimization Tracker

This file tracks ranking experiments and potential additions on top of the `0.851142` vector + cross-encoder setup.

## Reference baseline

Clean `0.851142` branch with `.env` exported correctly:

```bash
set -a
source docs/evaluations/ct/baseline-0.851142.env
set +a
```

Evaluator command:

```bash
OPENAI_ENABLED=false python -m evaluator.local_evaluator \
  --catalog /Users/cuitao/Documents/Tiktok\ TechJam/techjam-conversational-search/data/catalog.jsonl \
  --dataset data/public_set.jsonl
```

Baseline full 200-session score:

```text
score=0.851142
hit@10=0.980
mrr=0.705141
mttc=3.520
```

Note: plain `source .env` is not enough for Python subprocesses. Without `set -a`, the evaluator may silently run without vector/cross-encoder settings.

## 1. Popularity / quality score

Tested change:

```text
rating_coef: unchanged
rating_count_coef: 0.020
```

Correct reranker/vector-on full 200-session result:

```text
score=0.853214
hit@10=0.985
mrr=0.677714
mttc=3.130
```

Compared with baseline:

```text
score +0.002072
hit   +0.005
mrr   -0.027427
mttc  -0.390 turns
```

Summary:

- Leads to a small overall score improvement.
- Helps hit rate and finds some items earlier.
- Hurts MRR, especially because popularity can move a relevant target lower.
- Needs tuning before adoption.

Current recommendation: do not apply a strong popularity boost globally. It is more defensible as a small tie-breaker when relevance scores are close, or as a browsing-heavy signal where popular/high-quality products make more product sense.

## 2. Description matching + exact identifier matching

Tested changes:

- Include product `description` when checking material, color, and size preference matches.
- Detect exact labeled identifiers such as `Item model number`, `model number`, `style number`, `part number`, `MPN`, and `SKU`.
- Give products with matching identifiers a strong exact match boost.

Correct reranker/vector-on full 200-session result:

```text
score=0.854438
hit@10=0.985
mrr=0.708794
mttc=3.535
```

Compared with baseline:

```text
score +0.003296
hit   +0.005
mrr   +0.003653
mttc  +0.015 turns
```

Summary:

- Better than the raw popularity boost as a production-valid improvement.
- Improves both hit rate and MRR.
- Uses catalog fields more completely instead of relying on public-set quirks.
- Main risk: description text can add noisy matches, especially in intent-override sessions.

Current recommendation: likely keep, but inspect intent-override regressions before finalizing.

## 3. Earlier `other` clarification

Status: potential addition, not measured yet.

Current behavior:

- The question policy asks specific attributes first.
- `other` is only asked after no specific attribute has enough score.
- In weak examples, this means the agent can spend many turns asking `color`, `brand`, `size`, `style`, `use_case`, and `budget` before asking `other`.

Why this may help:

- Some hard sessions depend on details that do not fit cleanly into the normal attributes.
- Examples include `Item model number`, `Shaft measures approximately...`, `Rubber sole`, `Tie closure`, `Department`, and other catalog metadata.
- Asking `other` earlier can expose these details sooner, which may improve MTTC and recall.

Production-style option:

```text
Ask other earlier when:
- turn >= 4 or 5
- category is already known
- candidate pool is still crowded
- known preferences are mostly generic metadata
- remaining specific attributes have low discrimination
```

This is the most defensible approach. It treats `other` as a fallback when structured facets stop being useful.

Simpler alternatives:

### A. Hardcoded turn rule

```text
if turn >= 5 and other has not been declined:
    ask other
```

Pros:

- Simple.
- Likely exposes hidden details earlier.
- Easy to explain and implement.

Cons:

- May skip useful specific questions.
- More metric-shaped than production-shaped.
- Could hurt easy sessions where `color`, `size`, or `brand` would identify the product.

### B. Ask `other` after N no-preference answers

```text
if shopper has said no preference for 2 or more specific attributes:
    ask other
```

Pros:

- More natural than a fixed turn number.
- Uses conversation evidence instead of only turn count.

Cons:

- Still waits too long if the first few specific questions are weak.
- May not help cases where the useful metadata should be requested immediately after material/feature.

### C. Generic-evidence trigger

```text
if known evidence is mostly generic terms:
    ask other earlier
```

Generic terms include:

```text
Imported
100% Cotton
100% Polyester
Machine Wash
Pull On closure
Zipper closure
Rubber sole
```

Pros:

- Targets the exact failure pattern.
- More transferable than asking `other` at a fixed turn.

Cons:

- Needs careful implementation so it does not become public-set-specific.
- Should use broad generic-metadata logic, not a long hand-written list of public examples.

### D. Candidate-disagreement trigger

```text
if top candidates are still very similar after known preferences:
    ask other
```

Pros:

- Closest to production faceted search behavior.
- Asks open-ended clarification when the ranking system cannot separate candidates.

Cons:

- Requires reliable diagnostics from the candidate pool.
- More implementation work than a turn rule.

Current recommendation: test C or D first. If time is short, test B. Avoid pure hardcoded turn 5 unless the full 200-session score clearly improves and the report frames it as progressive clarification.

## Next checks

1. Tune popularity/quality weights on repeated stratified holdout, not full public score.
2. Compare moved sessions for description/identifier changes, especially intent override.
3. If popularity is kept, gate it to near-tied candidate pools rather than applying it globally.
4. Test earlier `other` using a generic-evidence or candidate-disagreement trigger.

## Remaining gaps and competition strategy

This section is for the next teammate working on the system. The current system is already submission-capable, but the likely winning improvements are not more public-set memorization. They should be production-style ranking and conversation fixes that match the hackathon rubric.

The competition explicitly rewards:

- Buying versus Browsing routing
- hybrid retrieval and semantic reranking
- structured constraint state
- intent override handling
- dynamic context construction
- adaptive clarification and question-value estimation
- safe personalization using aggregate profile
- failure detection and strategy switching
- low latency and low token cost
- transparent recommendation explanations

The weak sessions show that the remaining loss is mostly not from basic retrieval. The system already has high Hit@10. The gap is ranking quality, late conversion, and some commercially wrong items surviving after reranking.

### 1. Audience / department guardrail

Weak examples show wrong-audience products leaking into top 10:

```text
public_0174: Men's robe target, but top results are mostly Women's robes
public_0175: Men's jeans target, but rank 1 is Women's denim skirt
public_0035: Men's walking shoe target, but many Women's walking shoes outrank it
public_0137: Women hoodie target, but men/boys/baby items appear
```

Production reasoning:

Retail systems should understand audience or department:

```text
men
women
boys
girls
kids
baby
toddler
unisex
```

Recommended implementation:

```text
Infer requested audience from state.category, latest message, and confirmed preferences.
Infer product audience from product categories, title, and details.Department.

Then:
- boost clear same-audience products
- penalize clear mismatches
- do not hard-filter unless the requested audience is very explicit
```

Why this fits the rubric:

- business value: fewer obviously wrong recommendations
- technical value: final ranking guardrail after semantic reranking
- transferability: hidden set uses the same catalog fields and clothing departments

Careful:

- Do not treat `women` vs `girls` or `men` vs `boys` as always interchangeable.
- Do not over-penalize `unisex`, `kids`, or ambiguous products.
- Make it soft, not a hard filter, unless the mismatch is very clear.

### 2. Category purity after cross-encoder

The cross-encoder sometimes lets plausible but commercially wrong products survive:

```text
Men Jeans -> women's denim skirt, men's suit set, hoodie, shoulder bag
Crossbody Bags -> shoe accessories, backpacks, wallets, shoes
Robes -> sleepwear robes mixed with unrelated lounge categories
```

Production reasoning:

A standard commerce ranking stack usually looks like:

```text
candidate retrieval
semantic reranking
business/facet guardrail final pass
```

The cross-encoder should improve semantic relevance, but it should not be the final authority when category/facet constraints are clear.

Recommended implementation:

```text
After cross-encoder reranking:
- boost exact leaf-category match
- lightly boost same parent category
- neutral for same department but different leaf
- penalize unrelated category
```

Example:

```text
query/category: Men > Clothing > Jeans

best:
Men > Clothing > Jeans

acceptable:
Men > Clothing > Pants

weak:
Women > Clothing > Jeans

bad:
Women > Handbags
```

Why this fits the rubric:

- improves ranked recommendations
- makes the system more production-like
- supports transparent recommendation explanations

Careful:

- Do not overfit exact public categories.
- Use category path overlap and leaf-category matching, not a hand-written list of target products.
- Keep penalties small enough that a strong exact text/identifier match can still win.

### 3. Earlier `other` when structured facets stop helping

Current weak examples show repeated low-value questions:

```text
brand -> no preference
size -> no preference
use_case -> no preference
style -> no preference
budget -> no preference
```

Useful details often appear only when asking `other`:

```text
Item model number
cotton blend
100% Polyester
Rubber sole
Shaft measures approximately...
Tie closure
```

Production reasoning:

When normal facets stop separating candidates, a real shopping assistant should ask an open-ended but still structured question:

```text
Is there another must-have detail I should consider?
```

Recommended implementation:

```text
Ask other earlier when:
- category is known
- the candidate pool is still crowded
- known preferences are generic/common
- remaining specific attributes have low diagnostic value
- shopper has already said no preference for one or two specific attributes
```

Why this fits the rubric:

- adaptive clarification
- question-value estimation
- strategy switching when retrieval confidence is low

Careful:

- Avoid a pure hardcoded rule like `turn >= 5 -> ask other` unless it clearly wins in full evaluation.
- A fixed turn rule is easy to implement but more metric-shaped.
- Better options are generic-evidence triggers or candidate-disagreement triggers.

### 4. Generic metadata / IDF cap

The hard cases are dominated by common catalog metadata:

```text
Imported
100% Cotton
100% Polyester
Machine Wash
Pull On closure
Zipper closure
Rubber sole
Hand Wash Only
```

Production reasoning:

These terms are relevant, but they are weak evidence because they appear in many products. They should not have the same ranking force as distinctive terms like a brand, model number, exact style phrase, or rare feature.

Recommended implementation:

```text
Use document frequency:
- common terms get capped/lower boost
- rare terms get stronger evidence
- generic metadata only contributes strongly when category and audience already match
```

Why this fits the rubric:

- better retrieval/ranking quality
- transparent scoring
- avoids over-trusting noisy metadata

Careful:

- Do not hard-code only the public-set phrases.
- Use corpus statistics where possible.
- Keep exact identifier matching separate; model/SKU/part numbers should still be strong.

### 5. Safer intent override handling

Intent override sessions still have awkward behavior. The agent needs to distinguish:

```text
replace whole product/category
replace one preference
remove an old constraint
add a new constraint
```

Production reasoning:

If a user says "actually ignore that, I need polyester", they may be correcting one attribute, not replacing the whole product search. But if they say "actually I need waterproof hiking boots", that is probably a new product intent.

Recommended implementation:

```text
If override contains a new product noun/category:
    reset product intent and clear old product-specific attributes

If override contains only an attribute value:
    replace that attribute and keep category context

If override negates a value:
    move it to removed_preferences
```

Why this fits the rubric:

- directly matches "intent override handling"
- improves multi-turn state management
- makes behavior explainable in the final report

Careful:

- Do not drop useful category context for preference-only corrections.
- Do not preserve stale constraints when a real new product intent appears.
- Without an LLM, this should be conservative and rule-based.

### 6. User profile is underused

Current profile behavior is very small:

```text
MAX_PROFILE_BOOST = 0.03
PROFILE_TERM_BOOST = 0.005
```

It only checks simple lexical overlap between profile tags/summary and product text. It does not semantically connect:

```text
comfort -> cushioned, relaxed, soft, breathable
fit -> relaxed fit, standard fit, slim fit, stretch
style -> printed, floral, boot cut, casual
durability -> rugged, reinforced, leather, workwear
```

Production reasoning:

The profile should not override the active query, but it can help personalize ties and ask better questions.

Recommended implementation:

```text
Use profile lightly for:
- query rewrite in browsing sessions
- reranking tie-breaks
- question choice
- popularity/quality preference sensitivity
```

Why this fits the rubric:

- safe personalization using aggregate profile
- better browsing experience
- transparent business value

Careful:

- Keep profile weaker than explicit user preferences.
- Do not let profile change the category.
- Avoid strong boosts unless relevance is already close.

## Suggested implementation order

Best next steps:

```text
1. Audience guardrail
2. Category purity final pass
3. Earlier other trigger
4. Generic metadata / IDF cap
5. Gated popularity for close scores only
6. Semantic profile expansion
```

The highest-confidence next implementation is:

```text
audience + category guardrail after cross-encoder
```

Reason:

- directly visible in multiple weak examples
- strongly production-valid
- aligns with business value and ranking quality
- unlikely to depend on public target IDs
- easy to explain in the final report

## Submission judgment

The current system is already credible and submission-capable. It has:

```text
hybrid lexical/vector retrieval
cross-encoder reranking
conversation state
intent override handling
clarifying questions
offline fallback
profile tie-breaker
description-aware matching
exact identifier handling
```

But to be in a stronger possible-winning position, do not just add every idea blindly. Add one production-valid improvement at a time and run the full 200-session evaluator after each change.

Keep a change only if:

```text
score improves or stays close
Hit@10 does not meaningfully drop
MRR does not collapse
scenario-level regressions are explainable
the final report can justify it as production search logic
```

Do not chase public-set-only gains. The hidden 800 likely rewards robust ranking behavior more than brittle tuning.
