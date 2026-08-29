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
