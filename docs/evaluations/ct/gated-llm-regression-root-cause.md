# Gated LLM Regression Root-Cause Analysis

Date: 2026-08-30

## Executive Summary

The original ambiguity gate called the state LLM on every session's first turn.
That caused eight regressions by changing the deterministic parser's calibrated
state shape. The problem was not simply model quality. The integration allowed
model output to duplicate evidence, move values between weighted attribute
buckets, promote generic metadata into confirmed preferences, and leak an
attribute phrase back into category.

The fixes are semantic and independent of public sample IDs:

1. Explicit first-turn product requests use the deterministic parser.
2. The LLM remains available for ambiguous correction and override scope.
3. Model values cannot duplicate an existing preference or category value.
4. Generic catalog metadata remains lexical evidence on ordinary merges.
5. Preference-only overrides cannot overwrite category.
6. Reasserted values retain their established deterministic attribute bucket.

## Final Metrics

| Metric | Deterministic | Broad LLM gate | Fixed selective LLM |
| --- | ---: | ---: | ---: |
| Technical score | 0.887760 | 0.888631 | **0.888522** |
| Hit@10 | 0.990000 | 0.990000 | **0.990000** |
| MRR | 0.810200 | 0.807770 | **0.812075** |
| MTTC | 3.515000 | 3.435000 | **3.505000** |
| LLM routes | 0 | 230 | **32** |
| API tokens | 0 | 318,473 | **49,910** |

The final version gains `+0.000762` over deterministic, raises MRR, preserves
Hit@10, and reduces API tokens by about 84% relative to the broad gate. Only 31
of 200 sessions invoke the LLM. Thirty are intent-override sessions; the extra
route is a buying message containing correction-like language inside product text.

The model remains stochastic, so the exact score can vary slightly. The stronger
result is structural: the LLM can no longer modify normal initial requests, and
its output is compiled back into the deterministic ranking schema.

## Original Regressions

### `public_0189`: duplicate and mis-bucketed evidence

Request: `Shorts Denim` with `cotton`.

The model added:

- `denim` as material even though it was already part of category;
- `cotton` as feature even though deterministic parsing had already stored it as
  material.

The resulting state contained `cotton` in two weighted fields and changed ranking
from T3 R1 to T5 R2. Fixed by cross-field deduplication, category-value suppression,
and deterministic handling of explicit initial requests. Final result: T3 R1.

### `public_0072`: attribute leaked into category

Override: `ignore my earlier preference ... Faux Fur` while searching anoraks.

The LLM correctly returned preference replacement, unchanged category, and
`Faux Fur` as feature. The canonicalizer then restored the fallback parser's
incorrect category `faux fur`. Fixed by honoring `category=unchanged` when the
fallback category is exactly represented by the model's replacement preference.
Final result improves from deterministic T4 R1 to T3 R1.

### `public_0080`: generic metadata became a hard ranking signal

The initial `Button closure` text was promoted from lexical evidence to a confirmed
feature. That changed downstream weighting before the cotton override and produced
T4 R2 instead of T5 R1. Fixed by keeping ordinary generic metadata lexical and by
bypassing the LLM for the explicit initial request. Final result: T5 R1.

### `public_0183`: incorrect intent plus metadata promotion

The model classified a normal vest request as browsing and promoted `Hand Wash
Only` to a confirmed feature. Both changed the pipeline's behavior before the
polyester override. Fixed by preserving explicit deterministic first-turn intent
and keeping generic metadata lexical. Final result: T5 R1.

### `public_0042`: `Imported` was over-weighted

The model turned generic `Imported` metadata into a confirmed feature for a watch
request. That surfaced the target earlier at R2, but the evaluator froze the lower
rank instead of allowing the later R1. Fixed by the generic-metadata guard and
safe initial route. Final result: T4 R1.

### `public_0002`: closure metadata changed calibrated state

The model promoted `Buckle closure` into a confirmed feature before a leather
override. This changed the candidate order from T5 R3 to T5 R6. Fixed by retaining
the deterministic initial state shape. Final result: T5 R3.

### `public_0001`: model-added material changed first-hit ordering

The LLM typed `alloy` as material where deterministic parsing retained it as
lexical evidence. Although the typing is semantically reasonable, the downstream
weights were tuned to the deterministic representation and rank fell from R2 to
R3. Explicit initial requests now use deterministic state extraction. Final result:
T4 R2.

### `public_0133`: generic `Imported` feature delayed the hit

The model promoted `Imported` to a confirmed sunglasses feature, moving T4 R1 to
T5 R1. The generic-metadata guard and safe initial route restore T4 R1.

## Final Remaining Regression

### `public_0125`: correct state, evaluator first-hit trade-off

The shopper is searching baseball caps, then says to ignore an earlier preference
and require `100% Acrylic`.

The deterministic parser incorrectly treats `100% acrylic` as a new product
category. It does not hit until turn 5, after the evaluator reveals a long,
near-verbatim product description, at R1.

The fixed LLM path correctly:

- preserves category `baseball caps`;
- removes the ignored hook-and-loop preference;
- retains `100% acrylic` in its established feature bucket;
- finds the target one turn earlier at T4 R2.

The evaluator assigns less reciprocal-rank value to T4 R2 than T5 R1, so this is
recorded as a regression even though the state and user experience are better.
Avoid hiding the T4 result or deliberately corrupting category to recover the
public score. Those changes would be metric exploitation and unlikely to transfer.

## Persistent Misses

### `public_0144`

Target: URBAN REPUBLIC women's water-resistant faux-fur parka.

The simulated conversation exposes only broad category, polyester, `Imported`,
and `Zipper closure`. It does not reveal the distinguishing brand, audience,
water resistance, faux-fur hood, or winter-use details. Many parkas satisfy the
observed evidence. The LLM parses the state correctly but cannot manufacture the
missing distinguishing information.

### `public_0175`

Target: Ariat men's M2 relaxed boot-cut jeans.

The conversation exposes men's jeans, cotton, `Imported`, and `Zipper closure`.
It never reveals Ariat, relaxed fit, boot cut, or model number `10026664`. This is
also underdetermined rather than an intent-parsing failure.

Popularity could move these targets by chance, but tuning popularity to rescue two
public products would not be defensible for the hidden 800 sessions. A production
system needs either richer shopper evidence, behavior/history signals, or a
candidate-diversification objective that the current single-target evaluator does
not reward.

## Final Session-Level Comparison

Against deterministic:

- 197 sessions unchanged;
- 2 improved: `public_0038` (T5 R8 to T5 R1) and `public_0072`
  (T4 R1 to T3 R1);
- 1 metric regression: `public_0125` (T5 R1 to T4 R2);
- no new misses;
- persistent misses remain `public_0144` and `public_0175`.

## Recommendation

Keep the fixed selective LLM path. It is more production-like than either extreme:

- deterministic parsing handles high-confidence, structured messages cheaply;
- the LLM resolves ambiguous destructive state transitions;
- canonicalization protects retrieval from model-specific schema drift;
- deterministic fallback remains available on API or validation failure.

Do not add sample-specific synonyms, product boosts, delayed-result suppression,
or popularity tuning to eliminate the remaining public-set loss. Those would be
less likely to transfer to the hidden evaluation than the current semantic rules.

## Held-Out Overfitting Check

The fixed version was evaluated on the separate 200-session
`data/synthetic_set.jsonl` split using the same configuration.

| Metric | Held-out deterministic | Held-out selective LLM | Delta |
| --- | ---: | ---: | ---: |
| Technical score | 0.870937 | 0.870937 | 0 |
| Hit@10 | 0.955000 | 0.955000 | 0 |
| MRR | 0.816123 | 0.816123 | 0 |
| MTTC | 3.570000 | 3.570000 | 0 |
| LLM routes | 0 | 30 | +30 |
| API tokens | 0 | 45,681 | +45,681 |

At session level:

- 198 sessions were unchanged;
- `synthetic_0101` improved from T5 R1 to T4 R1;
- `synthetic_0188` regressed from T3 R1 to T4 R1;
- both runs missed the same nine sessions;
- there were no new misses or rank regressions.

This is evidence that the selective gate and canonicalizer do not damage a
different session set. It is not evidence that the LLM improves hidden-set
accuracy: its held-out aggregate contribution is exactly neutral. The public
`+0.000762` should therefore be treated as a small observed result, not a reliable
forecast of hidden-set gain.

The held-out split also shares the same simulator family and catalog, and the
rules were developed after inspecting public failures. A truly independent hidden
evaluation remains the only conclusive test. The defensible generalization claim
is narrower: runtime code contains no public sample IDs or target-product boosts,
the rules express generic state invariants, and the independent synthetic split
shows no degradation.

## Verification

- 309 unit tests pass.
- Final result: `docs/evaluations/ct/results/public-selective-llm.json`.
- Deterministic comparison: `docs/evaluations/ct/results/public-deterministic.json`.
- Held-out deterministic: `docs/evaluations/ct/results/heldout-deterministic.json`.
- Held-out selective LLM: `docs/evaluations/ct/results/heldout-selective-llm.json`.
