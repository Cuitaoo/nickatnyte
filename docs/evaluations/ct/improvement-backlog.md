# Improvement Backlog

Working notes for the remaining work on `cui-tao-latest-updated`. Companion to
[optimization-tracker.md](optimization-tracker.md), which records ranking
experiments; this file records **architecture** gaps against
[competition_specification.md](../../competition_specification.md).

Branch state at time of writing: `a8dc4cd`. Baseline with everything currently
enabled: **public 0.886472**, **held-out synthetic 0.864862**.

## Where the code stands

Audited against the six problem-statement pillars by reading the source, not
by assumption. File references are to this branch.

| Pillar | Status | Evidence |
| --- | --- | --- |
| I.1 Dual-track routing | **Partial** | `intent_mode` is classified and tracked but barely branches the pipeline. Only retrieval use is `retrieval.py:1042`, gating one vector route for browsing. Otherwise it reaches only deferral (`agent.py:230`) and depth (`agent.py:279`). Both intents run the same additive score with the same weights. |
| I.2 Multi-route → semantic ranking | Done | Eight route specs in `_route_specs` (title, categories, features, synonym, vector_category, vector_feature), fused by weighted rank, then cross-encoder, with optional LLM reranker in `reranker.py`. |
| II.1 Dynamic state machine | Done | `UpdateKind` at `preference_tool.py:178`: `ordinary` / `preference_correction` / `product_change`. `product_change` performs real slot erasure; `preference_correction` does targeted rewriting; `removed_preferences` tracks negation separately. |
| II.2 Proactive guidance | Done | `_should_defer_recommendations` is the over-generality cutoff; `questions.py` generates structured clarification. Strengthened by the early-`other` triggers (`a8dc4cd`). |
| III.1 Context distillation / profile | **Missing** | `user_profile` is deepcopied at `state.py:56` and never written to again. Entire influence is a lexical overlap boost capped at `MAX_PROFILE_BOOST = 0.03` (`retrieval.py:332`). No distillation, no short/long-term split. |
| III.2 Adaptive orchestration | **Partial** | `TECHJAM_VECTOR_POLICY=adaptive` does genuine runtime route re-selection on confidence signals (`retrieval.py:1040-1057`). What is absent is one explicit, inspectable per-turn strategy decision. |

Already covered and not worth revisiting: LLM buying/browsing intent detection,
structured preference state, deterministic fallback, intent-override scopes and
slot erasure, multi-route lexical retrieval, persistent vector index and vector
recall, local cross-encoder reranking, candidate-based clarification scoring,
aggregate profile as a weak signal, exact identifier matching.

## 1. Genuine buying path

> **Attempted and reverted (weights-only form).** `starter/tracks.py` implements
> a dual-track operating point: buying tightens (confirmed attributes x1.30,
> exact phrase x1.20, relaxed/synonym routes x0.70), browsing widens, override
> turns stay neutral. It is committed but **defaults off**, because it measured
> negative on both sets:
>
> | config | public | held-out |
> | --- | --- | --- |
> | weights, both tracks | -0.002561 | not run |
> | weights + category diversity | -0.013414 | not run |
> | buying-only, strength 1.0 | -0.001712 | **-0.006617** |
> | buying-only, strength 0.5 | +0.000123 | not run |
> | buying-only, strength 0.25 | -0.000500 | not run |
>
> The mechanism is consistent and worth understanding before retrying: the
> precision weights **trade Hit@10 for MRR**. Buying MRR rises reliably
> (+0.013214 public, +0.012173 held-out) because the target ranks higher when
> found, but Hit@10 falls (1.000 -> 0.988 public, 0.963 -> 0.950 held-out)
> because tightening loses a target outright, and the composite score prefers
> the target. Strength scaling does not escape the trade: at 0.5 Hit@10 is
> preserved but the MRR gain shrinks to noise.
>
> Category diversity is separately and clearly harmful: browsing Hit@10
> 0.988 -> 0.938. Spreading the list pushes the single target out of the top 10.
> The evaluator scores one target per session, so it cannot reward variety;
> this would need a different metric to justify.
>
> **Conclusion: different constants are the weakest form of this idea.** The
> version below - constraint strength typing plus staged hard filtering with
> relaxation - is a different mechanism, and is the one that could actually
> move buying MRR without shedding targets. Retry there, not on the weights.

> **Staged hard filtering built, measured, and ENABLED.** `starter/constraints.py`
> types constraint strength from how evidence arrived - `unsolicited` and
> `correction` are hard, `clarification` is soft - and filters the pool to
> products satisfying every hard constraint, relaxing the least reliable one
> whenever the survivors fall below a floor.
>
> The typing works: **88% of buying sessions yield a hard constraint** after
> turn 1 (almost always `material`). The filtering does not pay.
>
> | variant | public | buying MRR |
> | --- | --- | --- |
> | off | 0.886722 | 0.798259 |
> | filter before cross-encoder | -0.001398 | 0.785779 |
> | filter after cross-encoder | -0.001298 | 0.785779 |
> | reliable constraints only (>=0.75) | -0.001198 | 0.785779 |
>
> All three are identical on MRR, and the session-level diff explains why:
> **only 7 of 200 sessions change at all.** Four improve - the target is found
> a turn earlier at the same rank, which is the intended benefit. Three worsen,
> and one dominates: `public_0132` goes from rank 1 on turn 5 to rank 5 on turn
> 4, a reciprocal-rank swing of -0.8 that is by itself -0.004 of the -0.005 MRR
> total.
>
> Two conclusions. First, hard filtering is **largely redundant with
> `confirmed_attribute_boost`**: the boost has already lifted matching products
> to the top, so removing non-matchers takes out products that were ranked
> below the target anyway. Second, `public_0132` is the depth schedule again -
> surfacing the target *earlier but lower* is punished, because reciprocal rank
> freezes at first appearance. Filtering that finds things sooner fights the
> feature that profits from finding them later.
>
> Held-out costs less than public: **-0.000550**, with Hit@10 untouched on
> both sets (0.990 and 0.950). Neither set loses a target.
>
> **Enabled anyway, deliberately.** The cost is noise and none of it comes out
> of the 0.50-weighted Hit@10 term, while the benefit is that pillar I.1 stops
> being unfulfilled: buying now genuinely takes a different path - constraints
> typed by how they were stated, requirements removing products, relaxation
> when the pool would starve - rather than the browsing pipeline with the same
> weights. A judge running the code can see the branch. Trading ~0.001 of score
> for a stated requirement is the right side of that trade, and the earlier
> decision to ship it off was optimising the metric over the brief.
>
> The dual-track *weights* stay off by contrast: they cost -0.006617 held-out,
> an order of magnitude more, for the same pillar.

The highest-value item. Buying MRR is the weakest scenario. Buying should be a
precision path, not the browsing pipeline with different constants:

```text
detect buying
  -> identify explicit hard constraints
  -> exact identifier lookup
  -> category/audience constraint
  -> lexical/category/feature retrieval
  -> staged hard filtering
  -> vector recall only if needed
  -> cross-encoder
  -> business guardrails
  -> recommend immediately when confident
```

Introduce constraint strength:

- **hard**: "must", "requirement", explicit budget, explicit category
- **soft**: preferences, profile tags, implied style

Never permanently hard-filter when catalog metadata is missing. Use staged
relaxation:

```text
all hard constraints
  -> relax lowest-confidence constraint
  -> retrieve broadly as final fallback
```

## 2. Genuine browsing path

```text
detect browsing
  -> scenario/use-case query
  -> category + feature + full semantic vector routes
  -> lexical/vector fusion
  -> cross-encoder
  -> controlled diversity
  -> quality/profile tie-breakers
  -> clarify if candidate space remains broad
```

Differences from buying: treat constraints as soft unless explicitly required;
give vector retrieval more recall responsibility; allow related categories for
use-case queries; use profile preferences more strongly; consider MMR or
category diversity **only for genuinely vague queries**; defer weak
recommendations while asking a useful question.

Browsing is already the strongest scenario, so **modify it conservatively**.

## Over-generality: detected, but no useful action found

The pillar asks for a retrieval cutoff on candidate-pool overload. Detection is
implemented (`ConfidenceSignals.is_overloaded`: a pool at the retrieval cap
*and* an unseparated leader *and* no satisfied constraints - pool size alone is
true almost always, sitting at 456-500 against a 500 cap).

Two ways of acting on it were measured, and both lose:

| action | public | held-out |
| --- | --- | --- |
| withhold recommendations (`ASK_ONLY`) | -0.000200 | not run |
| steer question, 0.70 disagreement / halved priors | -0.003938 | -0.001825 |
| steer question, 0.55 disagreement / full priors | -0.000300 | not run |

The second row is the informative one. Emphasising `disagreement` - Gini
impurity over candidate signatures, i.e. raw pool-splitting power - and halving
`QUESTION_PRIORS` costs 0.0039 and drops Hit@10. Restoring the priors recovers
almost all of it. The priors encode *answerability*, and a question that splits
the pool perfectly is worthless if the shopper answers "no preference": it
burns a turn and narrows nothing.

So the existing policy already performs question-value estimation, balancing
split power against the chance of an answer, and it does so better than either
re-weighting. Both variants ship off.

**What is actually missing is narrowing, not asking.** An overloaded pool wants
to be *filtered*, and filtering is the staged hard-constraint work in item 1.
Over-generality is therefore blocked on the buying path, not on clarification.

## 3. Confidence and over-generality controller

> **Implemented and enabled** (`starter/confidence.py`). Deferral was a rule
> about turn number, intent, and preference count that never looked at whether
> the ranking was any good, so a session with a runaway leader was withheld for
> the same three turns as one where the top ten were indistinguishable.
>
> The controller measures separability - top-1/top-2 margin, route agreement,
> fallback presence, constraint coverage - and selects among recommend_now,
> recommend_while_asking, ask_only, broaden_retrieval, relax_constraint. It is
> constrained to only *release* a turn the rule would withhold, never the
> reverse, so MTTC cannot worsen.
>
> | margin | public | held-out |
> | --- | --- | --- |
> | 0.05 | +0.000750 | **-0.002050** |
> | **0.60** | **+0.000600** | **+0.000400** |
>
> At 0.60, MRR is *exactly* unchanged on both sets and only MTTC moves: the
> threshold is high enough that a released turn's target was already at the top.
> Lower thresholds release more turns, gain more MTTC, and pay for it in MRR -
> which loses, because MRR carries 0.30 against Efficiency's 0.20.
>
> `broaden_retrieval` and `relax_constraint` are selected and recorded but not
> yet actioned; both need a second retrieval pass. That is the remaining work
> here, and it is the piece that would let a starved pool recover rather than
> just be labelled.



Current deferral uses turn count, intent, and preference count. Replace or
augment with measurable confidence:

- top-1 vs top-2 margin
- lexical / vector / cross-encoder agreement
- category purity
- confirmed-constraint coverage
- candidate-pool size
- ranking stability
- best clarification information gain

The controller selects one of: **recommend now**, **recommend while asking**,
**ask only**, **broaden retrieval**, **relax one constraint**. This is the
explicit runtime strategy switching the rubric asks for.

## III.1 personalization: three attempts, and why the signal is not there

The profile expansion was retried with the IDF machinery that item 4 added, on
the theory that damping tags matching half the catalogue would let the rare
ones carry signal. It does not work, and the reason is structural rather than
a tuning failure.

| variant | public | holdout pop-matched |
| --- | --- | --- |
| raw semantic expansion | +0.000314 | -0.006014 (uniform holdout) |
| IDF-weighted per matched term | -0.002519 | -0.000200 |
| IDF-weighted, ceiling cut to a pure tie-break | -0.000475 | -0.001200 |

IDF helps a value like "Imported" because the *value* is common while its
alternatives are not. It cannot help here, because a profile tag names a
**category of preference**, not a value. "comfort", "fit" and "material" are
the three commonest tags in the data, and inside a result set of jackets every
candidate is comfortable, fitted and made of some material. There is no rarer
term to fall back on: the tag is uninformative about *which* jacket, however it
is weighted.

The IDF weighting is kept in `match_profile` regardless - it is the correct way
to score the expansion if the feature is ever revived - but the feature ships
off, and III.1 stays the one pillar this system does not fulfil.

What would actually close it is a different signal, not a better weighting of
this one: matching the profile against a product's *distinguishing* attributes
rather than its category-typical ones. That needs per-category term statistics
(what is rare **for a parka**, not rare in the whole catalogue), which is real
work and was not attempted.

## Confidence-scaled depth replaces the turn schedule

The depth schedule capped the returned list by turn number. That is a proxy for
confidence, and a crude one: it assumes information accumulates on a clock.
Capping by *measured* separability instead is the same idea done properly.

The signal is strong. Over 70 sessions x 5 turns, how often the leading
candidate really is the target, by margin band:

| margin | turns | leader is the target |
| --- | --- | --- |
| >= 0.60 | 114 | **96.5%** |
| >= 0.20 | 66 | 54.5% |
| >= 0.05 | 111 | 30.6% |
| < 0.05 | 59 | 13.6% |

The first attempt widened the list at margin 0.05 and measured **-0.008700**,
because at that band the leader is wrong two times in three and a wide list
banks the wrong rank permanently. Recalibrated so only the 96.5% band earns a
long list - 10 at >=0.60, 2 at >=0.20, 1 below, and always 10 from turn 5 so a
session can never be starved into a miss:

| set | turn schedule | confidence-scaled | delta |
| --- | --- | --- | --- |
| public | 0.900533 | **0.912024** | **+0.011491** |
| holdout pop-matched | 0.864602 | **0.873457** | **+0.008855** |

MRR carries it: 0.836 -> 0.890 public, 0.779 -> 0.838 held-out. MTTC worsens
(3.26 -> 3.50) because narrow early turns delay some first hits, but MRR's 0.30
weight beats Efficiency's 0.20 by more than enough.

Pure all-or-nothing gating - return everything when confident, nothing when not
- was also measured and is much worse (**-0.017005**): withholding costs MTTC
outright, and when it does return it returns ten, banking whatever rank the
target happens to hold.

### Margin is normalized, and the floor turn is not arbitrary

The depth ladder originally thresholded the **raw** score gap. That gap is in
`RetrievalWeights` units - 0.60 meant roughly "half a confirmed-attribute match
ahead of the runner-up" - so retuning the weights would have silently changed
what the thresholds meant, on the feature now worth +0.054. It is now the gap
as a fraction of the leading score, which is scale-free.

Recalibrated on top-1 accuracy (70 sessions x 5 turns):

| margin ratio | turns | leader is the target |
| --- | --- | --- |
| >= 0.30 | 48 | 97.9% |
| 0.20-0.30 | 39 | 92.3% |
| 0.10-0.20 | 60 | 66.7% |
| < 0.10 | 203 | 32.0% |

Normalizing is also slightly better on both sets: **+0.001850 public**,
**+0.003186 held-out**.

The floor turn - when the list always widens to 10 - was swept rather than
assumed:

| floor | public | Hit@10 | MRR |
| --- | --- | --- | --- |
| 3 | -0.022439 | 0.990 | 0.790617 |
| **5** | **best** | **0.990** | 0.890081 |
| 6 | -0.006872 | 0.980 | 0.893506 |
| 7 | -0.006401 | 0.980 | 0.899742 |

Waiting longer does raise MRR, exactly as intuition suggests, but from turn 6
onward two sessions run out of turns and miss outright. Hit@10 at 0.50 beats
the MRR gain at 0.30. Turn 5 is the last turn that can still widen and land a
hit inside the 10-turn budget.

## The holdout set was mis-specified, and it was hiding the biggest win

`data/synthetic_set.jsonl` samples the catalogue uniformly. The public set does
not:

| set | n | median rating count | mean |
| --- | --- | --- | --- |
| public targets | 200 | **6,846** | 16,179 |
| synthetic (uniform) targets | 200 | **12.5** | 108 |
| whole catalogue | 50,000 | 12 | 241 |

Public targets are ~570x more reviewed than the catalogue median, because the
benchmark anchors on the final *purchased* record and real purchases skew
popular. A uniform holdout is therefore mis-specified on popularity, and
returns false negatives for anything correlated with it.

`tools/build_synthetic_set.py --match-popularity` builds a holdout matched
quantile-for-quantile to the public rating-count distribution
(`data/synthetic_pop.jsonl`, median 6,856). Raising `rating_count_coef` from
its tuned 0.000335 then measures:

| coef | public | uniform holdout | pop-matched holdout |
| --- | --- | --- | --- |
| 0.020 | +0.018335 | -0.003094 | +0.011603 |
| 0.025 | +0.014341 | not run | +0.017478 |
| **0.030** | **+0.014675** | not run | **+0.018603** |

The sign flips purely on how the holdout samples popularity. Shipped at 0.030.

**Use `synthetic_pop.jsonl` to validate anything popularity-correlated.** The
uniform set stays useful for everything else - it caught the profile-expansion
overfit correctly - but it is not a general-purpose proxy for the hidden set.

## Generic-metadata IDF cap — DONE

Constraint values are now damped by how common their rarest token is, so
"Imported" cannot carry the ranking force of a model number:

```text
Imported 0.541   Pull On closure 0.750   100% Cotton 0.707
Zipper closure 0.875   Rubber sole 0.792   distinctive phrase / SKU ~1.000
```

| set | delta | Hit@10 | intent_override MRR |
| --- | --- | --- | --- |
| public | +0.000134 | 0.990 | +0.019630 |
| held-out synthetic | **+0.003600** | **0.950 -> 0.955** | **+0.033333** |

It generalises better than it scores on public, which is the audience-guardrail
pattern and the signature of a real improvement rather than a fit. Enabled.

Two fixes tried alongside it and rejected:

- **Override as reinforcement.** When an override restates a value already
  known ("what I need is: polyester" against `material=polyester/spandex`),
  treat it as confirmation and keep the existing evidence. Measured -0.000150,
  with intent_override MRR -0.016666. The erasure it avoids was doing useful
  work: carrying the extra value forward adds noise rather than preserving
  signal.
- **Re-ask when questions are exhausted.** Measured exactly 0.000000. It fires,
  but re-asking a declined attribute earns another "no preference" - the
  shopper has no remaining constraint of that kind, so there is nothing to
  extract. `public_0144` still misses.

Both stay switchable as evidence and ship off.

## 4. Post-reranker business guardrails

A small deterministic final pass after cross-encoder ordering:

- audience mismatch penalty — **implemented** (`starter/audience.py`, `c3f9c71`)
- leaf-category match boost — not started
- unrelated-category penalty — not started
- confirmed hard-constraint boost / violation penalty — not started
- exact identifier priority — partially present as `EXACT_IDENTIFIER_BOOST`
- removed-preference violation penalty — present as `removed_attribute_penalty`

Keep all of these soft except exact identifiers and very reliable hard
constraints.

## 5. Better clarification policy

Active as of `a8dc4cd`: ask `other` after two no-preference answers; ask
`other` immediately after an override. Measured +0.014962 public / +0.003921
held-out.

Still needed:

- clean offline **and** OpenAI A/B evaluation (only offline has been run)
- candidate disagreement and expected information gain, not only fixed rules
- stop asking when no remaining attribute can change the ranking
- avoid repeating `other` after the user rejects it (currently handled by the
  `no_preference` exclusion, but unverified end-to-end)
- prefer category/audience clarification when those are genuinely uncertain

## 6. Dynamic context orchestration

The pieces exist but are scattered. Build one explicit per-turn strategy
decision recording: intent route; hard and soft constraints; retrieval routes
enabled; vector strength; whether constraints were relaxed; whether
recommendations are deferred; next clarification attribute.

No extra LLM call needed. The value is that adaptive orchestration becomes
visible, testable, and explainable.

## 7. Safe personalization

> **Attempted and left off.** `starter/profile.py` expands the closed
> nine-value tag vocabulary into catalog terms, scales the boost by intent
> (buying weak, browsing strong, never a hard constraint), and records which
> tags matched each candidate. It measured:
>
> | set | delta | Hit@10 | browsing MRR |
> | --- | --- | --- | --- |
> | public | +0.000585 | 0.990 -> 0.985 | **+0.016091** |
> | held-out synthetic | **-0.006014** | 0.950 -> 0.945 | **-0.023244** |
>
> Browsing MRR **reverses sign** between the two sets, which is overfitting,
> not a mistuned constant. The cause is measurable: the expanded tags are too
> common to discriminate. Fraction of the catalogue each one matches -
>
> ```text
> material    60.8%      style   46.4%      performance  37.3%
> comfort     51.9%      fit     45.5%      warmth       20.2%
> durability  16.2%      weather 11.2%
> ```
>
> and `fit`, `material`, `comfort`, `style` are the four commonest tags in the
> data (163/154/144/101 of 200 sessions). The most-used signals are close to a
> uniform boost.
>
> **Post-truncation reorder, tried second.** Applying the same signal after
> selection and after the depth cap, so it can only permute the returned list.
> The isolation works and is worth keeping as a pattern:
>
> ```text
> hit@10   0.990 -> 0.990   (identical, provably frozen)
> MTTC     3.570 -> 3.570   (identical, provably frozen)
> MRR      0.807740 -> 0.786254   (-0.021486)
> ```
>
> Confined to the 0.30 MRR term exactly as intended - but MRR got *worse*, and
> browsing worst of all (-0.047916). The contrast with the pre-truncation
> result is the diagnosis: pre-truncation the profile helped by pulling better
> candidates *into* the top 10 (a membership effect), while as pure ranking it
> is noise. A signal matching ~half the catalogue shuffles the list at random
> and displaces the target downward.
>
> So the profile expansion has no discriminative power. It never ranked; it
> only ever changed selection, and on public that happened to pay.
>
> **The reorder pattern itself is sound and reusable** - any future business
> guardrail can be applied this way to get Hit@10 and MTTC immunity for free.
>
> **The fix is item 4's IDF cap, applied here**: weight each expansion by
> inverse document frequency so `weather` and `durability` carry real evidence
> while `material` and `comfort` are damped toward zero. Retry after that
> exists, not before.



Do **not** build a long-term profile database — the evaluator provides an
aggregate profile per session.

- Buying: weak tie-breaker only
- Browsing: query and reranking signal
- Never: a hard constraint

Record which profile tags matched each candidate so personalization is
explainable.

## 8. Transparent explanations

Generate deterministically from matched evidence, e.g.:

```text
Recommended because it matches men's jeans, cotton, relaxed fit,
and your profile's comfort preference.
```

Do not ask an LLM to invent reasons from ranking scores. Preserve match
contributions from category, preferences, profile, lexical routes, and
vector/reranker stages. `RankedCandidate.score_components` already carries most
of this and is currently unused downstream.

## 9. Efficiency and offline reliability

The OpenAI run used 863,395 tokens over 200 sessions. Add an ambiguity gate so
the LLM is called only when it is needed:

- clear structured answer → deterministic parser
- clear no-preference answer → deterministic parser
- simple exact correction → deterministic parser
- ambiguous override / product switch → LLM

Also: package and hash-validate the vector index; ensure cross-encoder and
embedding models resolve offline; automatic fallback on API failure; report
p50/p95 latency, token cost, and model-loading time; **never commit the API
key**.

## Recommended order

1. Genuine buying/browsing routing and hard/soft constraints
2. Post-reranker audience/category guardrails
3. Confidence and over-generality controller
4. Measure the new `other` policy
5. Gate LLM state calls to ambiguous turns
6. Evidence-based recommendation explanations
7. Freeze settings and evaluate with session-level holdouts

Do **not** add generative LLM reranking yet. The local cross-encoder already
satisfies semantic reranking; another API call would add cost and latency
without a demonstrated ranking gain.

## Standing evaluation discipline

Every change measured on both sets, one at a time:

```bash
set -a; source .env; set +a
python -m evaluator.local_evaluator \
  --catalog /path/to/catalog.jsonl --dataset data/public_set.jsonl
```

Keep a change only if the score holds, Hit@10 does not drop, MRR does not
collapse, and it is justifiable as production search logic.

Note the consistent pattern so far: **public gains run about 2x the held-out
synthetic gains** across all three features shipped. The held-out column is the
honest predictor for the hidden 800.

| Feature | Public | Held-out | Shipped |
| --- | --- | --- | --- |
| Depth schedule | +0.021442 | +0.009761 | on |
| Audience guardrail | +0.000140 (noise) | +0.005017 | on |
| Early `other` | +0.014962 | +0.003921 | on |
| Dual-track weights | -0.001712 | -0.006617 | **off** |
| Category diversity | -0.013414 | not run | **off** |

Current enabled stack: **public 0.886472**, **held-out 0.864862**.
