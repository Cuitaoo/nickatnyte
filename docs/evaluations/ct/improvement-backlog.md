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

## 3. Confidence and over-generality controller

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
