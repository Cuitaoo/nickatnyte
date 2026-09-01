# Conversational E-Commerce Search Agent

A multi-turn shopping agent for the TechJam Conversational Search Challenge.
It treats **when to show results** as a ranking decision, not a formality: each
turn it measures how far the leading candidate has separated from the runner-up
and returns a list sized to that confidence, asking a question instead when the
ranking has not separated.

| | Technical score | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Public set, 200 sessions | **0.914274** | 0.990 | 0.900581 | 3.545 |
| Held-out set, 200 unseen targets | **0.880493** | 0.960 | 0.854976 | 3.800 |

Run-to-run variation with the model enabled is about ±0.0002 on the public set
and ±0.004 on the held-out set; treat smaller differences as noise.

---

## Quick start

```bash
pip install -r requirements.txt -r requirements-vector.txt
```

One command runs the agent in the official harness:

```bash
OPENAI_API_KEY=sk-... python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl
```

**No environment setup is required.** Drop the API key and it still runs — see
*Network access* below. If your number disagrees with the table above:

```bash
PYTHONPATH=. python tools/verify_setup.py --catalog data/catalog.jsonl
```

## Requirements

- **Python 3.14** (developed and measured on 3.14.4; 3.11+ should work)
- `requirements.txt` — langchain-core, langchain-openai, langgraph, pydantic
- `requirements-vector.txt` — sentence-transformers and its dependencies
- `data/catalog.jsonl` — the official 50,000-product catalogue, supplied by the
  organizer and deliberately not vendored here
- `data/vector_index/` — prebuilt embeddings, committed, validated by a sha256
  of the catalogue they were built from

## Configuration is in the code, not in a `.env`

Every tuned constant lives in [`starter/config.py`](starter/config.py) as a
single table of 85 named settings. The only thing that comes from the
environment is `OPENAI_API_KEY`.

**This was a bug fix, not a preference.** Our configuration used to live in a
`.env` file. `.env` is gitignored, so it never reached a fresh clone, and
teammates checking out the branch measured **0.857069** instead of 0.914274 — a
gap of 0.057 — because every feature silently fell back to a code default that
was off. Nothing in the repository was wrong; the repository just wasn't the
thing being measured. Compiling the configuration in makes the checked-out
commit and the measured system the same object, which is what the
reproducibility rules ask for.

Precedence is `os.environ` → `DEFAULTS` → the call site's own default, so an
exported variable still wins. That is how every ablation below works, and how
`tests/` pins a value without touching the file. Two tests keep it honest: one
fails if any `starter/` module reads `os.getenv` directly and bypasses the
table, another fails if the table carries a default nothing reads.

`.env.example` is now a stub holding the API key and a pointer here.

## Entry point

`starter/agent.py` exports `Agent` with the required interface:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
```

The response payload is exactly these four keys. Diagnostics that would
otherwise widen the contract — the per-turn strategy record, the LLM routing
decision, distilled profile deltas — are exposed as separate accessors
(`last_decision`, `last_parse_decision`, `profile_updates`) and never appear in
the payload.

### Output rules

All five are checked against a real 200-session run rather than asserted:

```bash
PYTHONPATH=. python tools/check_output_contract.py --catalog data/catalog.jsonl
```

It wraps the real `Agent` in a validating proxy, drives it through the
unmodified evaluator loop, and exits non-zero on any violation. Last run:
**200 sessions, 709 turns, no violations.**

| rule | how it holds |
| --- | --- |
| `message` is a string | always constructed from a template or a deterministic explanation |
| `ask_attribute` is an allowed attribute or `null` | drawn from the ten-value vocabulary; `null` when recommending without asking |
| `recommendations` ordered best to worst | emitted in fused-score order; every later stage permutes rather than re-sorts |
| only the first 10 valid unique `parent_asin` are scored | we return at most 10, deduplicated, all catalogue-valid |
| `usage` reports non-negative token counts | integers, zero on the offline path |

---

## Network access and offline fallback

**The agent does not require live credentials.** With `OPENAI_API_KEY` unset it
uses the deterministic parser throughout and makes no network calls at any
point. Set `OPENAI_ENABLED=false` to force that path even when a key is
present.

With the model enabled, network failure is handled rather than fatal. Every LLM
path catches its own exceptions and falls back to the deterministic parser, and
this was verified against a real failure: during development the API returned
HTTP 429 `insufficient_quota` for an entire 200-session run, and the evaluation
completed normally with valid results and zero tokens reported.

Both local models — the cross-encoder and the embedding model — load from the
HuggingFace cache with `local_files_only=True`, so no download is attempted at
run time.

Measured contribution of the model path: **+0.0002 public, +0.0038 held-out**.
It is enabled because it is cheap and slightly positive, not because the agent
depends on it.

## Latency, tokens, and cost

Measured on an Apple Silicon laptop, CPU only.

**Startup, once per process**

| stage | time |
| --- | ---: |
| Catalogue index build | 0.29 s |
| Agent construction (SQLite FTS5 + vector index) | 2.95 s |
| Cross-encoder load (lazy, on first use) | 3.65 s |
| **Total cold start** | **6.89 s** |

**Per turn** (`respond()`, 300 turns, no API call)

| | latency |
| --- | ---: |
| p50 | 168 ms |
| p95 | 277 ms |
| p99 | 455 ms |

**Turns routed to the model** add one round trip: p50 **3.2 s**, mean 3.6 s,
max 5.4 s, at roughly 1,400 tokens per call.

**Tokens.** The ambiguity gate routes only turns the deterministic parser
cannot read confidently — about 32 of 800 turns. Measured on the headline run:

| | prompt | completion | total | per session |
| --- | ---: | ---: | ---: | ---: |
| Gated (shipped) | 42,430 | 7,922 | **50,352** | 252 |
| Every turn routed | 925,949 | 101,254 | 1,027,203 | 5,136 |

A **95% reduction** for a score that does not move.

**Cost.** We do not know the rate the organizer will score under, so the
estimate is given as arithmetic you can re-run with your own numbers. The full
800-session hidden set is 4× the table above: roughly **170k prompt + 32k
completion tokens**.

| reference rate (input / output per 1M) | 200 sessions | 800 sessions |
| --- | ---: | ---: |
| $0.15 / $0.60 | $0.011 | $0.044 |
| $1.25 / $10.00 | $0.132 | $0.529 |

**Under one US dollar to score the entire hidden set** at either rate. Ungated,
the same run would cost roughly $8.68 at the higher rate — the gate is the
difference between a rounding error and a real bill.

---

## Submission compliance

Each rule checked with a command, not an assertion.

**The official harness is unmodified.** `evaluator/` is byte-identical to the
published version:

```bash
git diff 2a6cc8e -- evaluator/     # no output
```

`starter/` never imports `evaluator`, so the agent has no dependency on it:

```bash
grep -rn "import evaluator" starter/     # no output
```

**No secrets.** `.env` is gitignored and never committed; `.env.example` ships
an empty key. A test fails if any value in `starter/config.py` looks like a
credential.

```bash
git ls-files | xargs grep -l "sk-" 2>/dev/null    # no output
```

**No organizer-only files or private evaluation data.** `organizer/`, `secure/`
and `data/catalog.jsonl` are gitignored; the catalogue is supplied by the
organizer at run time.

**No privileged host access, no undeclared external services.** The only
network dependency is the OpenAI API, which is optional — see *Network access*
above. Both local models load with `local_files_only=True`.

**Output contract.** `respond()` returns exactly `message`, `ask_attribute`,
`recommendations`, `usage`. Three tests pin the key set, so widening it fails
the suite rather than reaching a submission.

### File layout against the recommended one

The rules suggest `submission/{agent.py, requirements.txt, README.md, src/}`.
Ours maps onto it directly:

| recommended | here |
| --- | --- |
| `agent.py` | `starter/agent.py` — exports `Agent` |
| `src/` | `starter/` — helper modules |
| `requirements.txt` | `requirements.txt` + `requirements-vector.txt` |
| `README.md` | this file |

## Settings that most change the score

All of these ship at the value shown in `starter/config.py`. The column on the
right is what you get by exporting the opposite — that is how each was
measured.

| setting | ships as | effect if overridden |
| --- | --- | --- |
| `TECHJAM_RERANK_ENABLED` | `true` | no semantic reranking |
| `TECHJAM_VECTOR_ENABLED` | `true` | no vector recall |
| `TECHJAM_DEPTH_MODE` | `hybrid` | list sized by turn, not confidence |
| `TECHJAM_DEPTH_NORMALIZED_MARGIN` | `true` | thresholds become scale-dependent |
| `TECHJAM_RATING_COUNT_COEF` | `0.030` | quality signal effectively off |
| `TECHJAM_IDF_WEIGHTING` | `true` | boilerplate outweighs distinctive terms |
| `TECHJAM_AUDIENCE_GUARDRAIL` | `true` | department mismatches survive reranking |
| `TECHJAM_STAGED_FILTER` | `true` | buying stops filtering |
| `TECHJAM_CONFIDENCE_CONTROLLER` | `true` | deferral reverts to turn count |
| `OPENAI_ENABLED` | `true` | deterministic parsing only (valid, −0.0002) |

Overriding all of them back to the old code defaults scores 0.857069.

## How the held-out set was built

`data/synthetic_pop.jsonl` holds out the **target product**: every session uses
a catalogue product that appears as no public target, so no constant in this
repository was fitted on it. Scenario mix, difficulty mix and mean constraints
per session match the public set, and targets are drawn quantile-for-quantile
against the public rating-count distribution (median 6,856 against the public
set's 6,846), so a score gap reads as overfitting rather than as a harder set.

```bash
PYTHONPATH=. python tools/build_synthetic_set.py \
  --catalog data/catalog.jsonl --match-popularity --output data/synthetic_pop.jsonl
```

`data/synthetic_set.jsonl` is an earlier uniform draw from the catalogue. It is
kept because it correctly caught a profile-expansion overfit, but it is
mis-specified on popularity (median 12 ratings) and returns false negatives for
anything popularity-correlated. Validate popularity-sensitive changes against
`synthetic_pop.jsonl`.

---

## Method

### Retrieval

Eight lexical routes over SQLite FTS5 (BM25), fused by weighted reciprocal
rank, plus two vector routes used for recall when lexical evidence is thin.

| route | fields | weight |
| --- | --- | ---: |
| latest message | all | 2.484 (1.618 after an override) |
| attribute | title, details, store, description | 2.127 |
| exact phrase | title, features, details, description | 1.714 |
| identifier | title, features, details, description | 1.714 |
| feature / use case | features, details, description, categories | 1.258 |
| relaxed | all | 1.245 |
| category | title, categories | 0.783 |
| synonym | title, categories | 0.489 |

Fused ranks are combined with additive evidence: confirmed preferences (1.302),
exact phrase (2.176), exact identifier (6.000), category (1.059), a penalty for
matching a rejected value (−2.393), a multi-route agreement bonus, and quality
terms.

Every constraint value is **inverse-document-frequency damped** before it
scores, so catalogue-wide boilerplate cannot outweigh distinctive evidence:
`Imported` counts 0.54, `100% Cotton` 0.71, a model number 1.00.

### Buying and browsing take different paths

Intent is classified per turn and changes what the pipeline does.

**Buying.** Constraints are typed by how the evidence arrived — volunteered or
corrected is *hard*, an answer to a question we asked is *soft*. Hard
constraints **filter** the candidate pool rather than nudging it, with two
safeguards: a product is dropped only when it can be shown to violate, so
missing catalogue metadata never excludes anything; and the filter surrenders
its least reliable constraint whenever the survivors fall below a floor, rather
than starving. Free-text attributes never filter. Constraint strength is
detected in 88% of buying sessions after one turn.

**Browsing.** Nothing has been stated, so nothing is filtered; vector recall
carries more of the load and the agent converges by asking.

### Confidence decides how much to show

The evaluator freezes reciprocal rank at the target's first appearance, so
returning a long list before the ranking has separated permanently banks a poor
rank. Each turn we measure the gap between the best and second-best score as a
fraction of the best — scale-free, so it survives any retuning of the weights.

Calibrated against how often the leader really is the target, over 70 sessions
× 5 turns:

| separation | turns | leader is the target | returned |
| --- | ---: | ---: | ---: |
| ≥ 0.30 | 48 | 97.9% | 10 |
| 0.20 – 0.30 | 39 | 92.3% | 10 |
| 0.10 – 0.20 | 60 | 66.7% | 2 |
| < 0.10 | 203 | 32.0% | 1 |

From turn 5 the list always widens to 10, because never answering costs the
full 11-turn miss penalty against Hit@10's 0.50 weight.

### Dialogue

Slots accumulate across turns; an intent override erases and rewrites rather
than merging, and negations are tracked separately from preferences. The
open-ended question is promoted the moment structured facets stop working —
straight after an override, or once the shopper has declined two attributes.

A confidence controller selects among *recommend now*, *recommend while
asking*, *ask only*, *broaden retrieval* and *relax constraint*. It may only
release a turn the deferral rule would withhold, never the reverse, so it cannot
make time-to-conversion worse.

Every turn emits one inspectable `StrategyDecision`: intent, hard and soft
constraints, routes fired, pool size, separation, depth, and the next question.

### Guardrails and explanations

After reranking, an audience guardrail penalises department mismatch — the
catalogue skews 2.4:1 toward women's, so men's and boys' queries were being
outranked by products matching everything except the shopper. Wrong-audience
results in the top ten fall from **16.6% to 7.9%**.

Recommendation messages are generated deterministically from recorded match
evidence — *"Recommended because it matches cotton and relaxed fit"* — never
from a model and never from ranking scores, so the agent cannot claim a
preference the product does not match.

### Models

| purpose | model | notes |
| --- | --- | --- |
| State interpretation | `gpt-5.6-luna` | gated; ~32 of 800 turns |
| Semantic reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | local, CPU |
| Vector recall | `BAAI/bge-small-en-v1.5` | local, prebuilt index |

---

## Known trade-offs and open questions

Where a design choice has a cost or rests on an assumption, we state it with the
number attached.

**Result depth is tuned to the evaluation's stopping rule.** Reciprocal rank is
fixed at the target's first appearance, so a confidence-sized list is worth
more than a full one. Removing depth capping entirely scores **0.858477**
(MRR 0.662), which is the system's ranking quality on its own; the difference
is the contribution of returning results only once the ranking has separated.
The behaviour has a product reading — show a shortlist while you are still
narrowing — but the specific sizing is fitted to this metric, and a deployment
optimising for browsing would show ten results throughout.

**The quality signal assumes the hidden set is sampled like the public one.**
Public targets have a median of 6,846 ratings against the catalogue's 12,
because the benchmark anchors on the final purchased record and purchases skew
popular. Weighting for that is worth **+0.0147**. We validated it on a holdout
matched to the same popularity distribution (**+0.0186**), so it generalises
across products — but if the hidden set is drawn uniformly from the catalogue
instead, the setting costs −0.0031. `TECHJAM_RATING_COUNT_COEF=0.000335`
reverts it.

**Two public sessions are not solvable from the information disclosed.**
`public_0144`'s target carries no material or closure metadata, and the shopper
discloses only "polyester, imported, zipper" — true of every product in its
pool. It reaches rank 19–21 and stays there.

**Semantic reranking is currently neutral.** Disabling the cross-encoder
measures 0.914952 against 0.914524, so its contribution sits inside run-to-run
variation. It remains in the pipeline as the semantic reranking stage, and the
lexical and guardrail stages are carrying the current score.

**Separation thresholds were calibrated on public sessions.** The bands come
from 70 public sessions and were then validated on the held-out set, which
passed — but the calibration data itself was in-distribution.

**Query rewrite and profile-driven personalisation did not pay.** Both were
built, measured on both sets, and are switched off; see the table below.

## What we built and did not keep

Every feature was measured on both sets and kept only if it earned its place.
These ship switchable and off, each carrying its measurement as a comment in
`starter/config.py`:

| rejected | public | held out | why |
| --- | ---: | ---: | --- |
| Dual-track weight multipliers | −0.0017 | −0.0066 | bought MRR by shedding Hit@10 |
| Result diversity | −0.0134 | — | spreading pushed targets out of the top 10 |
| Semantic profile expansion | +0.0003 | −0.0060 | tags match half the catalogue |
| Query rewrite (deterministic) | −0.0029 | — | pipeline already normalises the query |
| Query rewrite (LLM, gated) | −0.0028 | — | matched the regex, cost 6,765 tokens |
| Over-generality cutoff | −0.0002 | — | withholding delays without narrowing |
| Question steering on overload | −0.0039 | −0.0018 | split power is not answerability |
| Override as reinforcement | −0.0002 | — | the erasure was doing useful work |
| Re-ask when questions exhausted | 0.0000 | — | a declined attribute stays declined |

`docs/evaluations/ct/improvement-backlog.md` records the reasoning behind each.

## Verifying the numbers in this README

Every figure above is reproducible. Run these in a clean shell — an exported
variable overrides the shipped configuration, which is the one way to get a
different number from the same commit.

```bash
# 1. setup is correct before anything else
PYTHONPATH=. python tools/verify_setup.py --catalog data/catalog.jsonl

# 2. the headline public score            -> 0.914274 (+/- 0.0002 with the model on)
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl

# 3. the held-out score                   -> 0.880493
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/synthetic_pop.jsonl

# 4. ranking quality without depth capping -> 0.858477
TECHJAM_DEPTH_MODE=turn TECHJAM_DEPTH_SCHEDULE= \
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl

# 5. the popularity assumption, both ways  -> +0.0147 / -0.0031
TECHJAM_RATING_COUNT_COEF=0.000335 python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl

# 6. that it runs with no network at all
OPENAI_ENABLED=false python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl

# 7. every output rule, over a real 200-session run
PYTHONPATH=. python tools/check_output_contract.py --catalog data/catalog.jsonl

# 8. the test suite
python -m unittest discover -s tests
```

The score decomposes as `0.50 x Hit@10 + 0.30 x MRR + 0.20 x Efficiency`, where
`Efficiency = (11 - MTTC) / 10`. For the public run:

```
0.50 x 0.990 + 0.30 x 0.900581 + 0.20 x 0.7455 = 0.914274
```

Each rejected feature in the table above is reproducible the same way by
exporting its setting.

## Repository layout

```
starter/            the agent
  agent.py            entry point, exports Agent
  config.py           the shipped configuration, 85 settings
  retrieval.py        routes, fusion, scoring, guardrails
  confidence.py       separation signals and strategy selection
  constraints.py      constraint strength and staged filtering
  audience.py         department guardrail
  questions.py        clarification policy
  orchestration.py    per-turn strategy record
  explain.py          deterministic recommendation explanations
  profile_memory.py   dialog distillation into profile deltas
  llm_agent.py        gated LLM state interpretation
evaluator/          official harness, unmodified (see Submission compliance)
tools/
  verify_setup.py           reproducibility diagnostic
  check_output_contract.py  output-rule checker
  build_synthetic_set.py    held-out set builder
docs/evaluations/ct/  measurement log
tests/              343 tests
```

## A demonstrated session

`public_0103`, an intent-override session. Left column is the shopper; the
right is what the agent returned and why.

| turn | shopper | separation | returned | asked |
| ---: | --- | ---: | ---: | --- |
| 1 | "I'm looking for Underwear Undershirts. Imported" | 0.02 | 1 | material |
| 2 | "For that, what matters is: cotton; 100% Cotton." | 0.08 | 1 | color |
| 3 | "For that, what matters is: color: white." | 0.01 | 1 | feature |
| 4 | "Actually, ignore my earlier preference. What I need is: cotton." | 0.03 | 1 | other |
| 5 | — | 0.03 | 10 | target found at rank 3 |

The separation never rises, so the agent shows a single best guess and keeps
asking rather than banking a poor rank; at turn 5 the list widens and the
target lands at rank 3. The override on turn 4 rewrites the material slot
rather than merging it, and the audience and IDF guardrails run on every turn.

Reproduce any session with:

```bash
PYTHONPATH=. python tools/demo_profile_memory.py
```

## Team contributions

Derived from `git log`; both authors reviewed and measured each other's work
before it merged.

| area | primary |
| --- | --- |
| Multi-route retrieval, fusion and scoring weights | xdJanaut |
| Confidence controller, depth policy, orchestration record | xdJanaut |
| Audience guardrail, IDF damping, staged constraint filtering | xdJanaut |
| Held-out set construction and the measurement discipline | xdJanaut |
| LLM state interpretation and the ambiguity gate | Cui Tao |
| Profile distillation and scenario query expansion | Cui Tao |
| Cross-encoder reranking and vector index | Cui Tao |
| Explanations, clarification policy, README | xdJanaut |

## Tests

```bash
python -m unittest discover -s tests
```

343 tests, no environment setup required.
