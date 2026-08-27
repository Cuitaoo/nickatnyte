# Fable Model Comparison

Date: 2026-08-27. Branch: `fable-model` (based on the codex branch tip
`c0bc419`). All numbers are deterministic offline runs
(`OPENAI_ENABLED=false`) on the 200-session public set unless noted.

## Headline

| Stage | Score | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Codex/selective-evidence base (`c0bc419`) | 0.784771 | 0.940 | 0.559571 | 3.655 |
| + budget fix, compound evidence, re-ask policy (Task 1–3) | 0.809199 | 0.970 | 0.573329 | 3.390 |
| + synonym rescue route (Task 6, inert on public set) | 0.809199 | 0.970 | 0.573329 | 3.390 |
| + LLM reranker (Task 7, inert offline) | 0.809199 | 0.970 | 0.573329 | 3.390 |
| **+ tuned retrieval weights (Task 5)** | **0.822008** | **0.980** | **0.590694** | **3.260** |

## Final per-scenario metrics

| Scenario | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary (10) | 1.000 | 0.746786 | 4.300 |
| Browsing (80) | 0.9875 | 0.568492 | 2.900 |
| Buying (80) | 0.975 | 0.58745 | 2.850 |
| Intent override (30) | 0.966667 | 0.606521 | 4.966667 |

Baseline comparison for intent override: `0.433` (main) -> `0.833`
(selective evidence) -> `0.967` here.

## What changed

1. **Budget matching** (`30c6ff6`): the evaluator phrases budgets as
   "budget around $X", which previously never matched; now a +/-25% price
   band applies.
2. **Compound evidence retention** (`3ca6505`, `757dee4`): raw compound
   values such as `90% cotton, 10% others` are kept as exact-phrase search
   evidence alongside canonical atomic values; direct clarification answers
   preserve raw phrases; same-product corrections retire only conflicting
   clarification evidence; "no additional preference" replies are recognized
   instead of polluting search terms.
3. **Question policy** (`757dee4`): attributes whose only evidence came from
   a correction can be asked once; `other` repeats until the shopper reports
   nothing left. Four of the five hard intent-override misses flipped; the
   remaining miss (`public_0144`) is data-limited — its intent card exposes
   only non-discriminative constraints ("polyester", "100% Polyester",
   "Imported", "Zipper closure") for a catalog with thousands of matching
   jackets.
4. **Boundary MRR** (`d6e7878`): the previously reported boundary MRR drop
   was a cross-branch comparison artifact; against the true code ancestor,
   boundary metrics improved. See `boundary-mrr-diagnosis.md`.
5. **Tunable weights + random search** (`2058138`, `dd1f8a5`): all ranking
   constants live in `RetrievalWeights`; `tools/tune_weights.py` (seed 7,
   20 trials, stratified 150/50 train/holdout split) found weights winning
   on both splits (holdout 0.8352 vs 0.8058). Notable shifts: lower
   confirmed-attribute boost (2.4 -> 1.30), higher exact-phrase boost
   (1.5 -> 2.18), stronger attribute route, smaller RRF offset.
6. **Synonym rescue route** (`ff25205`): curated clothing synonyms fire only
   when the category route finds no rows — inert on the public set (whose
   vocabulary always matches the catalog), a hedge for organizer-added
   paraphrasing on the private set.
7. **LLM semantic layer** (`bde147e`, `f6caa98`): the interpreter prompt now
   translates shopper vocabulary and implied intent into catalog terms, and
   an optional reranker reorders the top 20 candidates per turn. Both are
   inert offline (verified byte-identical score) and active only with
   `OPENAI_ENABLED=true`.

## Threat model for the 800 private sessions

The frozen 50k catalog underlies both splits, so catalog-derived assets
(FTS index, tuned weights, synonym groups) transfer. The spec allows the
organizer to add natural-language paraphrasing; the LLM interpreter and
reranker are the primary hedge for that case, with the synonym rescue route
as the thin offline backstop. Dense/embedding retrieval was evaluated and
deliberately skipped: intent cards are built from catalog text, so
out-of-catalog vocabulary can only enter through paraphrasing, which the
LLM path covers with world knowledge no catalog-only method has.

## Local cross-encoder reranking: replace rejected, blend adopted

Two designs were measured; the architecture, not the idea, decided the
outcome.

**Rejected — replace the ordering.** A prototype `bge-reranker-base`
(int8 ONNX, 279MB) that *overwrote* the top-20 order with cross-encoder
scores collapsed the offline score to 0.653464 (Hit@10 0.885, MRR
0.247546). Top candidates are near-duplicate products whose discriminator
is exact constraint matching, which the boost pipeline already encodes;
replacing that ordering with textual-similarity opinions destroyed it.

**Adopted — blend into the score** (merged from
`fable-plus-cross-encoder`, authored by Cui Tao). A 23MB
`ms-marco-MiniLM-L6-v2` cross-encoder scores only the top 10 candidates and
adds `0.35 * minmax(score)` to the existing boost score, so it can reorder
only near-ties — exactly where the remaining MRR sits — and cannot disturb
confident decisions:

| Set | Lexical only | + blended cross-encoder |
|---|---:|---:|
| Public 200 | 0.822008 | **0.834251** (MRR 0.591 -> 0.632) |
| Synthetic 200 (unseen targets) | 0.828392 | **0.834063** (MRR 0.651 -> 0.670) |

Hit@10 and MTTC are unchanged on both sets: the blend only reorders within
the top 10. Gated behind `TECHJAM_RERANK_ENABLED` (default off) with
`local_files_only` model loading and fall-back-to-original-order on any
failure. Trade-off: enabling it requires the `requirements-vector.txt`
extras (`sentence-transformers`, i.e. a torch stack) and adds roughly one
CPU-second per turn; the pure-lexical mode remains the zero-dependency
guaranteed configuration.

## Generalization check: synthetic unseen targets

200 synthetic sessions were generated from catalog products never used as
public-set targets (uniform sample, seed 42, official 40/40/15/5 scenario
mix, evaluator's own intent-card synthesis). Offline result: score
**0.828392**, Hit@10 0.96, MRR 0.6506, MTTC 3.34 — on par with the public
set, so the 0.822 does not depend on the specific 200 public samples. The
residual (deliberate) dependence is on the simulator's message templates;
the LLM interpreter is the documented hedge for organizer-added
paraphrasing.

## Submission posture

**Offline-first.** The default configuration is `OPENAI_ENABLED=false`:
deterministic, no network, milliseconds per turn, 0.822 public score. The
LLM mode is documented as an optional enhancement only: a full LLM-enabled
evaluation is roughly 50-100x slower end to end (one interpreter call plus
an optional rerank call per turn, serial), which is a poor fit for an
organizer harness that may impose CPU/timeout restrictions - and official
scoring may disable network entirely. The reranker additionally defaults
off (`OPENAI_RERANK_ENABLED`) pending evidence that its gain justifies
doubling per-turn calls.

## Paid-mode validation: deliberately not completed

A full OpenAI-enabled evaluation (interpreter + reranker) was started twice
on 2026-08-27 and killed both times by explicit decision. The second run was
terminated after ~1h54m (~80-100% complete, est. ~1,400 API calls) once the
offline-first posture made the result non-load-bearing: a stack that is
50-100x slower end to end was never going to become the default
configuration regardless of its score. Consequences to keep in mind:

- The interpreter prompt (vocabulary/intent translation) and the reranker
  remain **unvalidated against live `gpt-5.6-luna`**. They are covered by
  unit tests with fake models and verified inert offline, nothing more.
- If the online mode is ever promoted beyond "optional upside", run the
  command below first and record score, wall time, and token usage — the
  submission rules require those disclosures.

## Reproduction

```bash
OPENAI_ENABLED=false python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl \
  --output docs/evaluations/fable-model-offline-results.json

OPENAI_ENABLED=false python -m tools.tune_weights \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl \
  --trials 20 --seed 7 --output docs/evaluations/weight-tuning.json
```

Paid mode (after exporting `OPENAI_API_KEY`):

```bash
OPENAI_ENABLED=true python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl \
  --output docs/evaluations/fable-model-openai-results.json
```
