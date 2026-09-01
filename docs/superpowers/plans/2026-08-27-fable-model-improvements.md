# Fable Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the offline technical score above the current 0.7848 by fixing the five known intent-override misses, repairing dead budget matching, tuning ranking weights against held-out data, adding a synonym-expansion retrieval hedge, and adding an optional LLM reranker for MRR — while keeping the deterministic offline path fully working.

**Architecture:** All ranking constants move into a `RetrievalWeights` dataclass so a tuning harness can search them with a train/holdout split of the 200 public sessions. Evidence handling keeps raw compound values (e.g. `90% cotton, 10% others`) as exact-phrase search evidence alongside canonical atomic values. The question policy learns to re-ask attributes whose only evidence came from a correction and to repeat `other` asks until the shopper is exhausted. A curated synonym route hedges lexical mismatch; an optional LLM reranker reorders the top candidates when OpenAI is enabled.

**Tech Stack:** Python 3.10+, sqlite3 FTS5, unittest, langchain-openai (optional paths only).

## Global Constraints

- Offline mode (`OPENAI_ENABLED=false`) must stay fully deterministic — no network, no new heavy dependencies (no torch/sentence-transformers).
- No changes to `evaluator/local_evaluator.py` scoring logic (submission rules forbid evaluator modification; the evaluator file may only be touched by the tuning harness *importing* it).
- Tests run with `unittest` (`/Users/justin/nickatnyte/.venv/bin/python -m unittest discover -s tests -q`); repo has no pytest.
- Full offline eval command (catalog lives in the main checkout):
  `OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m evaluator.local_evaluator --catalog /Users/justin/nickatnyte/data/catalog.jsonl --dataset data/public_set.jsonl --output <out.json>`
- Branch: `fable-model`. Commit after every task.
- Regression gate for every task: overall offline score must not drop below the pre-task value; scenario Hit@10 must not drop.

---

### Task 1: Budget "around" matching

The evaluator discloses budgets as `budget around $59.99` ([evaluator/local_evaluator.py:63](../../evaluator/local_evaluator.py)), but `CatalogRetriever._preference_matches` only handles "under/below/less than/up to" and returns `False` otherwise, and `parse_preference_fallback` only regex-captures the "under…" family. Budget evidence is currently dead weight.

**Files:**
- Modify: `starter/retrieval.py` (`_preference_matches`)
- Modify: `starter/preference_tool.py` (fallback budget regex)
- Test: `tests/test_retrieval.py`, `tests/test_preferences.py`

**Interfaces:**
- Produces: `_preference_matches("budget", "around $60", product)` → `True` when `0.75 * 60 <= price <= 1.25 * 60`.

- [x] **Step 1: Write failing tests**

In `tests/test_retrieval.py` (uses existing `CatalogRetriever._preference_matches` staticmethod; products are plain metadata dicts):

```python
class BudgetMatchTest(unittest.TestCase):
    def test_around_budget_matches_price_band(self) -> None:
        product = {"price": 55.0}
        self.assertTrue(CatalogRetriever._preference_matches("budget", "around $60", product))
        self.assertTrue(CatalogRetriever._preference_matches("budget", "budget around $60", product))

    def test_around_budget_rejects_far_price(self) -> None:
        self.assertFalse(CatalogRetriever._preference_matches("budget", "around $60", {"price": 200.0}))
        self.assertFalse(CatalogRetriever._preference_matches("budget", "around $60", {"price": 10.0}))

    def test_under_budget_still_matches(self) -> None:
        self.assertTrue(CatalogRetriever._preference_matches("budget", "under $60", {"price": 55.0}))
```

In `tests/test_preferences.py`:

```python
class FallbackBudgetTest(unittest.TestCase):
    def test_fallback_captures_around_budget(self) -> None:
        state = ShoppingState.new("s", {})
        patch = parse_preference_fallback("For that, what matters is: budget around $59.99.", state)
        budgets = [item.value for item in patch.set_preferences if item.attribute == "budget"]
        self.assertEqual(budgets, ["around $59.99"])
```

Note: the direct-answer path (`previous_ask_attribute == "budget"`) already stores the raw text; the new assertion covers the unsolicited path where no question was pending.

- [x] **Step 2: Run tests, verify the new ones fail**
- [x] **Step 3: Implement**

In `starter/retrieval.py`, replace the budget branch of `_preference_matches`:

```python
        if attribute == "budget":
            match = NUMBER_RE.search(value)
            price = product["price"]
            if not match or price is None:
                return False
            budget = float(match.group(0))
            if any(term in value for term in ("under", "below", "less than", "up to")):
                return price <= budget
            if "around" in value or "about" in value or "approximately" in value:
                return 0.75 * budget <= price <= 1.25 * budget
            return False
```

In `starter/preference_tool.py`, widen the fallback budget capture (keep existing normalization for the bounded family):

```python
    budget = _first_match(
        r"\b((?:under|below|less than|up to|around|about)\s*\$?\s*\d+(?:\.\d{1,2})?)\b",
        lowered,
    )
    if budget:
        budget = re.sub(
            r"\b(under|below|less than|up to|around|about)\b\s*\$?\s*(\d+(?:\.\d{1,2})?)",
            r"\1 $\2",
            budget,
        )
        values.append(PreferenceValue(attribute="budget", value=budget))
```

- [x] **Step 4: Run full suite, verify pass**
- [x] **Step 5: Commit** `fix: match around-style budget preferences`

---

### Task 2: Retain compound evidence for exact-phrase retrieval

Canonicalization (`_canonical_preference_values`) collapses `90% cotton, 10% others` to `cotton`, discarding the near-unique composition phrase that discriminates the target in the hard clothing bucket. Keep the canonical atomic values for matching, but also retain the raw compound value as a search term, and feed multi-token search terms into the exact-phrase route.

**Files:**
- Modify: `starter/preference_tool.py` (`apply_preference_patch` set_preferences loop, `_direct_answer_values` callers keep working unchanged)
- Modify: `starter/retrieval.py` (`_route_specs` exact_phrase values)
- Test: `tests/test_preferences.py`, `tests/test_retrieval.py`

**Interfaces:**
- Produces: after applying a patch with `PreferenceValue(attribute="material", value="90% cotton, 10% others")`, `state.preferences["material"] == ("cotton",)` and `"90% cotton, 10% others" in state.search_terms`.
- Produces: `_route_specs` includes multi-token `state.search_terms` entries in the `exact_phrase` route values.

- [x] **Step 1: Write failing tests**

`tests/test_preferences.py`:

```python
class CompoundEvidenceTest(unittest.TestCase):
    def test_compound_material_keeps_raw_phrase_as_search_term(self) -> None:
        state = ShoppingState.new("s", {})
        patch = PreferencePatch(
            set_preferences=[PreferenceValue(attribute="material", value="90% Cotton, 10% Others")]
        )
        updated = apply_preference_patch(state, patch)
        self.assertEqual(updated.preferences["material"], ("cotton",))
        self.assertIn("90% cotton, 10% others", updated.search_terms)

    def test_atomic_material_adds_no_extra_search_term(self) -> None:
        state = ShoppingState.new("s", {})
        patch = PreferencePatch(
            set_preferences=[PreferenceValue(attribute="material", value="cotton")]
        )
        updated = apply_preference_patch(state, patch)
        self.assertNotIn("cotton", updated.search_terms)
```

`tests/test_retrieval.py` — end-to-end: a catalog product whose `details` contain `90% Cotton, 10% Others` must outrank a plain cotton product once the compound term is in `search_terms`. Add two products to the test fixture catalog (same category, both "cotton sweatshirt"), one with the compound composition in details; build state with `preferences={"material": ("cotton",)}, search_terms=("90% cotton, 10% others",)`, search, and assert the compound product ranks first.

- [x] **Step 2: Run tests, verify the new ones fail**
- [x] **Step 3: Implement**

In `apply_preference_patch`, inside the `set_preferences` loop, after appending canonical values:

```python
        canonical_values = _canonical_preference_values(attribute, value)
        for canonical_value in canonical_values:
            ...existing removed/bucket logic...
        if canonical_values != [value] and len(TOKEN_RE.findall(value)) >= 2:
            _append_unique(search_terms, value)
```

(The guard means: only when canonicalization actually rewrote the value, and only multi-token phrases.) Mirror the same retention in `_normalized_patch_values`-driven evidence by appending the raw value to the matching evidence `terms` — extend `terms_by_attribute[attribute]` assignment so the raw compound (already in `accepted_patch_terms`? it is NOT — it came from set_preferences, not patch.search_terms). Simplest correct wiring: collect retained raw phrases in a local list during the set_preferences loop and extend both `search_terms` (already done above) and, when the attribute has an evidence entry, that entry's `terms`. Implementation detail: build `raw_phrases_by_attribute: dict[str, list[str]]` in the loop and merge into `terms_by_attribute` before evidence construction.

In `starter/retrieval.py` `_route_specs`, extend the exact-phrase route values:

```python
            _RouteSpec(
                "exact_phrase",
                tuple(
                    value
                    for value in [*functional_values, *state.search_terms, latest_message]
                    if len(lexical_terms([value])) >= 2
                ),
                ("title", "features", "details", "description"),
                1.60,
                True,
            ),
```

- [x] **Step 4: Run full suite, verify pass**
- [x] **Step 5: Commit** `feat: retain compound preference phrases for exact-phrase retrieval`

---

### Task 3: Re-ask policy for correction-only attributes and repeatable `other`

Two question-policy changes in `starter/questions.py`:
1. An attribute present in `state.preferences` is currently never asked. Allow asking it when (a) it was never actually asked (`not in asked_attributes`) and (b) every evidence entry for it has `source_kind == "correction"` — i.e. the only signal came from an override, and the shopper may hold a more specific constraint (the composition string).
2. `other` is excluded after one ask via `asked_attributes`. Allow `other` to repeat as long as it is not in `no_preference_attributes` (the evaluator answers "I don't have an additional preference for other" when exhausted, which the fallback parser converts to a no-preference marker, terminating the loop naturally).

**Files:**
- Modify: `starter/questions.py` (`choose_clarification` exclusion logic; signature gains access to evidence via `state.preference_evidence` — already on state)
- Modify: `starter/agent.py` (do not append `other` to `asked_attributes`, or keep appending but stop excluding on it — choose: stop excluding `other` via asked list in questions.py; agent unchanged)
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: `PreferenceEvidence.source_kind` from `starter/state.py`.
- Produces: unchanged `choose_clarification(state, turn, diagnostics) -> tuple[str, str | None]`.

- [x] **Step 1: Write failing tests**

```python
class ReAskPolicyTest(unittest.TestCase):
    def _diag(self, attribute: str) -> AttributeDiagnostic:
        return AttributeDiagnostic(attribute=attribute, coverage=0.8, disagreement=0.6, relevance=0.8)

    def test_correction_only_attribute_can_be_asked(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            preference_evidence=(
                PreferenceEvidence(attribute="material", values=("cotton",), source_kind="correction"),
            ),
        )
        message, attribute = choose_clarification(state, 4, {"material": self._diag("material")})
        self.assertEqual(attribute, "material")

    def test_clarified_attribute_is_not_reasked(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            asked_attributes=("material",),
            preference_evidence=(
                PreferenceEvidence(attribute="material", values=("cotton",), source_kind="clarification"),
            ),
        )
        message, attribute = choose_clarification(state, 4, {"material": self._diag("material")})
        self.assertNotEqual(attribute, "material")

    def test_other_repeats_until_no_preference(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            asked_attributes=("other",),
        )
        message, attribute = choose_clarification(state, 4, {})
        self.assertEqual(attribute, "other")
        exhausted = replace(state, no_preference_attributes=frozenset({"other"}))
        message, attribute = choose_clarification(exhausted, 4, {})
        self.assertIsNone(attribute)
```

- [x] **Step 2: Run tests, verify the new ones fail**
- [x] **Step 3: Implement**

In `choose_clarification`:

```python
    excluded = set(state.asked_attributes)
    excluded.discard("other")
    excluded.update(state.no_preference_attributes)
    correction_only = {
        attribute
        for attribute in state.preferences
        if attribute not in state.asked_attributes
        and all(
            item.source_kind == "correction"
            for item in state.preference_evidence
            if item.attribute == attribute
        )
        and any(item.attribute == attribute for item in state.preference_evidence)
    }
    excluded.update(attribute for attribute in state.preferences if attribute not in correction_only)
    if state.category:
        excluded.add("category")
```

(Keep the rest of the function unchanged; the final `if "other" not in excluded` branch now repeats `other`.)

- [x] **Step 4: Run full suite, verify pass**
- [x] **Step 5: Checkpoint eval — run the full offline eval; record score, expect the five hard intent-override sessions to flip.** Save to `docs/evaluations/fable-task3-offline.json`. Gate: overall score > 0.7848, intent_override Hit@10 ≥ 0.8333.
- [x] **Step 6: Commit** `feat: re-ask correction-only attributes and repeat other asks`

---

### Task 4: Boundary MRR diagnosis

Boundary MRR regressed 0.696 → 0.613 across the selective-evidence change while Hit@10 held at 1.0. Diagnose using the deterministic evaluator; fix if the cause is a ranking defect, or document why the trade is acceptable.

**Files:**
- Create: `docs/evaluations/boundary-mrr-diagnosis.md`
- Possibly modify: `starter/retrieval.py` or `starter/preference_tool.py` (only if a defect is found)
- Test: regression test capturing the fix if one is made

- [x] **Step 1: Extract per-session boundary ranks from the Task 3 checkpoint eval output** (the evaluator's `sessions` array has `best_rank` per sample). Compare against `docs/evaluations/iteration2-offline-results.json` from the pre-change run (regenerate at merge-base `4f396be`'s parent if the file does not contain per-session data — check first).
- [x] **Step 2: For the 3 worst rank regressions, replay the session turn-by-turn** with a scratch script in the scratchpad directory that instantiates `Agent(catalog)` offline and mirrors `evaluate()`'s message loop for one sample, printing `search_result.candidates[:10]` with `score_components` per turn. Identify which component moved the target down.
- [x] **Step 3: Fix or document.** If a defect (e.g. boundary "no preference, use your judgment" replies polluting `search_terms` or evidence retirement dropping useful terms): write a failing unit test reproducing the component-level cause, fix, run suite, re-run offline eval, gate as in Task 3. Otherwise write the diagnosis doc explaining the trade and stop.
- [x] **Step 4: Commit** `fix: restore boundary ranking quality` or `docs: diagnose boundary MRR trade`

---

### Task 5: RetrievalWeights config + tuning harness

Move hand-tuned constants into a frozen dataclass and add a reproducible random-search tuner with a stratified train/holdout split.

**Files:**
- Modify: `starter/retrieval.py` (add `RetrievalWeights`, thread through `CatalogRetriever` and `_route_specs`)
- Modify: `starter/agent.py` (`Agent(..., weights: RetrievalWeights | None = None)`)
- Create: `tools/__init__.py`, `tools/tune_weights.py`
- Test: `tests/test_retrieval.py` (weights plumbing), `tests/test_tuning.py` (split determinism, no-network)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class RetrievalWeights:
    rrf_offset: float = 8.0
    category_boost: float = 1.8
    confirmed_attribute_boost: float = 2.4
    exact_phrase_boost: float = 1.5
    removed_attribute_penalty: float = 4.0
    multi_route_bonus: float = 0.12
    rating_coef: float = 0.002
    rating_count_coef: float = 0.0002
    route_category: float = 1.40
    route_feature_use_case: float = 1.35
    route_exact_phrase: float = 1.60
    route_attribute: float = 1.25
    route_relaxed: float = 0.80
    route_latest: float = 1.50
    route_latest_override: float = 2.20
```

- Produces: `CatalogRetriever(catalog_path, weights=None)` stores `self.weights = weights or RetrievalWeights()`; `search()` and `_route_specs(state, latest_message, weights)` read every constant from it. Module-level constants remain as the dataclass defaults' single source (delete the standalone constants, keep names importable via `RetrievalWeights()` defaults in tests — update the two tests that import `CONFIRMED_ATTRIBUTE_BOOST`, `MAX_PROFILE_BOOST`, `RRF_OFFSET` to read from `RetrievalWeights` / keep `MAX_PROFILE_BOOST` module-level since profile boost is not tuned).
- Produces: `tools/tune_weights.py` CLI:
  `python -m tools.tune_weights --catalog ... --dataset data/public_set.jsonl --trials 60 --seed 7 --output docs/evaluations/weight-tuning.json`
  - Stratified split by `scenario_type`, 75/25, `random.Random(seed)`.
  - Builds ONE `CatalogRetriever`; per trial constructs `Agent` with `interpreter=None` and injects the shared retriever (`agent.retriever = shared`), samples each weight from a log-uniform multiplier in [0.5, 2.0] around the default, runs `evaluator.local_evaluator.evaluate` on the train split, keeps top 5 by train score, evaluates those on holdout, reports best-by-holdout alongside the default weights' train/holdout scores.
  - Writes JSON: `{"seed":…, "split_sizes":…, "default": {...}, "trials": [...], "best": {"weights": {...}, "train": …, "holdout": …}}`.

- [x] **Step 1: Write failing plumbing test** — `CatalogRetriever(path, weights=RetrievalWeights(category_boost=0.0))` yields a different ordering than defaults on the existing fixture (assert the category-boosted product no longer outranks). Plus `test_tuning.py::test_split_is_deterministic_and_stratified`.
- [x] **Step 2: Run tests, verify fail**
- [x] **Step 3: Implement weights plumbing** (mechanical: replace constant reads with `self.weights.…` / parameter pass into `_route_specs`).
- [x] **Step 4: Run full suite, verify pass; commit** `refactor: make retrieval weights configurable`
- [x] **Step 5: Implement `tools/tune_weights.py`; smoke-run with `--trials 2` on the full catalog to verify wiring; commit** `feat: add retrieval weight tuning harness`
- [x] **Step 6: Real tuning run** — `--trials 60 --seed 7`, save output JSON. If best holdout score beats default holdout score by ≥ 0.005, update `RetrievalWeights` defaults to the winner, run the full offline eval (all 200) and the unit suite, gate as before, commit `feat: adopt tuned retrieval weights` with the tuning JSON. Otherwise commit the JSON with `docs: record weight tuning results (defaults retained)`.

---

### Task 6: Synonym-expansion route (lexical-mismatch hedge)

Pure-python curated clothing synonym map; no new dependencies; deterministic. Expands category and latest-message tokens into a low-weight extra route so paraphrased vocabulary ("trainers", "jumper", "parka") still reaches catalog terms.

**Files:**
- Create: `starter/synonyms.py`
- Modify: `starter/retrieval.py` (`_route_specs` adds a `synonym` route, weight from `RetrievalWeights.route_synonym: float = 0.55`)
- Test: `tests/test_synonyms.py`, `tests/test_retrieval.py`

**Interfaces:**
- Produces: `expand_terms(terms: Iterable[str]) -> tuple[str, ...]` returning only NEW terms (synonyms not already in the input), deduplicated, lowercase.

- [x] **Step 1: Write failing tests**

```python
class SynonymTest(unittest.TestCase):
    def test_expands_known_synonyms(self) -> None:
        self.assertIn("sneaker", expand_terms(["trainers"]))
        self.assertIn("sweatshirt", expand_terms(["hoodie"]))

    def test_returns_only_new_terms(self) -> None:
        self.assertNotIn("hoodie", expand_terms(["hoodie"]))

    def test_unknown_terms_expand_to_nothing(self) -> None:
        self.assertEqual(expand_terms(["zzzz"]), ())
```

Retrieval test: fixture product titled "Fleece Hoodie Pullover"; state with `category="sweatshirt"`; assert it appears in candidates (it does not match any current route since no fixture text contains "sweatshirt").

- [x] **Step 2: Run tests, verify fail**
- [x] **Step 3: Implement `starter/synonyms.py`**

```python
from __future__ import annotations

from typing import Iterable

_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"sneaker", "sneakers", "trainer", "trainers", "shoe", "shoes"}),
    frozenset({"hoodie", "hoodies", "sweatshirt", "sweatshirts", "pullover"}),
    frozenset({"jacket", "jackets", "coat", "coats", "parka", "windbreaker"}),
    frozenset({"tee", "tshirt", "t", "shirt", "shirts", "top", "tops", "blouse"}),
    frozenset({"pants", "trousers", "slacks", "chinos"}),
    frozenset({"jumper", "sweater", "sweaters", "cardigan", "knitwear"}),
    frozenset({"purse", "handbag", "bag", "tote"}),
    frozenset({"jeans", "denim"}),
    frozenset({"boots", "boot", "booties"}),
    frozenset({"sandals", "sandal", "flip", "flops", "slides"}),
    frozenset({"leggings", "tights", "yoga"}),
    frozenset({"dress", "dresses", "gown", "sundress"}),
    frozenset({"shorts", "short"}),
    frozenset({"cap", "hat", "beanie"}),
    frozenset({"socks", "sock", "hosiery"}),
    frozenset({"underwear", "briefs", "boxers", "panties", "lingerie"}),
    frozenset({"swimsuit", "swimwear", "bikini", "trunks"}),
    frozenset({"scarf", "scarves", "shawl", "wrap"}),
    frozenset({"gloves", "glove", "mittens"}),
    frozenset({"belt", "belts", "waistband"}),
)

_INDEX: dict[str, frozenset[str]] = {term: group for group in _GROUPS for term in group}


def expand_terms(terms: Iterable[str]) -> tuple[str, ...]:
    seen = {str(term).lower() for term in terms}
    result: list[str] = []
    for term in list(seen):
        for synonym in sorted(_INDEX.get(term, frozenset())):
            if synonym not in seen and synonym not in result:
                result.append(synonym)
    return tuple(result)
```

In `_route_specs`, before the return, build:

```python
    synonym_terms = expand_terms(
        lexical_terms([*category_values, latest_message])
    )
```

and append `_RouteSpec("synonym", synonym_terms, ("title", "categories"), weights.route_synonym)` to the candidate tuple (the existing `if route.terms` filter drops it when empty). Add `route_synonym: float = 0.55` to `RetrievalWeights`.

- [x] **Step 4: Run full suite; full offline eval; gate (score must not drop). Commit** `feat: add synonym-expansion retrieval route`

---

### Task 7: LLM candidate reranker (MRR)

Optional second LLM call per turn: rerank the top 20 candidates against the accumulated preferences. Enabled only when the interpreter is enabled; every failure path falls back to the original order. Offline behavior is byte-identical to today.

**Files:**
- Create: `starter/reranker.py`
- Modify: `starter/agent.py` (integration + usage accounting)
- Test: `tests/test_reranker.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class RerankResult:
    ordering: tuple[str, ...]      # product_ids, best first, len == len(input)
    prompt_tokens: int
    completion_tokens: int

class CandidateReranker:
    def __init__(self, model: object) -> None: ...   # model: ChatOpenAI-compatible, .with_structured_output
    @classmethod
    def from_environment(cls) -> "CandidateReranker | None": ...  # None unless OPENAI_ENABLED and key set and OPENAI_RERANK_ENABLED != false
    def rerank(self, state: ShoppingState, latest_message: str,
               candidates: Sequence[RankedCandidate], metadata: Mapping[str, dict],
               limit: int = 20) -> RerankResult: ...
```

- Model call: `model.with_structured_output(RerankOutput)` where `RerankOutput(BaseModel): order: list[int]` (indices into the presented list). Prompt: system line "Rank the numbered products for this shopper, best match first. Return every index exactly once."; human message is JSON of `{"preferences": state.to_prompt_dict(), "latest_message": ..., "products": [{"index": i, "title": ..., "price": ..., "details_snippet": metadata[pid]["details"][:200]} ...]}`.
- Validation: returned `order` must be a permutation of `range(n)` — otherwise raise `InvalidRerank` (caught by agent → original order). Only the first `limit` candidates are sent; candidates beyond `limit` keep their relative order after the reranked head.
- Agent integration in `respond()` after `search_result` is computed and before `identifiers` slicing:

```python
        if self.reranker is not None and search_result.candidates:
            try:
                rerank = self.reranker.rerank(
                    state, str(user_message), search_result.candidates, self.retriever.metadata
                )
            except Exception:
                rerank = None
            if rerank is not None:
                prompt_tokens += rerank.prompt_tokens
                completion_tokens += rerank.completion_tokens
                search_result = replace_recommendations(search_result, rerank.ordering)
```

with a small helper `replace_recommendations(result, ordering)` in `retrieval.py` returning a new `SearchResult` whose `recommendations` follow `ordering` (filtered to known ids) and whose `candidates` are reordered to match. `Agent.__init__` gains `self.reranker = CandidateReranker.from_environment() if (openai_enabled is not False and interpreter is _AUTO_INTERPRETER) else None`, plus an injectable `reranker=` kwarg for tests.

- [x] **Step 1: Write failing tests** — fake model object whose `with_structured_output` returns a callable/`invoke`-able returning `RerankOutput(order=[2, 0, 1])` plus fixed `usage_metadata`; assert reordering, permutation validation (`order=[0, 0, 1]` → fallback path raises `InvalidRerank`), `from_environment` returns `None` when `OPENAI_ENABLED=false` (use `unittest.mock.patch.dict(os.environ, ...)`), and agent integration keeps original order on reranker exception.
- [x] **Step 2: Run tests, verify fail**
- [x] **Step 3: Implement `starter/reranker.py` and agent integration**
- [x] **Step 4: Run full suite; run full OFFLINE eval and assert score unchanged from Task 6 checkpoint (reranker must be inert offline). Commit** `feat: add optional LLM candidate reranker`
- [x] **Step 5 (only if the user has confirmed paid spend): OpenAI-enabled eval** comparing reranker on/off; record in the evaluations doc. Do NOT run paid calls without an explicit go-ahead recorded in the conversation.

---

### Task 8: Final evaluation and documentation

**Files:**
- Create: `docs/evaluations/fable-model-comparison.md`
- Create: `docs/evaluations/fable-model-offline-results.json`

- [x] **Step 1: Full offline eval on all 200 sessions**; write results JSON.
- [x] **Step 2: Write comparison doc**: table of baseline → selective-evidence (0.7848) → each fable-model task checkpoint, per-scenario metrics, list of remaining misses, reproduction commands, and an explicit "paid-mode validation pending" note if Step 7.5 was skipped.
- [x] **Step 3: Run the full unit suite one final time; verify all pass.**
- [x] **Step 4: Commit** `docs: record fable-model evaluation results`

## Self-Review Notes

- Spec coverage: (1)→Task 5, (2)→Tasks 2–3, (3)→Task 1, (4)→Task 7, (5)→Task 6, (6)→Task 4. All covered.
- Ordering rationale: cheap deterministic wins first (1–3) so the Task 4 diagnosis and Task 5 tuning run against the improved evidence pipeline; tuning before the synonym route so the new route's weight is added to an already-tuned base (its own weight uses a conservative default, tunable in a later pass).
- Paid-run gates: Tasks 7.5 requires explicit user confirmation; everything else is offline-only.
