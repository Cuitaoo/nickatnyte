# Selective Preference Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve intent-override retrieval by preserving clarification-confirmed evidence during same-product preference corrections while retaining complete resets for genuine product changes.

**Architecture:** Add an immutable, internal evidence ledger to `ShoppingState`, then make `apply_preference_patch` the single transition boundary that classifies ordinary updates, preference corrections, and product changes. Both the model tool and deterministic fallback continue to converge on this function; retrieval remains unchanged and consumes only the existing flattened state.

**Tech Stack:** Python 3, frozen dataclasses, Pydantic v2, LangChain/LangGraph, `unittest`, the repository's deterministic evaluator.

**Spec:** `docs/superpowers/specs/2026-08-27-selective-preference-evidence-design.md`

## Global Constraints

- Preserve the public `Agent.reset(...)`, `Agent.respond(...)`, and response payload contracts.
- Keep one model request and exactly one preference-tool call per turn.
- Do not add retrieval routes, change global ranking weights, edit evaluator logic, or encode target product IDs.
- Treat `user_profile` as soft context only; it must never create hard preference evidence.
- Keep `ShoppingState` immutable and all new state fields backward-compatible through defaults.
- Use `state.turn + 1` as the evidence turn so model and fallback paths have identical semantics without changing interpreter call signatures.
- Run OpenAI-backed evaluation only after every offline acceptance gate passes and after explicit spend approval.
- Preserve unrelated existing worktree changes; stage only files belonging to the task being committed.

---

### Task 1: Add the immutable evidence model

**Files:**
- Modify: `starter/state.py`
- Modify: `tests/test_preferences.py`

- [ ] **Step 1: Write the failing state-model test**

Add imports for `PreferenceEvidence` and construct a state with one record:

```python
def test_preference_evidence_is_immutable_and_hidden_from_model_prompt(self) -> None:
    evidence = PreferenceEvidence(
        attribute="feature",
        values=("machine washable",),
        terms=("machine washable",),
        source_turn=2,
        source_kind="clarification",
    )
    state = replace(
        ShoppingState.new("s1", {}),
        preference_evidence=(evidence,),
    )

    self.assertEqual(state.preference_evidence, (evidence,))
    self.assertNotIn("preference_evidence", state.to_prompt_dict())
    with self.assertRaises(FrozenInstanceError):
        evidence.source_turn = 3
```

Import `FrozenInstanceError` from `dataclasses`. Keeping provenance out of `to_prompt_dict()` prevents prompt growth and avoids changing model behavior.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest tests.test_preferences.PreferenceUpdateTest.test_preference_evidence_is_immutable_and_hidden_from_model_prompt -v`

Expected: import failure because `PreferenceEvidence` does not exist.

- [ ] **Step 3: Implement the evidence dataclass and state field**

In `starter/state.py`, add:

```python
EvidenceSource = Literal["unsolicited", "clarification", "correction"]


@dataclass(frozen=True)
class PreferenceEvidence:
    attribute: str
    values: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    source_turn: int = 0
    source_kind: EvidenceSource = "unsolicited"
```

Then add this defaulted field after `search_terms`:

```python
preference_evidence: tuple[PreferenceEvidence, ...] = ()
```

Do not expose the ledger in `to_prompt_dict()`.

- [ ] **Step 4: Run the focused test and the current preference suite**

Run: `OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest tests.test_preferences -v`

Expected: all preference tests pass.

- [ ] **Step 5: Commit the state model**

```bash
git add starter/state.py tests/test_preferences.py
git commit -m "feat: add preference evidence state"
```

---

### Task 2: Record ordinary and clarification evidence at the shared transition boundary

**Files:**
- Modify: `starter/preference_tool.py`
- Modify: `tests/test_preferences.py`

- [ ] **Step 1: Write failing tests for source classification and normalization**

Add two tests. The first applies a normal feature patch and expects `source_kind == "unsolicited"` and `source_turn == 1`. The second starts from `turn=1, previous_ask_attribute="material"`, applies `material="95% polyester, 5% spandex"`, and expects one clarification record with canonical values `("polyester", "spandex")` and `source_turn == 2`.

Use assertions equivalent to:

```python
self.assertEqual(
    updated.preference_evidence[-1],
    PreferenceEvidence(
        attribute="material",
        values=("polyester", "spandex"),
        source_turn=2,
        source_kind="clarification",
    ),
)
```

- [ ] **Step 2: Run the two focused tests and confirm they fail**

Run the new tests by their fully qualified names with `OPENAI_ENABLED=false`.

Expected: the flattened preferences update, but the evidence ledger remains empty.

- [ ] **Step 3: Add small normalization and evidence helpers**

In `starter/preference_tool.py`, import `PreferenceEvidence` and add helpers with these responsibilities:

```python
def _normalized_patch_values(
    patch: PreferencePatch,
) -> dict[str, tuple[str, ...]]:
    """Return allowed, canonical, non-category values grouped by attribute."""


def _evidence_source(
    state: ShoppingState,
    attributes: set[str],
    *,
    correction: bool,
) -> Literal["unsolicited", "clarification", "correction"]:
    if correction:
        return "correction"
    if state.previous_ask_attribute in attributes:
        return "clarification"
    return "unsolicited"
```

Build one evidence record per normalized attribute. When search terms clearly belong to a single structured attribute, attach only matching/contained terms to that record; otherwise add one `attribute="other"` lexical record. Never record empty values or empty terms.

- [ ] **Step 4: Append evidence during ordinary patch application**

Initialize with `evidence = list(state.preference_evidence)`. After validating/canonicalizing patch values, append records using `source_turn=state.turn + 1`. Return `preference_evidence=tuple(evidence)` in `replace(...)`.

Do not change reset behavior in this task. Removal/no-preference lifecycle behavior is handled in Task 3.

- [ ] **Step 5: Run the preference suite**

Run: `OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest tests.test_preferences -v`

Expected: all tests pass.

- [ ] **Step 6: Commit ordinary evidence recording**

```bash
git add starter/preference_tool.py tests/test_preferences.py
git commit -m "feat: record preference evidence sources"
```

---

### Task 3: Implement selective correction and evidence retirement

**Files:**
- Modify: `starter/preference_tool.py`
- Modify: `tests/test_preferences.py`

- [ ] **Step 1: Write the failing same-product correction test**

Build state through real patch calls so it contains:

1. turn 1 unsolicited `feature="hand wash only"`;
2. turn 2 clarification `material="cotton"` after `previous_ask_attribute="material"`;
3. asked attributes `("material", "color")`, no-preference `{"color"}`, and stale recommendations;
4. turn 3 reset patch with category unchanged and `material="nylon"`.

Assert that the corrected state:

```python
self.assertEqual(updated.category, "accessories belts")
self.assertEqual(updated.preferences, {"material": ("nylon",)})
self.assertEqual(updated.asked_attributes, ("material", "color"))
self.assertEqual(updated.no_preference_attributes, frozenset({"color"}))
self.assertEqual(updated.latest_recommendations, ())
self.assertEqual(
    [(item.attribute, item.values, item.source_kind) for item in updated.preference_evidence],
    [("material", ("nylon",), "correction")],
)
```

This verifies replacement of the corrected attribute, retirement of the latest unsolicited evidence, preservation of same-intent question history, and recommendation invalidation.

- [ ] **Step 2: Write failing tests for conflict-free preservation and full product reset**

Add one test where clarification-confirmed `feature="machine washable"` survives a correction to `material="nylon"`. Add another where `reset_product_preferences=True, category="shirts"` clears the entire ledger and all question/no-preference context before recording new shirt evidence.

- [ ] **Step 3: Run the new correction tests and confirm they fail**

Expected: current code clears all same-product state, appends corrected values to conflicts, or leaves stale evidence.

- [ ] **Step 4: Extract explicit update classification**

Add an internal literal and classifier:

```python
UpdateKind = Literal["ordinary", "preference_correction", "product_change"]


def _classify_update(
    state: ShoppingState,
    patch: PreferencePatch,
    patch_category: str,
    values_by_attribute: dict[str, tuple[str, ...]],
) -> UpdateKind:
    if not patch.reset_product_preferences:
        return "ordinary"
    keeps_category = patch_category == "unchanged" or _looks_like_attribute_constraint(
        patch_category
    )
    if state.category and keeps_category and values_by_attribute:
        return "preference_correction"
    return "product_change"
```

The classifier is conservative: an ambiguous reset stays a complete product change.

- [ ] **Step 5: Implement immutable retirement helpers**

Add helpers that return new collections rather than mutating the old state:

```python
def _without_attributes(
    evidence: list[PreferenceEvidence], attributes: set[str]
) -> list[PreferenceEvidence]:
    return [item for item in evidence if item.attribute not in attributes]


def _retire_latest_unsolicited(
    evidence: list[PreferenceEvidence],
    excluded_attributes: set[str],
) -> tuple[list[PreferenceEvidence], PreferenceEvidence | None]:
    for index in range(len(evidence) - 1, -1, -1):
        item = evidence[index]
        if item.source_kind == "unsolicited" and item.attribute not in excluded_attributes:
            return evidence[:index] + evidence[index + 1 :], item
    return evidence, None
```

When a record is retired, remove only its values from the matching flattened preference bucket and only its terms from `search_terms`. Do not remove identical values/terms still supported by another active record.

- [ ] **Step 6: Implement the three transition branches**

For `preference_correction`:

- preserve `category`;
- remove all ledger records and flattened values for corrected attributes before applying replacements;
- retire the latest remaining unsolicited product-preference record;
- preserve unaffected clarification records, asked attributes, prior no-preference markers, and profile;
- discard no-preference markers only for corrected attributes;
- clear `latest_recommendations` but retain `previous_ask_attribute` until the normal response cycle selects the next question;
- record new values with `source_kind="correction"`.

For `product_change`, keep current complete-clear semantics and clear the evidence ledger. For `ordinary`, retain Task 2 behavior.

- [ ] **Step 7: Make removals and no-preference updates retire matching evidence**

An attribute-wide removal/no-preference update removes every evidence record for that attribute and removes its supported terms. A value-specific removal drops that value from matching records, discarding any record left with neither values nor terms. Preserve evidence for all unrelated attributes.

- [ ] **Step 8: Run preference tests and inspect the diff**

Run:

```bash
OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest tests.test_preferences -v
git diff --check
```

Expected: all preference tests pass and whitespace validation is clean.

- [ ] **Step 9: Commit selective correction semantics**

```bash
git add starter/preference_tool.py tests/test_preferences.py
git commit -m "feat: preserve confirmed evidence on correction"
```

---

### Task 4: Structure deterministic clarification answers and remove boilerplate noise

**Files:**
- Modify: `starter/preference_tool.py`
- Modify: `tests/test_preferences.py`

- [ ] **Step 1: Write failing direct-answer tests**

Cover these evaluator-style replies with an existing `previous_ask_attribute`:

```python
cases = [
    ("material", "For that, what matters is: cotton; polyester.", ("cotton", "polyester")),
    ("feature", "For that, what matters is: button closure; machine washable.", ("button closure", "machine washable")),
    ("use_case", "For that, what matters is: winter hiking.", ("winter", "hiking")),
]
```

For each case, parse then apply the patch and assert the expected structured values, `source_kind="clarification"`, and absence of boilerplate terms such as `"matters"` and `"that"` from `search_terms`.

- [ ] **Step 2: Write the failing no-preference noise test**

With `previous_ask_attribute="feature"`, parse and apply `"No additional preference; use your judgment."`. Assert `feature` is in `no_preference_attributes` and `search_terms == ()`.

- [ ] **Step 3: Run the focused parser tests and confirm they fail**

Expected: feature text remains unstructured and conversational tokens leak into search terms.

- [ ] **Step 4: Add a bounded direct-answer extractor**

Add helpers equivalent to:

```python
DIRECT_ANSWER_RE = re.compile(
    r"^(?:for that,?\s*)?what matters is\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _direct_answer_values(message: str, attribute: str) -> list[str]:
    match = DIRECT_ANSWER_RE.fullmatch(message.strip())
    if not match:
        return []
    parts = [normalize_value(part) for part in re.split(r"[;|]", match.group(1))]
    values: list[str] = []
    for part in parts:
        for value in _canonical_preference_values(attribute, part):
            _append_unique(values, value)
    return values
```

Invoke it only when `state.previous_ask_attribute` is an allowed non-category attribute and the message is not a no-preference reply. Requiring the evaluator-style `what matters is:` marker prevents an unrelated next-turn product change from being misclassified as a clarification answer. Add the resulting items as `PreferenceValue(attribute=previous_ask_attribute, value=value)`.

For `feature` and `other`, preserve concise semicolon-delimited phrases. For controlled vocabularies, reuse canonicalization. Do not attempt open-ended clause inference outside this bounded reply shape.

- [ ] **Step 5: Suppress search terms for direct and no-preference replies**

Extend no-preference recognition to bounded variants including `no additional preference`. If the parser recognized a direct clarification answer, let the structured values drive retrieval and omit generic token-derived `search_terms`. If it recognized no preference, return no search terms. Ordinary messages retain the current token fallback.

- [ ] **Step 6: Run all preference tests**

Run: `OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest tests.test_preferences -v`

Expected: all tests pass.

- [ ] **Step 7: Commit deterministic clarification parsing**

```bash
git add starter/preference_tool.py tests/test_preferences.py
git commit -m "feat: structure clarification answers"
```

---

### Task 5: Verify model/fallback equivalence through the agent boundary

**Files:**
- Modify: `tests/test_llm_agent.py`
- Modify: `tests/test_agent.py`
- Modify only if a failing test requires it: `starter/llm_agent.py`
- Modify only if a failing test requires it: `starter/agent.py`

- [ ] **Step 1: Add model-tool provenance assertions**

Extend `test_valid_tool_call_updates_runtime_state_and_usage_once` to assert an unsolicited evidence record. Add a test starting from `turn=1, previous_ask_attribute="feature"` whose tool call sets `feature="machine washable"`; assert a clarification record at turn 2. This proves the LangGraph tool path reaches the shared transition unchanged.

- [ ] **Step 2: Add an end-to-end same-product correction test**

Use `QueueInterpreter` for the initial product and correction, but use real `apply_preference_patch` transitions. Assert that after a clarification-confirmed feature and a later same-category correction:

- the confirmed feature survives;
- the corrected attribute has only its replacement value;
- stale recommendations are gone before the new search result is stored;
- returned IDs are unique, valid, and limited;
- the public response keys remain exactly `message`, `ask_attribute`, `recommendations`, and `usage`.

- [ ] **Step 3: Add fallback/model semantic equivalence test**

Create the same pre-correction state twice. Apply a model-shaped `PreferencePatch` to one and parse the evaluator-style correction through `parse_preference_fallback` for the other. Compare:

```python
self.assertEqual(model_state.category, fallback_state.category)
self.assertEqual(model_state.preferences, fallback_state.preferences)
self.assertEqual(model_state.no_preference_attributes, fallback_state.no_preference_attributes)
self.assertEqual(model_state.preference_evidence, fallback_state.preference_evidence)
```

- [ ] **Step 4: Run agent and interpreter suites**

Run:

```bash
OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest tests.test_agent tests.test_llm_agent -v
```

Expected: all tests pass. If they already pass without production edits, leave `starter/agent.py` and `starter/llm_agent.py` untouched. If context is not reaching the tool, make the smallest change at the existing shared patch call; do not add a second interpretation path.

- [ ] **Step 5: Commit integration coverage**

```bash
git add tests/test_agent.py tests/test_llm_agent.py
git add starter/agent.py starter/llm_agent.py  # only if changed in this task
git commit -m "test: cover preference evidence integration"
```

---

### Task 6: Run the complete offline acceptance gate

**Files:**
- Create temporarily outside tracked paths: `/tmp/selective-preference-offline-results.json`
- Modify only after acceptance: `docs/evaluations/iteration2-offline-results.json`
- Modify only after acceptance: `docs/evaluations/iteration2-comparison.md`

- [ ] **Step 1: Run the full test suite and static hygiene checks**

Run:

```bash
OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m unittest discover -s tests -v
git diff --check
rg -n "OPENAI_API_KEY\s*=|sk-[A-Za-z0-9_-]+|target_asin|expected_asin" starter tests docs/evaluations
```

Expected: all tests pass, `git diff --check` is silent, no secret appears, and any target-field matches are limited to evaluator/test fixtures rather than production ranking rules.

- [ ] **Step 2: Run the deterministic 200-session evaluator to a temporary file**

Run:

```bash
OPENAI_ENABLED=false /Users/justin/nickatnyte/.venv/bin/python -m evaluator.local_evaluator \
  --catalog /Users/justin/nickatnyte/data/catalog.jsonl \
  --public-set /Users/justin/nickatnyte/data/public_set.jsonl \
  --output /tmp/selective-preference-offline-results.json
```

Expected: 200 samples complete without API usage.

- [ ] **Step 3: Compare every acceptance metric against the fixed baseline**

The candidate must satisfy all of:

| Metric | Gate |
|---|---:|
| Overall technical score | `> 0.719486` |
| Intent-override Hit@10 | `> 0.666667` |
| Intent-override MRR | `>= 0.398611` |
| Boundary Hit@10 | `>= 0.90` |
| Browsing Hit@10 | no more than 1 case below `0.9125` |
| Buying Hit@10 | no more than 1 case below `0.90` |

Also inspect per-session intent misses to confirm the gain comes from preserved evidence rather than broad candidate inflation.

- [ ] **Step 4: Stop and revert only this feature if any gate fails**

Do not run the paid evaluator. Preserve diagnostic output under `/tmp`, report the exact failed gate, and either refine the failing transition under a new red test or revert only the feature commits from Tasks 1-5. Never reset or discard unrelated worktree changes.

- [ ] **Step 5: If every gate passes, update the tracked offline artifact and comparison**

Copy the accepted evaluator JSON into `docs/evaluations/iteration2-offline-results.json` using `apply_patch` or the evaluator's explicit output option on a fresh rerun. Update `docs/evaluations/iteration2-comparison.md` with the command, overall metrics, all four scenario metrics, baseline deltas, and a note that no API calls were made.

- [ ] **Step 6: Commit accepted offline evidence**

```bash
git add docs/evaluations/iteration2-offline-results.json docs/evaluations/iteration2-comparison.md
git commit -m "docs: record selective evidence evaluation"
```

---

### Task 7: Review, paid-evaluation decision, and branch completion

**Files:**
- Modify only after an approved successful run: `docs/evaluations/iteration2-openai-results.json`
- Modify only after an approved successful run: `docs/evaluations/iteration2-comparison.md`

- [ ] **Step 1: Perform an independent code review**

Use `superpowers:requesting-code-review` after Tasks 1-6. Review specifically for immutable-state violations, unsupported correction classification, stale flat values after evidence retirement, model/fallback divergence, public API drift, target leakage, and regression coverage.

- [ ] **Step 2: Address review findings with tests first**

For each valid finding, add or tighten a failing test, implement the smallest fix, rerun the affected suite, then rerun the complete test suite. Commit review fixes separately.

- [ ] **Step 3: Re-run the complete verification gate**

Use `superpowers:verification-before-completion` and rerun the full unit suite, deterministic evaluator, `git diff --check`, secret/target scan, and `git status --short`. Compare the new deterministic output to the accepted Task 6 artifact.

- [ ] **Step 4: Calculate paid-run exposure and request explicit approval**

Use the existing two-call smoke procedure and current official model pricing to estimate the maximum full-run spend for up to 2,000 model calls. Present the estimate and do not continue until the user explicitly approves that spend. Prior approval to implement this plan is not approval for a new paid run.

- [ ] **Step 5: Run exactly one full OpenAI evaluation if approved**

Run the existing documented evaluator command with `OPENAI_ENABLED=true` and write first to `/tmp/selective-preference-openai-results.json`. Compare overall and mode-for-mode metrics with the verified OpenAI baseline:

- overall score `0.728769`;
- Hit@10 `0.89`;
- MRR `0.469897`;
- MTTC `3.86`;
- intent override Hit@10 `0.633333`, MRR `0.28832`, MTTC `6.866667`.

Retain and publish the paid result only if it is a genuine mode-for-mode improvement without material boundary, browsing, or buying regression. Otherwise preserve the diagnostic output in `/tmp` and keep the last accepted tracked artifact.

- [ ] **Step 6: Update final artifacts and commit**

If accepted, update `docs/evaluations/iteration2-openai-results.json` and the final comparison report, then commit only those artifacts:

```bash
git add docs/evaluations/iteration2-openai-results.json docs/evaluations/iteration2-comparison.md
git commit -m "docs: record final task 7 evaluation"
```

- [ ] **Step 7: Finish the development branch**

Use `superpowers:finishing-a-development-branch` only after all required checks pass. Report the final offline and OpenAI metrics, commit list, remaining pre-existing worktree changes, and the available integration choices. Do not merge, push, or delete the worktree without explicit user direction.
