# LangChain Shopping Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a competition-compatible shopping agent that uses one OpenAI/LangChain preference-tool call per turn, remembers validated preferences for the session, and performs deterministic lexical catalog retrieval.

**Architecture:** `Agent` coordinates a LangGraph-backed `PreferenceInterpreter`, a deterministic fallback parser, an SQLite FTS5 `CatalogRetriever`, and a fixed clarification policy. The interpreter executes one required `update_user_preferences` tool call against thread-scoped runtime state; ordinary Python searches the frozen catalog and constructs catalog-safe responses.

**Tech Stack:** Python 3.10+, standard-library SQLite FTS5, `langchain-openai`, `langchain-core`, Pydantic v2, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-26-langchain-shopping-agent-design.md`

## Global Constraints

- Preserve `Agent.reset(session_id, user_profile)` and `Agent.respond(session_id, user_message, turn, top_k)`.
- Use at most one OpenAI request per evaluator turn and never ask the model to generate product IDs.
- Use `gpt-5.6-luna` by default and allow `OPENAI_MODEL` to override it.
- Fall back deterministically when OpenAI is disabled, unconfigured, unavailable, or returns an invalid tool call.
- Keep state isolated by `session_id` and replace it when that ID is reset.
- Return only unique `parent_asin` values loaded from the frozen catalog.
- Do not access `ground_truth`, modify evaluator code, commit secrets, or run the full API-backed evaluator without a cost estimate and explicit approval.

---

### Task 1: Runtime State and Validated Preference Updates

**Files:**
- Create: `requirements.txt`
- Create: `starter/state.py`
- Create: `starter/preference_tool.py`
- Create: `tests/test_preferences.py`

**Interfaces:**
- Produces: `ShoppingState.new(session_id: str, user_profile: dict) -> ShoppingState`.
- Produces: `PreferencePatch`, `PreferenceValue`, `PreferenceRemoval`, and `apply_preference_patch(state: ShoppingState, patch: PreferencePatch) -> ShoppingState`.
- Produces: `parse_preference_fallback(message: str, state: ShoppingState) -> PreferencePatch`.

- [ ] **Step 1: Write tests that catch lost memory, cross-session leakage, ignored removals, and failed category overrides**

```python
class PreferenceUpdateTest(unittest.TestCase):
    def test_patch_adds_normalized_values_without_mutating_old_state(self):
        state = ShoppingState.new("s1", {"summary": "likes comfort", "preference_tags": ["fit"]})
        updated = apply_preference_patch(state, PreferencePatch(
            intent_mode="buying",
            category="Shoes",
            set_preferences=[PreferenceValue(attribute="color", value=" Blue ")],
            search_terms=["Running"],
        ))
        self.assertEqual(updated.preferences, {"color": ["blue"]})
        self.assertEqual(state.preferences, {})

    def test_override_resets_product_specific_preferences(self):
        state = replace(ShoppingState.new("s1", {}), category="shirts", preferences={"color": ["red"]})
        updated = apply_preference_patch(state, PreferencePatch(
            category="boots", reset_product_preferences=True,
            set_preferences=[PreferenceValue(attribute="material", value="leather")],
        ))
        self.assertEqual(updated.category, "boots")
        self.assertEqual(updated.preferences, {"material": ["leather"]})

    def test_no_preference_clears_active_attribute(self):
        state = replace(ShoppingState.new("s1", {}), preferences={"color": ["red"]})
        updated = apply_preference_patch(state, PreferencePatch(no_preference_attributes=["color"]))
        self.assertNotIn("color", updated.preferences)
        self.assertEqual(updated.no_preference_attributes, frozenset({"color"}))
```

- [ ] **Step 2: Declare and install dependencies, then run the preference tests and confirm they fail because the new modules do not exist**

Create `requirements.txt` with `langchain-core>=1.0,<2.0`, `langchain-openai>=1.0,<2.0`, `langgraph>=1.0,<2.0`, and `pydantic>=2.7,<3.0`.

Run: `python3 -m pip install -r requirements.txt && python3 -m unittest tests.test_preferences -v`

Expected: import failure for `starter.preference_tool` or `starter.state`.

- [ ] **Step 3: Implement immutable session state, strict Pydantic patch types, normalization, ordered patch application, and the fallback parser**

```python
@dataclass(frozen=True)
class ShoppingState:
    session_id: str
    user_profile: dict[str, Any]
    intent_mode: Literal["buying", "browsing", "unknown"] = "unknown"
    category: str | None = None
    preferences: dict[str, list[str]] = field(default_factory=dict)
    removed_preferences: dict[str, list[str]] = field(default_factory=dict)
    no_preference_attributes: frozenset[str] = frozenset()
    search_terms: tuple[str, ...] = ()
    asked_attributes: tuple[str, ...] = ()
    previous_ask_attribute: str | None = None
    latest_recommendations: tuple[str, ...] = ()
    turn: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def new(cls, session_id: str, user_profile: dict) -> "ShoppingState":
        return cls(session_id=session_id, user_profile=deepcopy(user_profile))
```

`apply_preference_patch` must return a new state, apply reset before removals and additions, ignore unknown attributes and blank values, remove active values marked unwanted, and prevent rejected terms from being re-added as free search terms. The fallback parser must recognize explicit intent language, category phrases following `looking for`, common colors/materials, budget expressions, and phrases such as `don't have a preference for color` and `ignore my earlier preference`.

- [ ] **Step 4: Run the focused tests until they pass**

Run: `python3 -m unittest tests.test_preferences -v`

Expected: all preference tests pass without network access.

- [ ] **Step 5: Commit the state and preference update slice**

```bash
git add requirements.txt starter/state.py starter/preference_tool.py tests/test_preferences.py
git commit -m "feat: add session preference state"
```

### Task 2: Multi-Route Lexical Retrieval

**Files:**
- Create: `starter/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `ShoppingState` from Task 1.
- Produces: `CatalogRetriever(catalog_path: str | Path)`.
- Produces: `CatalogRetriever.search(state: ShoppingState, latest_message: str, top_k: int) -> list[str]`.

- [ ] **Step 1: Write fixture-backed tests that catch invalid IDs, duplicates, ignored state preferences, and stale-intent domination**

```python
class RetrievalTest(unittest.TestCase):
    def test_active_preferences_boost_matching_product(self):
        state = replace(ShoppingState.new("s", {}), category="boots", preferences={"material": ["leather"]})
        self.assertEqual(self.retriever.search(state, "winter footwear", 2)[0], "LEATHER_BOOT")

    def test_latest_message_route_can_surface_new_category(self):
        state = replace(ShoppingState.new("s", {}), category="shirts", search_terms=("cotton",))
        results = self.retriever.search(state, "actually I need a hiking backpack", 3)
        self.assertIn("HIKING_PACK", results)

    def test_results_are_unique_catalog_ids_and_respect_top_k(self):
        results = self.retriever.search(ShoppingState.new("s", {}), "blue shoes", 2)
        self.assertEqual(len(results), len(set(results)))
        self.assertLessEqual(len(results), 2)
        self.assertTrue(set(results) <= self.catalog_ids)
```

- [ ] **Step 2: Run the retrieval tests and confirm the missing retriever causes failure**

Run: `python3 -m unittest tests.test_retrieval -v`

Expected: import failure for `starter.retrieval`.

- [ ] **Step 3: Implement catalog loading, weighted FTS5 routes, rank fusion, preference boosts, removal penalties, and deterministic fallback ranking**

The FTS table stores `parent_asin`, title, categories, features, details, store, and description. Keep numeric `price`, `average_rating`, and `rating_number` in a metadata map. Retrieve up to `max(50, top_k * 10)` candidates per non-empty route, award reciprocal-rank fusion points, then add bounded exact-text/category/preference boosts and small rating tie-breakers. Sort by `(-score, parent_asin)` and return unique identifiers only.

- [ ] **Step 4: Run focused and existing tests**

Run: `python3 -m unittest tests.test_retrieval tests.test_evaluator -v`

Expected: all tests pass.

- [ ] **Step 5: Commit deterministic retrieval**

```bash
git add starter/retrieval.py tests/test_retrieval.py
git commit -m "feat: add state-aware lexical retrieval"
```

### Task 3: Deterministic Clarification Policy

**Files:**
- Create: `starter/questions.py`
- Create: `tests/test_questions.py`

**Interfaces:**
- Consumes: `ShoppingState` from Task 1.
- Produces: `choose_clarification(state: ShoppingState, turn: int) -> tuple[str, str | None]`.

- [ ] **Step 1: Write tests for valid ordering, non-repetition, no-preference exclusions, and turn-ten behavior**

```python
class ClarificationTest(unittest.TestCase):
    def test_question_is_valid_and_not_repeated(self):
        state = replace(ShoppingState.new("s", {}), category="shoes", asked_attributes=("material",))
        message, attribute = choose_clarification(state, turn=2)
        self.assertEqual(attribute, "color")
        self.assertIn("color", message.lower())

    def test_no_preference_attribute_is_skipped(self):
        state = replace(ShoppingState.new("s", {}), category="shoes", no_preference_attributes=frozenset({"material"}))
        self.assertNotEqual(choose_clarification(state, turn=1)[1], "material")

    def test_turn_ten_returns_no_question(self):
        self.assertEqual(choose_clarification(ShoppingState.new("s", {}), turn=10), ("Here are the closest matches I found.", None))
```

- [ ] **Step 2: Run the clarification tests and confirm failure because the policy does not exist**

Run: `python3 -m unittest tests.test_questions -v`

Expected: import failure for `starter.questions`.

- [ ] **Step 3: Implement fixed templates and a valid attribute priority list**

Use category first when missing, then material, color, size, style, brand, budget, feature, use_case, and other. Exclude active, asked, and no-preference attributes. Return the fixed recommendation message with `None` once the list is exhausted or `turn >= 10`.

- [ ] **Step 4: Run clarification tests**

Run: `python3 -m unittest tests.test_questions -v`

Expected: all clarification tests pass.

- [ ] **Step 5: Commit the clarification policy**

```bash
git add starter/questions.py tests/test_questions.py
git commit -m "feat: add clarification policy"
```

### Task 4: One-Call LangChain Preference Interpreter

**Files:**
- Create: `starter/llm_agent.py`
- Create: `tests/test_llm_agent.py`

**Interfaces:**
- Consumes: `ShoppingState` and `PreferencePatch` from Task 1.
- Produces: `Interpretation(state: ShoppingState, prompt_tokens: int, completion_tokens: int)`.
- Produces: `PreferenceInterpreter.from_environment() -> PreferenceInterpreter | None`.
- Produces: `PreferenceInterpreter.interpret(message: str, state: ShoppingState) -> Interpretation`.

- [ ] **Step 1: Write tests using a complete fake LangChain response that catch extra model calls, missing tool calls, malformed arguments, and lost token usage**

```python
class PreferenceInterpreterTest(unittest.TestCase):
    def test_valid_tool_call_returns_patch_and_usage(self):
        response = AIMessage(
            content="",
            tool_calls=[{"name": "update_user_preferences", "args": {
                "intent_mode": "buying", "category": "shoes",
                "set_preferences": [{"attribute": "color", "value": "blue"}],
                "remove_preferences": [], "no_preference_attributes": [],
                "reset_product_preferences": False, "search_terms": ["running"]
            }, "id": "call_1", "type": "tool_call"}],
            usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )
        model = FakeBoundModel(response)
        result = PreferenceInterpreter(model).interpret("blue running shoes", ShoppingState.new("s", {}))
        self.assertEqual(result.state.preferences, {"color": ["blue"]})
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (80, 20))
        self.assertEqual(model.calls, 1)

    def test_missing_tool_call_raises_invalid_interpretation(self):
        model = FakeBoundModel(AIMessage(content="plain text"))
        with self.assertRaises(InvalidInterpretation):
            PreferenceInterpreter(model).interpret("shoes", ShoppingState.new("s", {}))
```

- [ ] **Step 2: Run the tests and confirm the interpreter is missing**

Run: `python3 -m unittest tests.test_llm_agent -v`

Expected: import failure for `starter.llm_agent`.

- [ ] **Step 3: Implement the strict tool schema, compact prompt, environment configuration, one bound-model invocation, tool-call validation, and usage extraction**

Configure `ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"), timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20")), max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "1")), use_responses_api=True)`, then call `bind_tools([update_user_preferences], tool_choice="update_user_preferences", strict=True)`. Compile a `StateGraph` with an `InMemorySaver`, one model node, one `ToolNode`, and `START -> model -> tools -> END`. The tool receives `ToolRuntime`, applies `PreferencePatch.model_validate` to its arguments, and returns a `Command` containing updated `ShoppingState` plus its matching tool message. The prompt includes the latest message, safe profile summary/tags, current normalized state, and previous asked attribute. Invoke the graph once with `session_id` as `configurable.thread_id`; the graph performs one model request and never returns to the model node.

- [ ] **Step 4: Run the interpreter and preference tests**

Run: `python3 -m unittest tests.test_llm_agent tests.test_preferences -v`

Expected: all tests pass without an API key or network request.

- [ ] **Step 5: Commit the LangChain boundary**

```bash
git add starter/llm_agent.py tests/test_llm_agent.py
git commit -m "feat: add LangChain preference interpreter"
```

### Task 5: Competition Agent Integration and Fallback

**Files:**
- Modify: `starter/agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: all Task 1–4 interfaces.
- Preserves: organizer `Agent` constructor, `reset`, and `respond` methods.
- Adds optional test seam: `Agent(catalog_path, interpreter=None, openai_enabled=None)`.

- [ ] **Step 1: Write integration tests that catch respond-before-reset, memory loss, session leakage, API-failure propagation, invalid recommendations, repeated questions, and usage-accounting errors**

```python
class AgentIntegrationTest(unittest.TestCase):
    def test_preferences_persist_across_turns(self):
        agent = Agent(self.catalog_path, interpreter=QueueInterpreter([
            Interpretation(replace(ShoppingState.new("s", {}), category="shoes", preferences={"color": ["blue"]}), 12, 4),
            Interpretation(replace(ShoppingState.new("s", {}), category="shoes", preferences={"color": ["blue"]}), 10, 3),
        ]))
        agent.reset("s", {"summary": "", "preference_tags": []})
        agent.respond("s", "blue shoes", 1, 10)
        agent.respond("s", "show me more", 2, 10)
        self.assertEqual(agent.session_state("s").preferences, {"color": ["blue"]})

    def test_api_failure_uses_fallback_and_returns_contract_shape(self):
        agent = Agent(self.catalog_path, interpreter=FailingInterpreter())
        agent.reset("s", {"summary": "", "preference_tags": []})
        response = agent.respond("s", "black leather boots", 1, 10)
        self.assertTrue(response["recommendations"])
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES | {None})
```

- [ ] **Step 2: Run the agent integration tests and confirm the old baseline fails the new behaviors**

Run: `python3 -m unittest tests.test_agent -v`

Expected: failures for constructor injection, missing state memory, and missing fallback behavior.

- [ ] **Step 3: Replace the baseline coordinator while preserving the public API**

`reset` creates a fresh `ShoppingState`. `respond` validates the session, calls the interpreter once when present, falls back on any interpretation exception, applies the patch, searches the catalog, chooses a question, accumulates token usage in state, stores latest recommendations/asked attribute/turn, and returns exactly the contract keys. Cap results to `max(0, top_k)` and deduplicate before wrapping IDs as dictionaries.

- [ ] **Step 4: Run all offline tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: organizer and new tests all pass without an API key.

- [ ] **Step 5: Commit the integrated agent**

```bash
git add starter/agent.py tests/test_agent.py
git commit -m "feat: integrate conversational shopping agent"
```

### Task 6: Configuration, Documentation, and Offline Evaluation

**Files:**
- Create: `.env.example`
- Modify: `README.md`
- Modify: `.gitignore`
- Generated but untracked: `results.json`

**Interfaces:**
- Documents: installation, `OPENAI_*` variables, API-key safety, fallback mode, smoke testing, evaluator commands, model disclosure, and full-evaluation cost gate.

- [ ] **Step 1: Create `.env.example`, dependency/setup instructions, fallback instructions, and the paid-evaluation cost warning**

`.env.example` contains empty values for `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-5.6-luna`, `OPENAI_TIMEOUT_SECONDS=20`, `OPENAI_MAX_RETRIES=1`, and `OPENAI_ENABLED=true`. The README explains that the key stays only in `.env` or the shell environment and that `OPENAI_ENABLED=false` forces free offline mode.

- [ ] **Step 2: Verify local secrets, generated results, and the catalog remain ignored**

Run: `git check-ignore .env results.json data/catalog.jsonl`

Expected: the command prints all three paths and exits successfully.

- [ ] **Step 3: Run the complete suite and the evaluator with OpenAI explicitly disabled**

Run: `OPENAI_ENABLED=false python3 -m unittest discover -s tests -v`

Run: `OPENAI_ENABLED=false python3 -m evaluator.local_evaluator`

Expected: tests pass; evaluator completes all 200 sessions and writes valid aggregate metrics to ignored `results.json` without API calls.

- [ ] **Step 4: Inspect the final diff for secrets, catalog assets, generated output, public-target special cases, and unrelated files**

Run: `git status --short && git diff --check && git diff -- . ':!starter/agent_baseline.py'`

Expected: no `.env`, `data/catalog.jsonl`, `results.json`, or `starter/agent_baseline.py` is staged; no whitespace errors or target-specific rules appear.

- [ ] **Step 5: Commit the verified setup and documentation**

```bash
git add .env.example .gitignore README.md
git commit -m "docs: add agent setup and safety guidance"
```

- [ ] **Step 6: Stop before paid evaluation**

Report the offline evaluator metrics and the bounded smoke-test command. Do not run a real-API smoke test until the user has put the key in their own environment, and do not run the full API-backed evaluator until call count and token cost are estimated and explicitly approved.
