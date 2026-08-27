# LangChain Shopping Agent — Iteration 1 Design

**Date:** 2026-08-26
**Status:** Approved for implementation planning

## 1. Objective

Replace the provided stateless BM25 agent with a controlled LangChain agent that uses an OpenAI model to interpret each shopper message and call an `update_user_preferences` tool. The tool stores structured preferences in thread-scoped runtime state for the rest of the competition session. Product retrieval remains deterministic, local, and lexical.

Iteration 1 succeeds when it:

- preserves the organizer's `Agent.reset()` and `Agent.respond()` interface;
- stores and applies preferences across turns;
- handles explicit preference replacement and category/intent overrides;
- always returns a contract-valid response with up to ten catalog IDs;
- never reads ground-truth labels from participant agent code;
- passes the organizer tests and new focused tests;
- retains a deterministic fallback when OpenAI is unavailable; and
- can be evaluated against the verified public baseline score of `0.10671`.

## 2. Current Baseline

The supplied agent loads the 50,000-product catalog into an in-memory SQLite FTS5 index. On each turn it tokenizes only the latest message, creates an OR query, ranks matches with weighted BM25, and returns the first ten product IDs.

It currently does not:

- use the supplied user profile;
- remember earlier messages;
- identify structured preferences or intent;
- detect overrides;
- ask clarification questions; or
- use an LLM.

The verified baseline metrics are:

- Hit Rate@10: `0.125`
- MRR: `0.068034`
- MTTC: `9.81`
- Technical score: `0.10671`

## 3. Scope

### In scope

- LangChain integration with OpenAI via `langchain-openai`.
- One model-assisted preference interpretation step per evaluator turn.
- A mandatory `update_user_preferences` tool with validated arguments.
- Thread-scoped in-memory state keyed by evaluator `session_id`.
- Lexical multi-route retrieval and deterministic reranking.
- Intent mode, category, preference, override, and no-preference tracking.
- Deterministic clarification-question selection.
- Token-usage reporting, timeouts, bounded retries, and fallback behavior.
- Unit, integration, and evaluator verification.

### Out of scope

- Dense/vector retrieval or embeddings.
- LLM-generated or hallucinated product IDs.
- A web or mobile interface.
- Long-term memory across evaluator sessions.
- Training or fine-tuning a model.
- External vector databases.
- Running the full 200-session API evaluation without a prior cost estimate and user approval.

## 4. Architectural Decision

Iteration 1 uses a controlled hybrid architecture rather than a fully autonomous agent loop.

The OpenAI model has one responsibility: interpret the latest shopper message in light of compact current state and produce a validated call to `update_user_preferences`. The tool applies the state mutation. Ordinary Python code then performs catalog retrieval, reranking, clarification selection, and competition-response construction.

This design limits the workflow to one OpenAI request per evaluator turn, prevents the model from inventing catalog identifiers, and guarantees that every turn can fall back to deterministic behavior.

## 5. Components

### 5.1 Competition adapter — `starter/agent.py`

The public `Agent` class remains the evaluator entry point.

- `__init__` builds the local catalog index and constructs the model client when configured.
- `reset(session_id, user_profile)` creates clean runtime state for the session.
- `respond(session_id, user_message, turn, top_k)` coordinates preference interpretation, retrieval, clarification, usage accounting, and response validation.
- Calling `respond` before `reset` remains an error.

### 5.2 LLM preference interpreter — `starter/llm_agent.py`

This component configures `ChatOpenAI` through LangChain and binds the preference-update tool. It uses a one-step controlled workflow rather than LangChain's default open-ended tool loop: the bound model is required to emit one `update_user_preferences` call, Python executes that call, and the workflow ends without invoking the model a second time. Retrieval and final response construction then run outside the model.

The default model is `gpt-5.6-luna`, configurable through `OPENAI_MODEL`. The integration explicitly enables the OpenAI Responses API and binds `update_user_preferences` as a required tool with a strict argument schema. Model timeout and retry count are bounded through configuration.

The model receives only:

- the latest shopper message;
- the current compact intent and preference state;
- the supplied anonymized profile summary and tags; and
- the previous clarification attribute, when present.

The full raw conversation is not the source of truth and is not required in the model prompt.

### 5.3 Runtime state — `starter/state.py`

`ShoppingState` is a typed custom LangChain/LangGraph agent state. The controlled one-step workflow is compiled with an in-memory checkpointer and invoked with the evaluator `session_id` as its thread ID. The preference tool receives LangChain `ToolRuntime` and returns a state-update command; it does not mutate an unvalidated global object.

Each `session_id` therefore maps to isolated, thread-scoped state containing:

- `intent_mode`: `buying`, `browsing`, or `unknown`;
- `category`: the current product category or null;
- `preferences`: active normalized values by allowed attribute;
- `removed_preferences`: values explicitly rejected by the shopper;
- `no_preference_attributes`: attributes the shopper does not care about;
- `search_terms`: relevant free terms not represented by a structured attribute;
- `asked_attributes`: clarification attributes already requested;
- `previous_ask_attribute`: the most recent clarification;
- `latest_recommendations`: the most recent catalog IDs;
- `user_profile`: the safe profile supplied to `reset`;
- `turn`: the latest evaluator turn; and
- accumulated prompt and completion token counts.

State lasts only for the evaluator process. A subsequent `reset` for the same ID explicitly replaces the prior checkpointed state before another turn is accepted.

### 5.4 Preference tool — `starter/preference_tool.py`

`update_user_preferences` is the only model-facing mutation path. Its schema requires explicit fields so malformed or partial updates can be rejected safely.

The tool arguments contain:

- an intent-mode value or `unchanged`;
- a category value or `unchanged`;
- a list of attribute/value pairs to set;
- a list of attribute/value removals, where an omitted value removes the entire attribute;
- a list of no-preference attributes;
- a boolean indicating whether product-specific preferences must be reset; and
- normalized free search terms.

Mutation order is deterministic:

1. Reset product-specific state when requested.
2. Apply removals.
3. Record no-preference attributes and remove active values for them.
4. Apply intent and category updates.
5. Apply new preference values.
6. Merge normalized free search terms while excluding removed values.

Unknown attributes and blank values are ignored. Allowed preference attributes match the competition contract: `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, and `other`.

### 5.5 Lexical retrieval — `starter/retrieval.py`

The existing in-memory SQLite FTS5 approach is retained and extended with multiple routes:

1. A precision route emphasizes active terms in titles and categories.
2. A broad route searches titles, categories, features, details, stores, and descriptions.
3. A latest-message route ensures a new intent is not overwhelmed by older preferences.

Each route retrieves an internal candidate set. Candidate lists are merged using deterministic rank fusion and reranked with boosts for:

- current category matches;
- active material, color, size, style, brand, budget, feature, and use-case matches;
- exact title or category phrase matches;
- agreement across retrieval routes; and
- rating and rating count as small tie-breakers.

Preferences are normally soft boosts instead of absolute filters because incomplete catalog metadata could otherwise remove the correct target. Removed values may receive bounded penalties when they appear explicitly.

The retriever returns valid, unique `parent_asin` values from the frozen catalog only.

### 5.6 Clarification policy — `starter/questions.py`

Clarification selection remains deterministic. The policy:

- never repeats an asked or no-preference attribute;
- prioritizes missing attributes likely to narrow the current candidate pool;
- uses only values permitted by `ask_attribute`;
- emits a natural question from a fixed template; and
- returns no clarification on turn 10.

The agent returns its current best recommendations even when asking a question.

## 6. Turn Flow

For each call to `respond`:

1. Validate that the session was reset.
2. Load current state.
3. Invoke the OpenAI-backed LangChain preference interpreter once.
4. Execute and validate the preference-tool update.
5. If the model call or tool update fails, run the deterministic preference parser instead.
6. Build lexical queries from updated state and the latest message.
7. Retrieve, merge, and rerank candidates.
8. Select the next clarification attribute and message.
9. Validate and deduplicate recommendations.
10. Store the latest turn, clarification, recommendations, and token usage.
11. Return the exact competition response schema.

## 7. Failure Handling

- Missing `OPENAI_API_KEY`: log no secret and use deterministic parsing.
- Model timeout, rate limit, or network failure: retry only within the configured bound, then fall back.
- Invalid or missing tool call: ignore unsafe mutation and use deterministic parsing.
- Invalid preference patch: apply valid normalized fields only; never corrupt existing state.
- Empty lexical query: use category/profile terms when available, otherwise a deterministic catalog fallback.
- Search failure: return a contract-valid empty recommendation list and a useful clarification rather than raising through the evaluator.
- Invalid/duplicate product IDs: remove them before returning the response.

## 8. Configuration and Security

The repository will include an `.env.example` containing variable names but no values. The real `.env` remains ignored by Git.

Configuration includes:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default `gpt-5.6-luna`)
- `OPENAI_TIMEOUT_SECONDS` (bounded request timeout)
- `OPENAI_MAX_RETRIES` (bounded retry count)
- `OPENAI_ENABLED` (set false to force deterministic fallback)

No API key, catalog file, generated result, prompt transcript, or private evaluator data may be committed. Usage metadata returned by the model is accumulated into the competition `usage` fields.

## 9. Testing Strategy

### Automated offline tests

- Existing organizer evaluator tests remain unchanged and passing.
- State isolation across session IDs.
- Preference additions, replacements, removals, and no-preference updates.
- Category override with product-specific state reset.
- Tool-schema validation and normalized updates.
- Deterministic fallback after simulated API failure.
- Lexical route merging, boost behavior, uniqueness, and valid IDs using a small fixture catalog.
- Clarification ordering and non-repetition.
- Required response schema and token-usage accounting.
- No participant-agent import or access to `ground_truth`.

LLM behavior is represented by a fake/stub model in automated tests, so the suite does not spend money or depend on network access.

### Online smoke verification

After the user configures the API key, run a small bounded set of representative messages to verify:

- correct LangChain/OpenAI connectivity;
- actual preference tool calls;
- persistence across turns;
- intent override behavior; and
- token usage reporting.

### Evaluator verification

Run the organizer tests and the full evaluator in deterministic-fallback mode first. Before a full OpenAI-backed run, estimate the maximum number of calls and expected token cost and obtain explicit user approval. Compare overall and scenario-level metrics against the verified baseline.

## 10. Planned Repository Changes

- Replace the implementation in `starter/agent.py` while preserving its public interface.
- Add `starter/llm_agent.py`.
- Add `starter/state.py`.
- Add `starter/preference_tool.py`.
- Add `starter/retrieval.py`.
- Add `starter/questions.py`.
- Add focused agent and component tests under `tests/`.
- Add `requirements.txt` and `.env.example`.
- Update the README with setup, configuration, evaluation, cost, and fallback instructions.

The existing evaluator, public labels, catalog structure, and organizer contract files remain unmodified.

## 11. Acceptance Criteria

The implementation is ready for iteration-1 review when:

1. All existing and new offline tests pass.
2. The unchanged evaluator runs successfully in deterministic-fallback mode.
3. A bounded real-API smoke test demonstrates preference persistence and override handling.
4. Every response conforms to the official API contract.
5. The agent never returns catalog IDs invented by the model.
6. Failure of the OpenAI call does not terminate an evaluator session.
7. No secret, catalog asset, generated result, or public-target special case appears in the Git diff.
8. The full OpenAI-backed public evaluation is run only after cost review and explicit approval.
