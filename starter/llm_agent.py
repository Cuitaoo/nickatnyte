from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from starter.preference_tool import (
    PreferencePatch,
    PreferenceRemoval,
    PreferenceValue,
    apply_preference_patch,
    references_unnamed_earlier_preference,
)
from starter.query_expansion import (
    ScenarioHypothesis,
    looks_like_scenario_query,
    query_expansion_enabled,
    validate_scenario_hypotheses,
)
from starter.state import ShoppingState


SYSTEM_PROMPT = """You are the state updater for a conversational catalog search system.
Interpret only the latest shopper message in the context of current_state. Call
update_user_preferences exactly once and never answer in prose.

Choose exactly one update_type:
- merge: continue the same product search. Add newly stated constraints, explicit
  removals, or a no-preference answer while preserving unrelated state.
- replace_preferences: continue the same product search, but replace the active
  values for every attribute present in set_preferences. Preserve the category
  and all unrelated confirmed attributes.
- product_change: the shopper clearly switches to a different product type or
  category. Product-specific state will be reset; the aggregate user profile is
  retained outside this tool.

For replace_preferences, choose one correction_scope:
- corrected_attributes: retire only the old values of attributes explicitly
  corrected by the shopper. This is the default.
- latest_unsolicited: the shopper says to ignore/replace an earlier or previous
  preference without naming it. Retire exactly the latest active unsolicited
  preference evidence in addition to replacing the stated attributes. Never use
  this scope for a named old value or a complete product change.

Critical transition rules:
- "Actually, make it polyester" is replace_preferences, not product_change.
- "Ignore leather; make it suede" is replace_preferences for material.
- "Ignore my earlier preference. What I need is: Water Resistant" is
  replace_preferences with correction_scope=latest_unsolicited.
- "I no longer want leather" is merge plus remove_preferences.
- "I have no color preference" is merge plus no_preference_attributes=["color"].
- "Instead, I need waterproof hiking boots" while shopping for shirts is
  product_change with the new category and constraints.
- Words such as "actually", "instead", "ignore", or "changed my mind" do not by
  themselves prove a product change. Use the product noun/category as evidence.
- A direct answer to previous_ask_attribute updates that attribute unless the
  shopper explicitly names another one.

State extraction rules:
- Use "unchanged" for intent_mode and category when the message does not change
  them. category is a concise product noun phrase, including an explicit audience
  or department when stated (for example "men's jeans"), never a material or
  feature by itself.
- Infer buying or browsing only from the shopper's language. Do not turn a normal
  clarification answer into a new intent.
- Use only these attributes: material, color, size, style, brand, budget, feature,
  use_case, and other. Put explicit rejected values in remove_preferences.
- search_terms contains only distinctive catalog evidence from the latest message
  that is not already represented cleanly by category or a structured preference.
  Preserve exact model/part identifiers as one term. Every category, preference
  value, removal value, and search term must be copied from an explicit span in
  the latest shopper message.
- Do not translate, paraphrase, expand synonyms, or infer implied needs. For
  example, keep "trainers" as "trainers" rather than rewriting it to "sneakers".
  Never invent an unstated brand, material, audience, budget, or feature.
- Infer confirmed constraints only from shopper messages. The aggregate profile is
  deliberately not provided because it must not become a hard active constraint.
- Never create, request, or return product IDs."""

TEMPORARY_RETRIEVAL_PROMPT = """
Temporary retrieval hypothesis rules:
- scenario_hypotheses is separate from confirmed state. Use it only when the
  latest message expresses a broad scenario or goal whose ordinary product
  implications could improve semantic catalog recall.
- scenario_query contains only concise inferred functional catalog language,
  such as "portable long battery life". Omit literal product nouns, audience,
  brand, location, identifiers, stated attributes, and numeric limits: the
  deterministic category and feature routes already handle those facts.
- Do not infer weather, season, or insulation from a country or city alone. If
  the scenario is too ambiguous to support a functional hypothesis, return [].
- Inferred qualities are temporary recall ideas, never set_preferences,
  search_terms, removals, or category values.
- When the scenario is explicit, return one focused hypothesis. Examples:
  - "shoes for wet weather" -> scenario_query="waterproof traction" with
    basis="wet weather".
  - "laptop backpack for commuting" -> scenario_query="padded laptop sleeve
    comfortable carrying" with basis="commuting".
- basis must be copied verbatim from the latest shopper message. confidence is
  between 0 and 1 and reflects how strongly that text supports the expansion.
- Return at most three concise hypotheses. Return [] for precise catalog,
  identifier, or attribute-led queries that need no expansion.
"""


PreferenceAttribute = Literal[
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
]
MODEL_PREFERENCE_ATTRIBUTES = frozenset(
    {
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)
MODEL_REMOVABLE_ATTRIBUTES = MODEL_PREFERENCE_ATTRIBUTES | {"category"}
RemovableAttribute = Literal[
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
]
ShortText = Annotated[str, Field(min_length=1, max_length=200)]
EXPLICIT_TOKEN_RE = re.compile(r"[a-z0-9$]+", re.IGNORECASE)
NO_PREFERENCE_RE = re.compile(
    r"(?:\bno\b|\bdon['’]?t\b|\bdo\s+not\b).{0,80}\bpreference\b",
    re.IGNORECASE,
)


class ModelPreferenceValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: PreferenceAttribute
    value: ShortText


class ModelPreferenceRemoval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: RemovableAttribute
    value: ShortText | None = None


class ModelScenarioHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_query: Annotated[str, Field(min_length=1, max_length=240)]
    basis: Annotated[str, Field(min_length=1, max_length=120)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class InvalidInterpretation(RuntimeError):
    """The model did not produce one valid preference update tool call."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = max(0, int(prompt_tokens))
        self.completion_tokens = max(0, int(completion_tokens))


class PreferenceToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    intent_mode: Literal["buying", "browsing", "unknown", "unchanged"] = "unchanged"
    update_type: Literal["merge", "replace_preferences", "product_change"] = "merge"
    correction_scope: Literal["corrected_attributes", "latest_unsolicited"] = (
        "corrected_attributes"
    )
    category: Annotated[str, Field(min_length=1, max_length=200)] = "unchanged"
    set_preferences: Annotated[
        list[ModelPreferenceValue], Field(default_factory=list, max_length=16)
    ]
    remove_preferences: Annotated[
        list[ModelPreferenceRemoval], Field(default_factory=list, max_length=16)
    ]
    no_preference_attributes: Annotated[
        list[RemovableAttribute], Field(default_factory=list, max_length=10)
    ]
    search_terms: Annotated[
        list[ShortText], Field(default_factory=list, max_length=24)
    ]
    scenario_hypotheses: Annotated[
        list[ModelScenarioHypothesis], Field(default_factory=list, max_length=3)
    ]
    runtime: ToolRuntime


class PreferenceWorkflowState(MessagesState):
    shopping_state: ShoppingState
    latest_user_message: str
    preference_patch: PreferencePatch | None
    prompt_tokens: int
    completion_tokens: int
    tool_applied: bool
    scenario_hypotheses: tuple[ScenarioHypothesis, ...]
    allow_scenario_hypotheses: bool


@tool("update_user_preferences", args_schema=PreferenceToolInput)
def update_user_preferences(
    intent_mode: Literal["buying", "browsing", "unknown", "unchanged"],
    update_type: Literal["merge", "replace_preferences", "product_change"],
    correction_scope: Literal["corrected_attributes", "latest_unsolicited"],
    category: str,
    set_preferences: list[ModelPreferenceValue],
    remove_preferences: list[ModelPreferenceRemoval],
    no_preference_attributes: list[RemovableAttribute],
    search_terms: list[str],
    scenario_hypotheses: list[ModelScenarioHypothesis],
    runtime: ToolRuntime,
) -> Command:
    """Validate and store shopper preferences in thread-scoped runtime state."""
    latest_message = str(runtime.state.get("latest_user_message", ""))
    current = runtime.state["shopping_state"]
    if not isinstance(current, ShoppingState):
        raise TypeError("shopping runtime state is invalid")
    explicit_preferences = [
        item for item in set_preferences if _is_explicit_span(item.value, latest_message)
    ]
    explicit_removals = [
        item
        for item in remove_preferences
        if item.value is None or _is_explicit_span(item.value, latest_message)
    ]
    explicit_search_terms = [
        term for term in search_terms if _is_explicit_span(term, latest_message)
    ]
    explicit_no_preferences = [
        attribute
        for attribute in no_preference_attributes
        if NO_PREFERENCE_RE.search(latest_message)
        and (
            _is_explicit_span(attribute.replace("_", " "), latest_message)
            or attribute == current.previous_ask_attribute
        )
    ]
    explicit_category = (
        category
        if category == "unchanged" or _is_explicit_span(category, latest_message)
        else "unchanged"
    )
    explicit_update_type = update_type
    if (
        update_type == "product_change"
        and explicit_category == "unchanged"
        and not explicit_preferences
        and not explicit_search_terms
    ):
        explicit_update_type = "merge"
    explicit_correction_scope = (
        "latest_unsolicited"
        if explicit_update_type == "replace_preferences"
        and correction_scope == "latest_unsolicited"
        and references_unnamed_earlier_preference(latest_message)
        else "corrected_attributes"
    )

    patch = PreferencePatch(
        intent_mode=intent_mode,
        update_type=explicit_update_type,
        correction_scope=explicit_correction_scope,
        category=explicit_category,
        set_preferences=[
            PreferenceValue(attribute=item.attribute, value=item.value)
            for item in explicit_preferences
        ],
        remove_preferences=[
            PreferenceRemoval(attribute=item.attribute, value=item.value)
            for item in explicit_removals
        ],
        no_preference_attributes=explicit_no_preferences,
        search_terms=explicit_search_terms,
    )
    allow_scenario_hypotheses = bool(
        runtime.state.get("allow_scenario_hypotheses", False)
    )
    validated_scenario_hypotheses = (
        validate_scenario_hypotheses(
            [
                ScenarioHypothesis(
                    scenario_query=item.scenario_query,
                    basis=item.basis,
                    confidence=item.confidence,
                )
                for item in scenario_hypotheses
            ],
            latest_message,
        )
        if allow_scenario_hypotheses
        else ()
    )
    if not runtime.tool_call_id:
        raise ValueError("preference tool call is missing its identifier")
    updated = apply_preference_patch(current, patch)
    return Command(
        update={
            "shopping_state": updated,
            "preference_patch": patch,
            "scenario_hypotheses": validated_scenario_hypotheses,
            "tool_applied": True,
            "messages": [
                ToolMessage(
                    content="Preferences updated.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _is_explicit_span(value: str, message: str) -> bool:
    value_tokens = EXPLICIT_TOKEN_RE.findall(str(value).lower())
    message_tokens = EXPLICIT_TOKEN_RE.findall(str(message).lower())
    if not value_tokens or len(value_tokens) > len(message_tokens):
        return False
    width = len(value_tokens)
    return any(
        message_tokens[index : index + width] == value_tokens
        for index in range(len(message_tokens) - width + 1)
    )


def _normalize_tool_arguments(raw_arguments: object) -> object:
    """Normalize compact but unambiguous tool output from smaller local models."""
    if not isinstance(raw_arguments, dict):
        return raw_arguments
    arguments = dict(raw_arguments)
    for field_name in ("set_preferences", "remove_preferences"):
        compact = arguments.get(field_name)
        if not isinstance(compact, dict):
            continue
        expanded: list[dict[str, object]] = []
        for attribute, raw_value in compact.items():
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                expanded.append({"attribute": attribute, "value": value})
        arguments[field_name] = expanded
    for field_name, allowed_attributes in (
        ("set_preferences", MODEL_PREFERENCE_ATTRIBUTES),
        ("remove_preferences", MODEL_REMOVABLE_ATTRIBUTES),
    ):
        values = arguments.get(field_name)
        if isinstance(values, list):
            arguments[field_name] = [
                item
                for item in values
                if isinstance(item, dict)
                and item.get("attribute") in allowed_attributes
            ]
    for field_name in (
        "no_preference_attributes",
        "search_terms",
        "scenario_hypotheses",
    ):
        value = arguments.get(field_name)
        if isinstance(value, (str, dict)):
            arguments[field_name] = [value]
    return arguments


@dataclass(frozen=True)
class Interpretation:
    state: ShoppingState
    prompt_tokens: int = 0
    completion_tokens: int = 0
    patch: PreferencePatch | None = None
    scenario_hypotheses: tuple[ScenarioHypothesis, ...] = ()


class PreferenceInterpreter:
    def __init__(self, model: object) -> None:
        self._usage_lock = Lock()
        self._usage_by_session: dict[str, tuple[int, int]] = {}
        self._bound_model = model.bind_tools(
            [update_user_preferences],
            tool_choice="update_user_preferences",
            strict=True,
            parallel_tool_calls=False,
        )
        self._checkpointer = InMemorySaver()
        graph = StateGraph(PreferenceWorkflowState)
        graph.add_node("model", self._call_model)
        graph.add_node(
            "tools", ToolNode([update_user_preferences], handle_tool_errors=False)
        )
        graph.add_edge(START, "model")
        graph.add_edge("model", "tools")
        graph.add_edge("tools", END)
        self._graph = graph.compile(checkpointer=self._checkpointer)

    @classmethod
    def from_environment(cls) -> "PreferenceInterpreter | None":
        enabled = os.getenv("OPENAI_ENABLED", "true").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        timeout = _bounded_float(
            os.getenv("OPENAI_TIMEOUT_SECONDS", "20"), minimum=1.0, maximum=60.0
        )
        max_retries = _bounded_int(
            os.getenv("OPENAI_MAX_RETRIES", "1"), minimum=0, maximum=3
        )
        model_kwargs = {
            "api_key": api_key,
            "model": model_name or "gpt-5.6-luna",
            "timeout": timeout,
            "max_retries": max_retries,
            "use_responses_api": not bool(base_url),
        }
        if base_url:
            model_kwargs["base_url"] = base_url
        reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
        if reasoning_effort:
            model_kwargs["reasoning_effort"] = reasoning_effort
        temperature = os.getenv("OPENAI_TEMPERATURE", "").strip()
        if temperature:
            model_kwargs["temperature"] = _bounded_float(
                temperature, minimum=0.0, maximum=2.0
            )
        model = ChatOpenAI(
            **model_kwargs,
        )
        return cls(model)

    def reset(self, session_id: str) -> None:
        self._checkpointer.delete_thread(session_id)

    def interpret(self, message: str, state: ShoppingState) -> Interpretation:
        config = {"configurable": {"thread_id": state.session_id}}
        workflow_input: PreferenceWorkflowState = {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "shopping_state": state,
            "latest_user_message": message,
            "preference_patch": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tool_applied": False,
            "scenario_hypotheses": (),
            "allow_scenario_hypotheses": bool(
                query_expansion_enabled()
                and state.turn == 0
                and looks_like_scenario_query(message)
            ),
        }
        with self._usage_lock:
            self._usage_by_session.pop(state.session_id, None)
        try:
            result = self._graph.invoke(workflow_input, config=config)
        except InvalidInterpretation:
            raise
        except Exception as exc:
            with self._usage_lock:
                prompt_tokens, completion_tokens = self._usage_by_session.get(
                    state.session_id, (0, 0)
                )
            raise InvalidInterpretation(
                "preference tool execution failed",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ) from exc
        finally:
            with self._usage_lock:
                self._usage_by_session.pop(state.session_id, None)
        updated = result.get("shopping_state")
        if not result.get("tool_applied") or not isinstance(updated, ShoppingState):
            raise InvalidInterpretation("preference tool did not update runtime state")
        return Interpretation(
            state=updated,
            prompt_tokens=max(0, int(result.get("prompt_tokens", 0))),
            completion_tokens=max(0, int(result.get("completion_tokens", 0))),
            patch=(
                result.get("preference_patch")
                if isinstance(result.get("preference_patch"), PreferencePatch)
                else None
            ),
            scenario_hypotheses=tuple(
                item
                for item in result.get("scenario_hypotheses", ())
                if isinstance(item, ScenarioHypothesis)
            ),
        )

    def _call_model(self, state: PreferenceWorkflowState) -> dict:
        shopping_state = state["shopping_state"]
        payload = {
            "latest_shopper_message": state["latest_user_message"],
            "current_state": shopping_state.to_prompt_dict(),
        }
        system_prompt = SYSTEM_PROMPT
        if state.get("allow_scenario_hypotheses", False):
            system_prompt += TEMPORARY_RETRIEVAL_PROMPT
        response = self._bound_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(payload, sort_keys=True)),
            ]
        )
        if not isinstance(response, AIMessage):
            raise InvalidInterpretation("model response is not an AI message")
        prompt_tokens, completion_tokens = _usage_from_message(response)
        with self._usage_lock:
            self._usage_by_session[shopping_state.session_id] = (
                prompt_tokens,
                completion_tokens,
            )
        if len(response.tool_calls) != 1:
            raise InvalidInterpretation(
                "model must make exactly one tool call",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        if response.tool_calls[0].get("name") != "update_user_preferences":
            raise InvalidInterpretation(
                "model called an unsupported tool",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        arguments = _normalize_tool_arguments(response.tool_calls[0].get("args", {}))
        try:
            patch_arguments = dict(arguments)
            raw_hypotheses = patch_arguments.pop("scenario_hypotheses", [])
            PreferencePatch.model_validate(patch_arguments)
            if not isinstance(raw_hypotheses, list):
                raise TypeError("scenario_hypotheses must be a list")
            for item in raw_hypotheses:
                ModelScenarioHypothesis.model_validate(item)
        except Exception as exc:
            raise InvalidInterpretation(
                "model returned invalid preference arguments",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ) from exc
        response.tool_calls[0]["args"] = arguments
        return {
            "messages": [response],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


def _usage_from_message(message: AIMessage) -> tuple[int, int]:
    usage = message.usage_metadata or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    if not prompt_tokens and not completion_tokens:
        token_usage = message.response_metadata.get("token_usage", {})
        prompt_tokens = token_usage.get(
            "prompt_tokens", token_usage.get("input_tokens", 0)
        )
        completion_tokens = token_usage.get(
            "completion_tokens", token_usage.get("output_tokens", 0)
        )
    return _nonnegative_int(prompt_tokens), _nonnegative_int(completion_tokens)


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _bounded_float(value: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError:
        parsed = minimum
    return min(maximum, max(minimum, parsed))


def _bounded_int(value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed = minimum
    return min(maximum, max(minimum, parsed))
