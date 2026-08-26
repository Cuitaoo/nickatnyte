from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from starter.preference_tool import (
    PreferencePatch,
    PreferenceRemoval,
    PreferenceValue,
    apply_preference_patch,
)
from starter.state import ShoppingState


SYSTEM_PROMPT = """You interpret one shopper message for a catalog search agent.
You must call update_user_preferences exactly once and must not answer in prose.
Keep prior preferences unless the shopper changes, rejects, or overrides them.
Set reset_product_preferences=true only for a clear product-intent override.
Use unchanged for intent_mode and category when the message does not update them.
Normalize concise search terms. Never create, request, or return product IDs.
Allowed attributes are category, material, color, size, style, brand, budget,
feature, use_case, and other."""


class InvalidInterpretation(RuntimeError):
    """The model did not produce one valid preference update tool call."""


class PreferenceToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    intent_mode: Literal["buying", "browsing", "unknown", "unchanged"]
    category: str
    set_preferences: list[PreferenceValue]
    remove_preferences: list[PreferenceRemoval]
    no_preference_attributes: list[str]
    reset_product_preferences: bool
    search_terms: list[str]
    runtime: ToolRuntime


class PreferenceWorkflowState(MessagesState):
    shopping_state: ShoppingState
    latest_user_message: str
    prompt_tokens: int
    completion_tokens: int
    tool_applied: bool


@tool("update_user_preferences", args_schema=PreferenceToolInput)
def update_user_preferences(
    intent_mode: Literal["buying", "browsing", "unknown", "unchanged"],
    category: str,
    set_preferences: list[PreferenceValue],
    remove_preferences: list[PreferenceRemoval],
    no_preference_attributes: list[str],
    reset_product_preferences: bool,
    search_terms: list[str],
    runtime: ToolRuntime,
) -> Command:
    """Validate and store shopper preferences in thread-scoped runtime state."""
    patch = PreferencePatch(
        intent_mode=intent_mode,
        category=category,
        set_preferences=set_preferences,
        remove_preferences=remove_preferences,
        no_preference_attributes=no_preference_attributes,
        reset_product_preferences=reset_product_preferences,
        search_terms=search_terms,
    )
    current = runtime.state["shopping_state"]
    if not isinstance(current, ShoppingState):
        raise TypeError("shopping runtime state is invalid")
    if not runtime.tool_call_id:
        raise ValueError("preference tool call is missing its identifier")
    updated = apply_preference_patch(current, patch)
    return Command(
        update={
            "shopping_state": updated,
            "tool_applied": True,
            "messages": [
                ToolMessage(
                    content="Preferences updated.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@dataclass(frozen=True)
class Interpretation:
    state: ShoppingState
    prompt_tokens: int = 0
    completion_tokens: int = 0


class PreferenceInterpreter:
    def __init__(self, model: object) -> None:
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
        timeout = _bounded_float(
            os.getenv("OPENAI_TIMEOUT_SECONDS", "20"), minimum=1.0, maximum=60.0
        )
        max_retries = _bounded_int(
            os.getenv("OPENAI_MAX_RETRIES", "1"), minimum=0, maximum=3
        )
        model = ChatOpenAI(
            api_key=api_key,
            model=model_name or "gpt-5.6-luna",
            timeout=timeout,
            max_retries=max_retries,
            use_responses_api=True,
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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tool_applied": False,
        }
        try:
            result = self._graph.invoke(workflow_input, config=config)
        except InvalidInterpretation:
            raise
        except Exception as exc:
            raise InvalidInterpretation("preference tool execution failed") from exc
        updated = result.get("shopping_state")
        if not result.get("tool_applied") or not isinstance(updated, ShoppingState):
            raise InvalidInterpretation("preference tool did not update runtime state")
        return Interpretation(
            state=updated,
            prompt_tokens=max(0, int(result.get("prompt_tokens", 0))),
            completion_tokens=max(0, int(result.get("completion_tokens", 0))),
        )

    def _call_model(self, state: PreferenceWorkflowState) -> dict:
        shopping_state = state["shopping_state"]
        profile = shopping_state.user_profile
        payload = {
            "latest_shopper_message": state["latest_user_message"],
            "current_state": shopping_state.to_prompt_dict(),
            "profile": {
                "summary": str(profile.get("summary", "")),
                "preference_tags": [
                    str(value) for value in profile.get("preference_tags", [])
                ],
            },
        }
        response = self._bound_model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, sort_keys=True)),
            ]
        )
        if not isinstance(response, AIMessage):
            raise InvalidInterpretation("model response is not an AI message")
        if len(response.tool_calls) != 1:
            raise InvalidInterpretation("model must make exactly one tool call")
        if response.tool_calls[0].get("name") != "update_user_preferences":
            raise InvalidInterpretation("model called an unsupported tool")
        prompt_tokens, completion_tokens = _usage_from_message(response)
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
