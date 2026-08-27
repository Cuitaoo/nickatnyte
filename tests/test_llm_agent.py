from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from starter.llm_agent import (
    InvalidInterpretation,
    PreferenceInterpreter,
)
from starter.state import PreferenceEvidence, ShoppingState


def tool_args(**overrides: object) -> dict:
    values = {
        "intent_mode": "unchanged",
        "category": "unchanged",
        "set_preferences": [],
        "remove_preferences": [],
        "no_preference_attributes": [],
        "reset_product_preferences": False,
        "search_terms": [],
    }
    values.update(overrides)
    return values


class FakeChatModel:
    def __init__(self, *responses: AIMessage) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.bind_arguments: dict | None = None
        self.invocations: list[list] = []

    def bind_tools(self, tools: list, **kwargs: object) -> "FakeChatModel":
        self.bind_arguments = {"tools": tools, **kwargs}
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        self.invocations.append(messages)
        return self.responses.pop(0)


class PreferenceInterpreterTest(unittest.TestCase):
    def test_valid_tool_call_updates_runtime_state_and_usage_once(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        intent_mode="buying",
                        category="shoes",
                        set_preferences=[{"attribute": "color", "value": "blue"}],
                        search_terms=["running"],
                    ),
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 80,
                "output_tokens": 20,
                "total_tokens": 100,
            },
        )
        model = FakeChatModel(response)
        state = ShoppingState.new(
            "s", {"summary": "likes comfort", "preference_tags": ["fit"]}
        )
        interpreter = PreferenceInterpreter(model)

        result = interpreter.interpret("blue running shoes", state)

        self.assertEqual(result.state.category, "shoes")
        self.assertEqual(result.state.preferences, {"color": ("blue",)})
        self.assertEqual(result.state.search_terms, ("running",))
        self.assertEqual(
            result.state.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="color",
                    values=("blue",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
                PreferenceEvidence(
                    attribute="other",
                    terms=("running",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
        )
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (80, 20))
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.bind_arguments["tool_choice"], "update_user_preferences")
        self.assertTrue(model.bind_arguments["strict"])
        prompt_text = " ".join(
            str(message.content) for message in model.invocations[0]
        )
        self.assertIn("blue running shoes", prompt_text)
        self.assertNotIn("likes comfort", prompt_text)
        self.assertNotIn("fit", prompt_text)

    def test_existing_state_is_available_to_the_tool(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        set_preferences=[
                            {"attribute": "material", "value": "leather"}
                        ]
                    ),
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("s", {}), preferences={"color": ("black",)}
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret("leather", state)

        self.assertEqual(
            result.state.preferences,
            {"color": ("black",), "material": ("leather",)},
        )

    def test_tool_path_records_answer_to_previous_question_as_clarification(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        set_preferences=[
                            {"attribute": "feature", "value": "machine washable"}
                        ]
                    ),
                    "id": "call_clarification",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("s", {}),
            previous_ask_attribute="feature",
            turn=1,
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "machine washable",
            state,
        )

        self.assertEqual(
            result.state.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="feature",
                    values=("machine washable",),
                    source_turn=2,
                    source_kind="clarification",
                ),
            ),
        )

    def test_missing_tool_call_raises_invalid_interpretation(self) -> None:
        interpreter = PreferenceInterpreter(FakeChatModel(AIMessage(content="plain text")))

        with self.assertRaises(InvalidInterpretation):
            interpreter.interpret("shoes", ShoppingState.new("s", {}))

    def test_invalid_tool_call_preserves_billable_usage(self) -> None:
        response = AIMessage(
            content="plain text",
            usage_metadata={
                "input_tokens": 31,
                "output_tokens": 7,
                "total_tokens": 38,
            },
        )
        interpreter = PreferenceInterpreter(FakeChatModel(response))

        with self.assertRaises(InvalidInterpretation) as caught:
            interpreter.interpret("shoes", ShoppingState.new("s", {}))

        self.assertEqual(caught.exception.prompt_tokens, 31)
        self.assertEqual(caught.exception.completion_tokens, 7)

    def test_tool_execution_failure_preserves_billable_usage(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(category="shoes"),
                    "id": "call_fails",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 41,
                "output_tokens": 9,
                "total_tokens": 50,
            },
        )
        interpreter = PreferenceInterpreter(FakeChatModel(response))

        with (
            patch(
                "starter.llm_agent.apply_preference_patch",
                side_effect=RuntimeError("storage failed"),
            ),
            self.assertRaises(InvalidInterpretation) as caught,
        ):
            interpreter.interpret("shoes", ShoppingState.new("s", {}))

        self.assertEqual(caught.exception.prompt_tokens, 41)
        self.assertEqual(caught.exception.completion_tokens, 9)

    def test_more_than_one_tool_call_is_rejected(self) -> None:
        calls = [
            {
                "name": "update_user_preferences",
                "args": tool_args(),
                "id": f"call_{index}",
                "type": "tool_call",
            }
            for index in (1, 2)
        ]
        interpreter = PreferenceInterpreter(
            FakeChatModel(AIMessage(content="", tool_calls=calls))
        )

        with self.assertRaises(InvalidInterpretation):
            interpreter.interpret("shoes", ShoppingState.new("s", {}))

    def test_malformed_tool_arguments_are_rejected(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(intent_mode="invalid"),
                    "id": "call_bad",
                    "type": "tool_call",
                }
            ],
        )
        interpreter = PreferenceInterpreter(FakeChatModel(response))

        with self.assertRaises(InvalidInterpretation):
            interpreter.interpret("shoes", ShoppingState.new("s", {}))

    def test_disabled_environment_does_not_construct_a_client(self) -> None:
        with patch.dict(os.environ, {"OPENAI_ENABLED": "false"}, clear=False):
            self.assertIsNone(PreferenceInterpreter.from_environment())

    def test_missing_api_key_does_not_construct_a_client(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_ENABLED": "true", "OPENAI_API_KEY": ""},
            clear=False,
        ):
            self.assertIsNone(PreferenceInterpreter.from_environment())


if __name__ == "__main__":
    unittest.main()
