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
        "update_type": "merge",
        "correction_scope": "corrected_attributes",
        "category": "unchanged",
        "set_preferences": [],
        "remove_preferences": [],
        "no_preference_attributes": [],
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
    def test_omitted_no_op_tool_fields_use_safe_defaults(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": {
                        "category": "ankle boots",
                        "set_preferences": [
                            {"attribute": "material", "value": "leather"}
                        ],
                    },
                    "id": "call_defaults",
                    "type": "tool_call",
                }
            ],
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "I am looking for ankle boots made of leather",
            ShoppingState.new("s", {}),
        )

        self.assertEqual(result.state.category, "ankle boots")
        self.assertEqual(result.state.preferences, {"material": ("leather",)})
        self.assertEqual(result.state.last_update_type, "merge")

    def test_compact_preference_mapping_is_normalized_without_rewriting(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": {
                        "category": "ankle boots",
                        "set_preferences": {
                            "material": "leather",
                            "category": "ankle boots",
                        },
                    },
                    "id": "call_compact_mapping",
                    "type": "tool_call",
                }
            ],
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "I am looking for ankle boots made of leather",
            ShoppingState.new("s", {}),
        )

        self.assertEqual(result.state.category, "ankle boots")
        self.assertEqual(result.state.preferences, {"material": ("leather",)})

    def test_unsupported_no_preference_output_cannot_clear_confirmed_state(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        no_preference_attributes=[
                            "material",
                            "color",
                            "brand",
                            "size",
                        ]
                    ),
                    "id": "call_unsupported_no_preference",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("s", {}), preferences={"material": ("cotton",)}
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "For that, what matters is: Imported.", state
        )

        self.assertEqual(result.state.preferences, {"material": ("cotton",)})
        self.assertEqual(result.state.no_preference_attributes, frozenset())

    def test_generic_no_preference_answer_applies_only_to_previous_question(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        no_preference_attributes=["brand", "material"]
                    ),
                    "id": "call_direct_no_preference",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("s", {}),
            preferences={"material": ("cotton",)},
            previous_ask_attribute="brand",
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "I don't have an additional preference.", state
        )

        self.assertEqual(result.state.preferences, {"material": ("cotton",)})
        self.assertEqual(result.state.no_preference_attributes, frozenset({"brand"}))

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

    def test_product_change_clears_product_state_but_keeps_profile(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        intent_mode="buying",
                        update_type="product_change",
                        category="waterproof hiking boots",
                        set_preferences=[
                            {"attribute": "feature", "value": "waterproof"},
                            {"attribute": "use_case", "value": "hiking"},
                        ],
                    ),
                    "id": "call_product_change",
                    "type": "tool_call",
                }
            ],
        )
        profile = {"summary": "prefers comfort", "preference_tags": ["fit"]}
        state = replace(
            ShoppingState.new("s", profile),
            category="shirts",
            preferences={"color": ("red",)},
            asked_attributes=("material",),
            previous_ask_attribute="material",
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "Actually, I need waterproof hiking boots instead.", state
        )

        self.assertEqual(result.state.category, "waterproof hiking boots")
        self.assertNotIn("color", result.state.preferences)
        self.assertEqual(result.state.preferences["feature"], ("waterproof",))
        self.assertEqual(result.state.asked_attributes, ())
        self.assertEqual(result.state.user_profile, profile)
        self.assertEqual(result.state.last_update_type, "product_change")

    def test_preference_replacement_preserves_category_and_unrelated_state(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        update_type="replace_preferences",
                        set_preferences=[
                            {"attribute": "material", "value": "suede"}
                        ],
                    ),
                    "id": "call_preference_change",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("s", {}),
            category="women's boots",
            preferences={"material": ("leather",), "color": ("black",)},
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "Ignore leather; make it suede.", state
        )

        self.assertEqual(result.state.category, "women's boots")
        self.assertEqual(
            result.state.preferences,
            {"color": ("black",), "material": ("suede",)},
        )
        self.assertEqual(result.state.last_update_type, "replace_preferences")

    def test_unnamed_correction_scope_retires_latest_unsolicited_evidence(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        update_type="replace_preferences",
                        correction_scope="latest_unsolicited",
                        set_preferences=[
                            {"attribute": "feature", "value": "Water Resistant"}
                        ],
                    ),
                    "id": "call_unnamed_correction",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("public_0003", {}),
            category="watches wrist watches",
            preferences={"material": ("stainless steel band",)},
            no_preference_attributes=frozenset({"color"}),
            search_terms=("stainless steel band",),
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material",
                    values=("stainless steel band",),
                    terms=("stainless steel band",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
            previous_ask_attribute="feature",
            turn=2,
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "Actually, ignore my earlier preference. What I need is: Water Resistant.",
            state,
        )

        self.assertEqual(result.state.category, "watches wrist watches")
        self.assertEqual(
            result.state.preferences,
            {"feature": ("water resistant",)},
        )
        self.assertEqual(result.state.search_terms, ())
        self.assertEqual(
            result.state.no_preference_attributes,
            frozenset({"color"}),
        )

    def test_latest_unsolicited_scope_requires_unnamed_correction_evidence(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        update_type="replace_preferences",
                        correction_scope="latest_unsolicited",
                        set_preferences=[
                            {"attribute": "feature", "value": "water resistant"}
                        ],
                    ),
                    "id": "call_unsupported_scope",
                    "type": "tool_call",
                }
            ],
        )
        state = replace(
            ShoppingState.new("s", {}),
            category="watches",
            preferences={"material": ("steel",)},
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material",
                    values=("steel",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "Actually, make it water resistant.",
            state,
        )

        self.assertEqual(
            result.state.preferences,
            {
                "material": ("steel",),
                "feature": ("water resistant",),
            },
        )

    def test_rewritten_or_inferred_values_are_not_applied(self) -> None:
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "update_user_preferences",
                    "args": tool_args(
                        intent_mode="buying",
                        update_type="product_change",
                        category="sneakers",
                        set_preferences=[
                            {"attribute": "use_case", "value": "running"}
                        ],
                        search_terms=["sneakers", "running shoes"],
                    ),
                    "id": "call_rewrite",
                    "type": "tool_call",
                }
            ],
        )

        result = PreferenceInterpreter(FakeChatModel(response)).interpret(
            "I need trainers.", ShoppingState.new("s", {})
        )

        self.assertEqual(result.state.intent_mode, "buying")
        self.assertIsNone(result.state.category)
        self.assertEqual(result.state.preferences, {})
        self.assertEqual(result.state.search_terms, ())
        self.assertEqual(result.state.last_update_type, "merge")

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

    def test_custom_openai_compatible_endpoint_uses_chat_completions(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_ENABLED": "true",
                    "OPENAI_API_KEY": "EMPTY",
                    "OPENAI_MODEL": "qwen3:4b",
                    "OPENAI_BASE_URL": "http://localhost:11434/v1",
                    "OPENAI_REASONING_EFFORT": "none",
                    "OPENAI_TEMPERATURE": "0",
                },
                clear=False,
            ),
            patch("starter.llm_agent.ChatOpenAI") as chat_openai,
            patch.object(PreferenceInterpreter, "__init__", return_value=None),
        ):
            PreferenceInterpreter.from_environment()

        chat_openai.assert_called_once()
        kwargs = chat_openai.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://localhost:11434/v1")
        self.assertFalse(kwargs["use_responses_api"])
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
