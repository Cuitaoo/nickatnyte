from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.llm_agent import Interpretation, InvalidInterpretation
from starter.preference_tool import (
    PreferencePatch,
    PreferenceValue,
    apply_preference_patch,
    parse_preference_fallback,
)
from starter.state import ALLOWED_PREFERENCE_ATTRIBUTES, ShoppingState


CATALOG = [
    {
        "parent_asin": "BLUE_SHOE",
        "title": "Blue Running Shoe",
        "categories": ["Women", "Shoes", "Running"],
        "features": ["breathable mesh", "comfortable fit"],
        "details": {"color": "blue", "size": "8"},
        "description": ["road running sneaker"],
        "store": "RunCo",
        "price": 60.0,
        "average_rating": 4.7,
        "rating_number": 300,
    },
    {
        "parent_asin": "BLACK_BOOT",
        "title": "Black Leather Hiking Boot",
        "categories": ["Men", "Shoes", "Boots"],
        "features": ["waterproof leather"],
        "details": {"color": "black", "size": "10"},
        "description": ["winter hiking trail boot"],
        "store": "TrailCo",
        "price": 90.0,
        "average_rating": 4.6,
        "rating_number": 250,
    },
    {
        "parent_asin": "RED_SHIRT",
        "title": "Red Cotton Shirt",
        "categories": ["Men", "Clothing", "Shirts"],
        "features": ["soft cotton"],
        "details": {"color": "red", "size": "large"},
        "description": ["casual crew shirt"],
        "store": "Basics",
        "price": 25.0,
        "average_rating": 4.2,
        "rating_number": 100,
    },
]


class QueueInterpreter:
    def __init__(self, patches: list[PreferencePatch]) -> None:
        self.patches = list(patches)
        self.reset_ids: list[str] = []
        self.calls = 0

    def reset(self, session_id: str) -> None:
        self.reset_ids.append(session_id)

    def interpret(self, message: str, state) -> Interpretation:
        self.calls += 1
        patch = self.patches.pop(0)
        return Interpretation(
            apply_preference_patch(state, patch),
            prompt_tokens=10 + self.calls,
            completion_tokens=2,
        )


class FailingInterpreter:
    def reset(self, session_id: str) -> None:
        return None

    def interpret(self, message: str, state) -> Interpretation:
        raise TimeoutError("simulated model timeout")


class UsageFailingInterpreter(FailingInterpreter):
    def interpret(self, message: str, state) -> Interpretation:
        raise InvalidInterpretation(
            "invalid tool call", prompt_tokens=17, completion_tokens=5
        )


class AgentIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in CATALOG),
            encoding="utf-8",
        )
        cls.catalog_ids = {product["parent_asin"] for product in CATALOG}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_respond_before_reset_is_rejected(self) -> None:
        agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)

        with self.assertRaises(RuntimeError):
            agent.respond("missing", "blue shoes", 1, 10)

    def test_preferences_persist_across_turns_and_usage_is_per_turn(self) -> None:
        interpreter = QueueInterpreter(
            [
                PreferencePatch(
                    category="shoes",
                    set_preferences=[
                        PreferenceValue(attribute="color", value="blue")
                    ],
                ),
                PreferencePatch(),
            ]
        )
        agent = Agent(self.catalog_path, interpreter=interpreter)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        first = agent.respond("s", "blue shoes", 1, 10)
        second = agent.respond("s", "show me more", 2, 10)

        self.assertEqual(agent.session_state("s").preferences, {"color": ("blue",)})
        self.assertEqual(first["usage"], {"prompt_tokens": 11, "completion_tokens": 2})
        self.assertEqual(second["usage"], {"prompt_tokens": 12, "completion_tokens": 2})
        self.assertEqual(agent.session_state("s").prompt_tokens, 23)
        self.assertEqual(agent.session_state("s").completion_tokens, 4)
        self.assertEqual(interpreter.calls, 2)

    def test_api_failure_uses_fallback_and_returns_contract_shape(self) -> None:
        agent = Agent(self.catalog_path, interpreter=FailingInterpreter())
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        response = agent.respond("s", "black leather boots", 1, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "BLACK_BOOT")
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertIn(response["ask_attribute"], {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        })
        self.assertEqual(
            set(response), {"message", "ask_attribute", "recommendations", "usage"}
        )

    def test_retrieval_failure_uses_deterministic_catalog_fallback(self) -> None:
        agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        with patch.object(agent.retriever, "search", side_effect=RuntimeError("boom")):
            response = agent.respond("s", "shoes", 1, 2)

        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["BLUE_SHOE", "BLACK_BOOT"],
        )

    def test_invalid_model_output_reports_usage_before_falling_back(self) -> None:
        agent = Agent(self.catalog_path, interpreter=UsageFailingInterpreter())
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        response = agent.respond("s", "black leather boots", 1, 10)

        self.assertEqual(
            response["usage"], {"prompt_tokens": 17, "completion_tokens": 5}
        )
        self.assertEqual(agent.session_state("s").prompt_tokens, 17)
        self.assertEqual(agent.session_state("s").completion_tokens, 5)

    def test_sessions_are_isolated_and_reset_replaces_existing_state(self) -> None:
        agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        profile = {"summary": "", "preference_tags": []}
        agent.reset("one", profile)
        agent.reset("two", profile)
        agent.respond("one", "blue shoes", 1, 10)

        self.assertEqual(agent.session_state("one").preferences["color"], ("blue",))
        self.assertEqual(agent.session_state("two").preferences, {})

        agent.reset("one", profile)
        self.assertEqual(agent.session_state("one").preferences, {})

    def test_questions_are_not_repeated_and_no_preference_is_remembered(self) -> None:
        agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        first = agent.respond("s", "I'm looking for shoes", 1, 10)
        second = agent.respond(
            "s", f"I don't have a preference for {first['ask_attribute']}.", 2, 10
        )

        self.assertIsNotNone(first["ask_attribute"])
        self.assertNotEqual(second["ask_attribute"], first["ask_attribute"])
        self.assertIn(first["ask_attribute"], agent.session_state("s").no_preference_attributes)

    def test_recommendations_are_unique_valid_and_limited(self) -> None:
        agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        response = agent.respond("s", "shoes", 10, 2)
        identifiers = [item["parent_asin"] for item in response["recommendations"]]

        self.assertLessEqual(len(identifiers), 2)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(set(identifiers) <= self.catalog_ids)
        self.assertIsNone(response["ask_attribute"])

    def test_low_confidence_deferral_can_be_enabled(self) -> None:
        with patch.dict(
            "os.environ",
            {"TECHJAM_DEFER_LOW_CONFIDENCE_RECOMMENDATIONS": "true"},
        ):
            agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        response = agent.respond(
            "s", "I'm looking for shoes, but I'm still exploring.", 1, 10
        )

        self.assertIsNotNone(response["ask_attribute"])
        self.assertEqual(response["recommendations"], [])

    def test_low_confidence_deferral_can_be_disabled(self) -> None:
        with patch.dict(
            "os.environ",
            {"TECHJAM_DEFER_LOW_CONFIDENCE_RECOMMENDATIONS": "false"},
        ):
            agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        response = agent.respond(
            "s", "I'm looking for shoes, but I'm still exploring.", 1, 10
        )

        self.assertIsNotNone(response["ask_attribute"])
        self.assertNotEqual(response["recommendations"], [])

    def test_intent_override_clears_old_questions_and_recommendations(self) -> None:
        interpreter = QueueInterpreter(
            [
                PreferencePatch(
                    category="shoes",
                    set_preferences=[PreferenceValue(attribute="material", value="leather")],
                ),
                PreferencePatch(
                    reset_product_preferences=True,
                    category="shirts",
                    set_preferences=[PreferenceValue(attribute="color", value="red")],
                ),
            ]
        )
        agent = Agent(self.catalog_path, interpreter=interpreter)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "likes comfort", "preference_tags": ["comfort"]})

        first = agent.respond("s", "leather shoes", 1, 10)
        second = agent.respond("s", "Actually, I need a red shirt instead", 2, 10)
        state = agent.session_state("s")

        self.assertEqual(state.category, "shirts")
        self.assertEqual(state.preferences, {"color": ("red",)})
        self.assertNotIn(first["ask_attribute"], state.asked_attributes[:-1])
        self.assertEqual(second["recommendations"][0]["parent_asin"], "RED_SHIRT")

    def test_same_product_correction_preserves_confirmed_agent_evidence(self) -> None:
        interpreter = QueueInterpreter(
            [
                PreferencePatch(
                    reset_product_preferences=True,
                    set_preferences=[
                        PreferenceValue(attribute="material", value="cotton")
                    ],
                )
            ]
        )
        agent = Agent(self.catalog_path, interpreter=interpreter)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})
        confirmed = apply_preference_patch(
            ShoppingState(
                session_id="s",
                user_profile={},
                category="shirts",
                previous_ask_attribute="feature",
                turn=1,
            ),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="soft")
                ]
            ),
        )
        agent._sessions["s"] = replace(
            confirmed,
            asked_attributes=("feature", "material"),
            previous_ask_attribute="material",
            latest_recommendations=("OLD_PRODUCT",),
            turn=2,
        )

        response = agent.respond(
            "s",
            "Actually, ignore the old material; I need cotton.",
            3,
            2,
        )
        state = agent.session_state("s")
        identifiers = [item["parent_asin"] for item in response["recommendations"]]

        self.assertEqual(
            state.preferences,
            {"feature": ("soft",), "material": ("cotton",)},
        )
        self.assertNotIn("OLD_PRODUCT", state.latest_recommendations)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(set(identifiers) <= self.catalog_ids)
        self.assertLessEqual(len(identifiers), 2)
        self.assertEqual(
            set(response),
            {"message", "ask_attribute", "recommendations", "usage"},
        )

    def test_model_and_fallback_corrections_have_equivalent_state_semantics(self) -> None:
        state = apply_preference_patch(
            replace(ShoppingState.new("s", {}), category="accessories belts"),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="hand wash only")
                ]
            ),
        )
        state = replace(state, turn=1)
        message = "Actually, ignore my earlier preference. What I need is: nylon."
        fallback_patch = parse_preference_fallback(message, state)
        model_patch = PreferencePatch(
            intent_mode="buying",
            category="nylon",
            set_preferences=[
                PreferenceValue(attribute="material", value="nylon")
            ],
            reset_product_preferences=True,
            search_terms=["nylon"],
        )

        model_state = apply_preference_patch(state, model_patch)
        fallback_state = apply_preference_patch(state, fallback_patch)

        self.assertEqual(model_state.category, fallback_state.category)
        self.assertEqual(model_state.preferences, fallback_state.preferences)
        self.assertEqual(
            model_state.no_preference_attributes,
            fallback_state.no_preference_attributes,
        )
        self.assertEqual(
            model_state.preference_evidence,
            fallback_state.preference_evidence,
        )

    def test_agent_passes_candidate_diagnostics_to_question_policy(self) -> None:
        agent = Agent(self.catalog_path, interpreter=None)
        self.addCleanup(agent.close)
        agent.reset("s", {"summary": "", "preference_tags": []})

        response = agent.respond("s", "running shoes for wet weather", 1, 10)

        self.assertIn(response["ask_attribute"], ALLOWED_PREFERENCE_ATTRIBUTES | {None})
        self.assertNotEqual(response["ask_attribute"], "material")


class VocabularyTranslationTest(unittest.TestCase):
    """The LLM interpreter is the semantic layer: it translates shopper
    vocabulary and intent into catalog terms, and that translation must flow
    through state into retrieval."""

    def test_prompt_instructs_vocabulary_translation(self) -> None:
        from starter.llm_agent import SYSTEM_PROMPT

        self.assertIn("translate", SYSTEM_PROMPT.lower())
        self.assertIn("search_terms", SYSTEM_PROMPT)

    def test_translated_terms_reach_retrieval(self) -> None:
        products = [
            {
                "parent_asin": "PANTIES_PACK",
                "title": "Cotton Panties Multipack",
                "categories": ["Women", "Clothing", "Lingerie", "Panties"],
                "features": ["soft cotton"],
                "details": {"count": "6"},
                "description": ["everyday underwear"],
                "store": "Basics",
                "price": 18.0,
                "average_rating": 4.4,
                "rating_number": 900,
            },
            {
                "parent_asin": "WOOL_SOCKS",
                "title": "Merino Wool Hiking Socks",
                "categories": ["Women", "Clothing", "Socks"],
                "features": ["warm wool"],
                "details": {"count": "3"},
                "description": ["trail socks"],
                "store": "TrailCo",
                "price": 15.0,
                "average_rating": 4.8,
                "rating_number": 1500,
            },
        ]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        # The interpreter translates "knickers" (absent from the catalog) into
        # catalog vocabulary, as the system prompt instructs the real model to.
        interpreter = QueueInterpreter(
            [
                PreferencePatch(
                    intent_mode="buying",
                    category="panties",
                    search_terms=["knickers", "panties", "underwear"],
                )
            ]
        )
        agent = Agent(catalog_path, interpreter=interpreter)
        self.addCleanup(agent.close)
        agent.reset("s", {})

        response = agent.respond("s", "I'm looking for knickers", 1, 2)

        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(identifiers[0], "PANTIES_PACK")


if __name__ == "__main__":
    unittest.main()
