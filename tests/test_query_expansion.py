from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from starter.query_expansion import (
    ScenarioHypothesis,
    looks_like_scenario_query,
    validate_scenario_hypotheses,
)
from starter.state import ShoppingState
from starter.vector_index import VectorCatalogIndex


class ScenarioExpansionTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "TECHJAM_QUERY_EXPANSION_ENABLED": "true",
            "TECHJAM_QUERY_EXPANSION_MODE": "recall",
        },
    )
    def test_only_actionable_scenarios_open_the_expansion_branch(self) -> None:
        self.assertTrue(
            looks_like_scenario_query(
                "Search for appropriate men's shoes for wet weather."
            )
        )
        self.assertTrue(
            looks_like_scenario_query(
                "Can you recommend a good laptop backpack for commuting?"
            )
        )
        self.assertFalse(
            looks_like_scenario_query(
                "I'm looking for a good men's jacket for Canada."
            )
        )

    @patch.dict(
        os.environ,
        {
            "TECHJAM_QUERY_EXPANSION_ENABLED": "true",
            "TECHJAM_QUERY_EXPANSION_MODE": "recall",
        },
    )
    def test_scenario_validation_removes_explicit_category_and_location_terms(self) -> None:
        hypotheses = validate_scenario_hypotheses(
            [
                ScenarioHypothesis(
                    scenario_query="men's insulated cold weather jacket for Canada",
                    basis="Canada",
                    confidence=0.85,
                )
            ],
            "I'm looking for a good men's jacket for Canada.",
        )

        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].scenario_query, "insulated cold weather")

    @patch.dict(
        os.environ,
        {
            "TECHJAM_QUERY_EXPANSION_ENABLED": "true",
            "TECHJAM_QUERY_EXPANSION_MODE": "recall",
        },
    )
    def test_scenario_validation_rejects_plain_query_repetition(self) -> None:
        hypotheses = validate_scenario_hypotheses(
            [
                ScenarioHypothesis(
                    scenario_query="men jacket Canada",
                    basis="Canada",
                    confidence=0.85,
                )
            ],
            "I'm looking for a good men's jacket for Canada.",
        )

        self.assertEqual(hypotheses, ())

    def test_vector_index_keeps_category_feature_and_scenario_queries_separate(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def encode(self, queries, **_kwargs):
                self.queries = list(queries)
                return list(range(len(self.queries)))

        model = FakeModel()
        index = object.__new__(VectorCatalogIndex)
        index.category_embeddings = "category-matrix"
        index.feature_embeddings = "feature-matrix"
        index.embeddings = "full-matrix"
        index._embedding_model = lambda: model
        index._rank_matrix_with_scores = (
            lambda matrix, vector, _top_k: [(f"{matrix}-{vector}", 0.9)]
        )
        state = replace(
            ShoppingState.new("session", {}),
            category="men jackets",
            preferences={"feature": ("waterproof",)},
        )

        routes = index.search_routes_with_scores(
            state,
            "Find a good men's jacket for commuting.",
            5,
            allowed_routes={
                "vector_category",
                "vector_feature",
                "vector_scenario",
            },
            scenario_hypotheses=(
                ScenarioHypothesis(
                    scenario_query="weather resistant practical carrying",
                    basis="commuting",
                    confidence=0.75,
                ),
            ),
        )

        self.assertEqual(
            model.queries,
            [
                "men jackets",
                "waterproof",
                "weather resistant practical carrying",
            ],
        )
        self.assertEqual(
            set(routes),
            {"vector_category", "vector_feature", "vector_scenario_1"},
        )
