from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from starter.retrieval import (
    MAX_PROFILE_BOOST,
    RRF_OFFSET,
    CatalogRetriever,
    RankedCandidate,
    SearchResult,
    _attribute_signature,
    select_diverse_recommendations,
)
from starter.state import ShoppingState


PRODUCTS = [
    {
        "parent_asin": "LEATHER_BOOT",
        "title": "Black Leather Winter Hiking Boot",
        "categories": ["Women", "Shoes", "Boots"],
        "features": ["waterproof full grain leather", "warm lining"],
        "details": {"color": "black", "size": "8"},
        "description": ["trail ready winter footwear"],
        "store": "TrailCo",
        "price": 89.0,
        "average_rating": 4.6,
        "rating_number": 800,
    },
    {
        "parent_asin": "SYNTH_BOOT",
        "title": "Black Synthetic Fashion Boot",
        "categories": ["Women", "Shoes", "Boots"],
        "features": ["synthetic upper", "side zipper"],
        "details": {"color": "black", "size": "8"},
        "description": ["winter fashion footwear"],
        "store": "CityStyle",
        "price": 59.0,
        "average_rating": 4.8,
        "rating_number": 1200,
    },
    {
        "parent_asin": "HIKING_PACK",
        "title": "Waterproof Hiking Backpack",
        "categories": ["Outdoor", "Backpacks"],
        "features": ["lightweight trail pack", "waterproof shell"],
        "details": {"color": "green", "capacity": "30L"},
        "description": ["day hiking backpack"],
        "store": "TrailCo",
        "price": 49.0,
        "average_rating": 4.5,
        "rating_number": 400,
    },
    {
        "parent_asin": "RAIN_RUNNER",
        "title": "All Weather Performance Trainer",
        "categories": ["Women", "Shoes", "Athletic"],
        "features": ["waterproof membrane", "marathon cushioning"],
        "details": {"color": "blue", "size": "8"},
        "description": ["long distance road running"],
        "store": "EnduranceCo",
        "price": 109.0,
        "average_rating": 4.4,
        "rating_number": 90,
    },
    {
        "parent_asin": "COMFORT_SHOE",
        "title": "Gray Everyday Walking Shoe",
        "categories": ["Women", "Shoes", "Walking"],
        "features": ["comfort foam footbed"],
        "details": {"color": "gray", "size": "8"},
        "description": ["casual walking sneaker"],
        "store": "DailyCo",
        "price": 64.0,
        "average_rating": 4.3,
        "rating_number": 150,
    },
    {
        "parent_asin": "EXACT_BLUE_SHOE",
        "title": "Blue Everyday Walking Shoe",
        "categories": ["Women", "Shoes", "Walking"],
        "features": ["foam footbed"],
        "details": {"color": "blue", "size": "8"},
        "description": ["casual walking sneaker"],
        "store": "DailyCo",
        "price": 64.0,
        "average_rating": 4.3,
        "rating_number": 150,
    },
    {
        "parent_asin": "COTTON_SHIRT",
        "title": "Red Cotton Casual Shirt",
        "categories": ["Men", "Clothing", "Shirts"],
        "features": ["soft cotton"],
        "details": {"color": "red", "size": "large"},
        "description": ["everyday crew shirt"],
        "store": "Basics",
        "price": 25.0,
        "average_rating": 4.1,
        "rating_number": 100,
    },
]


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        cls.retriever = CatalogRetriever(cls.catalog_path)
        cls.catalog_ids = {product["parent_asin"] for product in PRODUCTS}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.retriever.close()
        cls._directory.cleanup()

    def test_active_preferences_boost_matching_product(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="boots",
            preferences={"material": ("leather",)},
        )

        results = self.retriever.search(state, "black winter footwear", 2).recommendations

        self.assertEqual(results[0], "LEATHER_BOOT")

    def test_latest_message_route_can_surface_new_category(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shirts",
            preferences={"material": ("cotton",)},
            search_terms=("casual",),
        )

        results = self.retriever.search(
            state, "Actually I need a waterproof hiking backpack", 3
        ).recommendations

        self.assertIn("HIKING_PACK", results)
        self.assertLess(results.index("HIKING_PACK"), results.index("COTTON_SHIRT"))

    def test_removed_value_penalizes_matching_product(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="boots",
            removed_preferences={"material": ("leather",)},
        )

        results = self.retriever.search(state, "black winter boots", 2).recommendations

        self.assertEqual(results[0], "SYNTH_BOOT")

    def test_results_are_unique_catalog_ids_and_respect_top_k(self) -> None:
        results = self.retriever.search(
            ShoppingState.new("s", {}), "black boots winter leather", 2
        ).recommendations

        self.assertEqual(len(results), len(set(results)))
        self.assertLessEqual(len(results), 2)
        self.assertTrue(set(results) <= self.catalog_ids)

    def test_empty_query_has_deterministic_catalog_fallback(self) -> None:
        results = self.retriever.search(ShoppingState.new("s", {}), "the and", 3)

        self.assertEqual(
            results.recommendations, ("SYNTH_BOOT", "LEATHER_BOOT", "HIKING_PACK")
        )

    def test_search_returns_structured_route_evidence(self) -> None:
        result = self.retriever.search(
            ShoppingState.new("s", {}),
            "waterproof marathon cushioning",
            3,
        )

        self.assertIsInstance(result, SearchResult)
        self.assertIn("RAIN_RUNNER", result.recommendations)
        candidate = next(
            item for item in result.candidates if item.product_id == "RAIN_RUNNER"
        )
        route_names = {name for name, _ in candidate.route_ranks}
        self.assertIn("feature_use_case", route_names)
        self.assertIn("exact_phrase", route_names)

    def test_routes_recover_candidates_from_different_evidence(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="athletic shoes",
            preferences={
                "feature": ("waterproof membrane",),
                "use_case": ("marathon",),
                "color": ("blue",),
            },
        )

        result = self.retriever.search(state, "blue shoes for rainy races", 4)
        evidence = {
            item.product_id: {name for name, _ in item.route_ranks}
            for item in result.candidates
        }

        self.assertIn("category", {route for routes in evidence.values() for route in routes})
        self.assertIn("attribute", {route for routes in evidence.values() for route in routes})
        self.assertIn("latest_message", {route for routes in evidence.values() for route in routes})


    def test_confirmed_preference_outweighs_profile_tendency(self) -> None:
        state = replace(
            ShoppingState.new(
                "s",
                {"summary": "always prioritizes comfort", "preference_tags": ["comfort"]},
            ),
            category="shoes",
            preferences={"color": ("blue",)},
        )

        result = self.retriever.search(state, "shoes", 4)

        self.assertLess(
            result.recommendations.index("EXACT_BLUE_SHOE"),
            result.recommendations.index("COMFORT_SHOE"),
        )

    def test_profile_breaks_a_near_tie_without_becoming_a_filter(self) -> None:
        plain_state = replace(ShoppingState.new("plain", {}), category="shoes")
        profile_state = replace(
            ShoppingState.new(
                "profile", {"summary": "comfort", "preference_tags": []}
            ),
            category="shoes",
        )

        plain = self.retriever.search(plain_state, "everyday walking shoes", 10)
        profiled = self.retriever.search(profile_state, "everyday walking shoes", 10)
        plain_scores = {item.product_id: item.score for item in plain.candidates}
        profiled_scores = {item.product_id: item.score for item in profiled.candidates}

        self.assertIn("COMFORT_SHOE", profiled.recommendations)
        profile_delta = profiled_scores["COMFORT_SHOE"] - plain_scores["COMFORT_SHOE"]
        self.assertGreater(profile_delta, 0.0)
        self.assertLessEqual(profile_delta, MAX_PROFILE_BOOST)

    def test_diagnostics_report_candidate_disagreement(self) -> None:
        result = self.retriever.search(
            replace(ShoppingState.new("s", {}), category="boots"),
            "winter boots",
            3,
        )

        self.assertGreater(result.diagnostics["material"].coverage, 0.0)
        self.assertGreater(result.diagnostics["material"].disagreement, 0.0)
        self.assertGreaterEqual(result.diagnostics["feature"].relevance, 0.0)
        self.assertLessEqual(result.diagnostics["feature"].relevance, 1.0)

    def test_budget_and_removed_values_change_order(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="boots",
            preferences={"budget": ("under $70",)},
            removed_preferences={"material": ("leather",)},
        )

        result = self.retriever.search(state, "black winter boots", 2)

        self.assertEqual(result.recommendations[0], "SYNTH_BOOT")
        winner = next(
            item for item in result.candidates if item.product_id == "SYNTH_BOOT"
        )
        self.assertIn("preference:budget", dict(winner.score_components))

    def test_top_six_are_strict_and_tail_can_cover_an_underrepresented_route(self) -> None:
        candidates = tuple(
            RankedCandidate(
                product_id=f"P{index}",
                score=10.0 - index * 0.1,
                route_ranks=(("category" if index < 9 else "feature_use_case", index + 1),),
            )
            for index in range(12)
        )

        selected = select_diverse_recommendations(candidates, 10)

        self.assertEqual(selected[:6], tuple(f"P{index}" for index in range(6)))
        self.assertIn("P9", selected[6:])

    def test_diversity_never_ejects_a_strict_top_ten_candidate(self) -> None:
        candidates = tuple(
            RankedCandidate(
                product_id=f"P{index}",
                score=10.0 - index * 0.1,
                route_ranks=(("category" if index < 10 else "feature_use_case", index + 1),),
            )
            for index in range(12)
        )

        selected = select_diverse_recommendations(candidates, 10)

        self.assertEqual(selected, tuple(f"P{index}" for index in range(10)))

    def test_late_route_can_compete_after_candidate_pool_reaches_cap(self) -> None:
        state = replace(ShoppingState.new("s", {}), category="boots")

        def route_results(route) -> list[str]:
            if route.name == "category":
                return ["SYNTH_BOOT", "COTTON_SHIRT"]
            if route.name == "latest_message":
                return ["LEATHER_BOOT"]
            return []

        with (
            patch("starter.retrieval.CANDIDATE_LIMIT", 2),
            patch.object(self.retriever, "_run_route", side_effect=route_results),
        ):
            result = self.retriever.search(state, "leather winter boot", 2)

        self.assertIn("LEATHER_BOOT", {item.product_id for item in result.candidates})

    def test_generic_feature_words_do_not_create_fake_disagreement(self) -> None:
        product = {
            "title": "everyday shoe",
            "categories": "women shoes",
            "features": "made in the usa or imported",
            "details": "",
            "store": "example",
            "description": "standard item",
            "corpus": "everyday shoe made in the usa or imported",
            "price": None,
        }

        self.assertEqual(_attribute_signature(product, "feature"), "")

    def test_profile_boost_stays_below_a_single_route_hit(self) -> None:
        """The profile is a tie-breaker, so it must never outweigh lexical evidence.

        Fusion contributes at most ``weight / (RRF_OFFSET + rank)`` per route. If
        the profile boost exceeds what one top-ranked route hit is worth, a
        product nothing matched can outrank the best lexical match, which is the
        opposite of a tie-breaker.
        """
        strongest_route_weight = 2.20
        best_single_route_hit = strongest_route_weight / (RRF_OFFSET + 1)

        self.assertLess(MAX_PROFILE_BOOST, best_single_route_hit)


if __name__ == "__main__":
    unittest.main()
