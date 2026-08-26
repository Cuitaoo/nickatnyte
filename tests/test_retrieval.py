from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from starter.retrieval import CatalogRetriever, SearchResult
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


if __name__ == "__main__":
    unittest.main()
