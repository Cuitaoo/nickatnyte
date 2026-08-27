from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from starter.retrieval import CatalogRetriever
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

        results = self.retriever.search(state, "black winter footwear", 2)

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
        )

        self.assertIn("HIKING_PACK", results)
        self.assertLess(results.index("HIKING_PACK"), results.index("COTTON_SHIRT"))

    def test_removed_value_penalizes_matching_product(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="boots",
            removed_preferences={"material": ("leather",)},
        )

        results = self.retriever.search(state, "black winter boots", 2)

        self.assertEqual(results[0], "SYNTH_BOOT")

    def test_results_are_unique_catalog_ids_and_respect_top_k(self) -> None:
        results = self.retriever.search(
            ShoppingState.new("s", {}), "black boots winter leather", 2
        )

        self.assertEqual(len(results), len(set(results)))
        self.assertLessEqual(len(results), 2)
        self.assertTrue(set(results) <= self.catalog_ids)

    def test_empty_query_has_deterministic_catalog_fallback(self) -> None:
        results = self.retriever.search(ShoppingState.new("s", {}), "the and", 3)

        self.assertEqual(results, ["SYNTH_BOOT", "LEATHER_BOOT", "HIKING_PACK"])


if __name__ == "__main__":
    unittest.main()
