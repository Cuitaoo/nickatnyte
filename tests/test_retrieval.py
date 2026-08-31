from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from starter.retrieval import (
    EXACT_IDENTIFIER_BOOST,
    MAX_PROFILE_BOOST,
    CatalogRetriever,
    RankedCandidate,
    RetrievalWeights,
    SearchResult,
    _attribute_signature,
    _exact_identifiers,
    _signature_disagreement,
    select_diverse_recommendations,
)
from starter.query_expansion import ScenarioHypothesis
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

    def test_model_shorthand_gets_exact_identifier_boost(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "MODEL_TARGET",
                "title": "Bestform Women's Wire Free Bra",
                "categories": ["Women", "Clothing", "Bras", "Everyday Bras"],
                "features": ["Hand Wash Only"],
                "details": {"Item model number": "5006715"},
                "description": ["100% Cotton cups. Colors: White and Black."],
                "store": "Bestform",
                "price": 14.98,
                "average_rating": 4.2,
                "rating_number": 96,
            },
            {
                "parent_asin": "GENERIC_BRA",
                "title": "Hanes 100% Cotton White Everyday Bra",
                "categories": ["Women", "Clothing", "Bras", "Everyday Bras"],
                "features": ["Hand Wash Only"],
                "details": {"Item model number": "ABC123"},
                "description": ["Cotton white bra."],
                "store": "Hanes",
                "price": 12.0,
                "average_rating": 4.8,
                "rating_number": 1000,
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        retriever = CatalogRetriever(catalog_path)
        self.addCleanup(retriever.close)
        state = replace(
            ShoppingState.new("s", {}),
            category="bras everyday bras",
            preferences={
                "material": ("cotton",),
                "color": ("white",),
                "feature": ("hand wash only", "item model number: 5006715"),
            },
        )

        result = retriever.search(
            state,
            "Please find model 5006715.",
            2,
        )

        self.assertEqual(result.recommendations[0], "MODEL_TARGET")
        target = next(
            item for item in result.candidates if item.product_id == "MODEL_TARGET"
        )
        self.assertEqual(
            dict(target.score_components)["exact_identifier"], EXACT_IDENTIFIER_BOOST
        )

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

    def test_override_message_does_not_weaken_updated_preferences(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="boots",
            preferences={"material": ("leather",)},
        )

        result = self.retriever.search(
            state,
            "Actually, ignore my earlier preference. What I need is leather.",
            3,
        )
        target = next(
            item for item in result.candidates if item.product_id == "LEATHER_BOOT"
        )

        self.assertEqual(
            dict(target.score_components)["preference:material"],
            RetrievalWeights().confirmed_attribute_boost,
        )

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

    def test_vector_route_can_rescue_candidate_when_enabled(self) -> None:
        class FakeVectorIndex:
            def search_routes_with_scores(
                self, state, latest_message, top_k, allowed_routes=None
            ):
                assert allowed_routes is not None
                assert "vector_feature" in allowed_routes
                return {"vector_feature": [("RAIN_RUNNER", 0.9)]}

        self.retriever.vector_index = FakeVectorIndex()
        self.retriever.vector_weight = 1.0
        self.retriever.vector_feature_weight = 1.0
        self.retriever.vector_recall_only = False
        self.retriever.vector_max_doc_frequency = 50_000
        self.addCleanup(setattr, self.retriever, "vector_index", None)
        self.addCleanup(setattr, self.retriever, "vector_weight", 0.0)
        self.addCleanup(setattr, self.retriever, "vector_feature_weight", 0.05)
        self.addCleanup(setattr, self.retriever, "vector_recall_only", True)
        self.addCleanup(setattr, self.retriever, "vector_max_doc_frequency", 750)

        state = replace(
            ShoppingState.new("s", {}),
            preferences={"feature": ("marathon",)},
        )
        result = self.retriever.search(state, "monsoon sprint", 3)

        self.assertIn("RAIN_RUNNER", result.recommendations)
        candidate = next(
            item for item in result.candidates if item.product_id == "RAIN_RUNNER"
        )
        self.assertIn(
            "vector_feature", [route_name for route_name, _rank in candidate.route_ranks]
        )

    def test_scenario_route_adds_candidate_with_small_rrf_signal(self) -> None:
        class FakeVectorIndex:
            def search_routes_with_scores(
                self,
                state,
                latest_message,
                top_k,
                allowed_routes=None,
                scenario_hypotheses=(),
            ):
                assert allowed_routes is not None
                assert "vector_scenario" in allowed_routes
                assert (
                    scenario_hypotheses[0].scenario_query
                    == "waterproof marathon trainer"
                )
                return {"vector_scenario_1": [("RAIN_RUNNER", 0.9)]}

        self.retriever.vector_index = FakeVectorIndex()
        self.retriever.vector_policy = "always"
        self.retriever.vector_recall_only = True
        self.addCleanup(setattr, self.retriever, "vector_index", None)
        self.addCleanup(setattr, self.retriever, "vector_policy", "adaptive")
        self.addCleanup(setattr, self.retriever, "vector_recall_only", True)

        result = self.retriever.search(
            replace(ShoppingState.new("s", {}), category="boots"),
            "black boots",
            10,
            scenario_hypotheses=(
                ScenarioHypothesis(
                    scenario_query="waterproof marathon trainer",
                    basis="boots",
                    confidence=0.75,
                ),
            ),
        )

        candidate = next(
            item for item in result.candidates if item.product_id == "RAIN_RUNNER"
        )
        self.assertIn("vector_scenario_1", dict(candidate.route_ranks))
        self.assertAlmostEqual(
            dict(candidate.score_components)["fusion"],
            self.retriever.scenario_vector_weight
            / (self.retriever.weights.rrf_offset + 1),
        )

    def test_low_similarity_vector_candidate_is_filtered(self) -> None:
        class FakeVectorIndex:
            def search_routes_with_scores(
                self, state, latest_message, top_k, allowed_routes=None
            ):
                return {"vector_feature": [("RAIN_RUNNER", 0.1)]}

        self.retriever.vector_index = FakeVectorIndex()
        self.retriever.vector_feature_weight = 1.0
        self.retriever.vector_recall_only = False
        self.retriever.vector_policy = "always"
        self.retriever.vector_max_doc_frequency = 50_000
        self.retriever.vector_min_similarity = 0.5
        self.addCleanup(setattr, self.retriever, "vector_index", None)
        self.addCleanup(setattr, self.retriever, "vector_feature_weight", 0.05)
        self.addCleanup(setattr, self.retriever, "vector_recall_only", True)
        self.addCleanup(setattr, self.retriever, "vector_policy", "adaptive")
        self.addCleanup(setattr, self.retriever, "vector_max_doc_frequency", 750)
        self.addCleanup(setattr, self.retriever, "vector_min_similarity", 0.45)

        state = ShoppingState.new("s", {})
        result = self.retriever.search(state, "monsoon sprint", 3)

        self.assertNotIn("RAIN_RUNNER", result.recommendations)

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

    def test_signature_disagreement_tracks_distribution_not_unique_count(self) -> None:
        self.assertEqual(_signature_disagreement(["leather"] * 10), 0.0)
        self.assertAlmostEqual(
            _signature_disagreement(["leather"] * 5 + ["synthetic"] * 5),
            0.5,
        )
        self.assertGreater(
            _signature_disagreement(["leather"] * 5 + ["synthetic"] * 5),
            _signature_disagreement(["leather"] * 9 + ["synthetic"]),
        )

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
        weights = RetrievalWeights()
        weakest_route_weight = min(
            weights.route_category,
            weights.route_feature_use_case,
            weights.route_exact_phrase,
            weights.route_attribute,
            weights.route_relaxed,
            weights.route_latest,
            weights.route_latest_override,
            weights.route_synonym,
        )
        weakest_single_route_hit = weakest_route_weight / (weights.rrf_offset + 1)

        self.assertLess(MAX_PROFILE_BOOST, weakest_single_route_hit)


COMPOUND_PRODUCTS = [
    {
        "parent_asin": "COMPOUND_SWEATSHIRT",
        "title": "Loose Cotton Sweatshirt Top",
        "categories": ["Women", "Clothing", "Sweatshirts"],
        "features": ["pull on closure"],
        "details": {"fabric": "90% Cotton, 10% Others"},
        "description": ["simple spring top"],
        "store": "Mordenmiss",
        "price": 30.0,
        "average_rating": 4.2,
        "rating_number": 200,
    },
    {
        "parent_asin": "PLAIN_SWEATSHIRT",
        "title": "Loose Cotton Sweatshirt Top",
        "categories": ["Women", "Clothing", "Sweatshirts"],
        "features": ["pull on closure"],
        "details": {"fabric": "cotton blend"},
        "description": ["simple spring top"],
        "store": "Basics",
        "price": 30.0,
        "average_rating": 4.6,
        "rating_number": 900,
    },
]


class CompoundPhraseRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        catalog_path = Path(cls._directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in COMPOUND_PRODUCTS),
            encoding="utf-8",
        )
        cls.retriever = CatalogRetriever(catalog_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.retriever.close()
        cls._directory.cleanup()

    def test_compound_search_term_prefers_exact_composition(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            search_terms=("90% cotton, 10% others",),
        )

        results = self.retriever.search(state, "cotton", 2).recommendations

        self.assertEqual(results[0], "COMPOUND_SWEATSHIRT")


class SynonymRouteTest(unittest.TestCase):
    def test_category_synonym_reaches_matching_product(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "HOODIE_TOP",
                "title": "Fleece Hoodie Pullover",
                "categories": ["Women", "Clothing", "Fashion Hoodies"],
                "features": ["kangaroo pocket"],
                "details": {"fabric": "fleece"},
                "description": ["cozy layer"],
                "store": "CozyCo",
                "price": 30.0,
                "average_rating": 4.5,
                "rating_number": 300,
            },
            {
                "parent_asin": "DENIM_PANTS",
                "title": "Classic Denim Pants",
                "categories": ["Women", "Clothing", "Jeans"],
                "features": ["five pockets"],
                "details": {"fabric": "denim"},
                "description": ["everyday jeans"],
                "store": "JeanCo",
                "price": 40.0,
                "average_rating": 4.4,
                "rating_number": 500,
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        retriever = CatalogRetriever(catalog_path)
        self.addCleanup(retriever.close)
        state = replace(ShoppingState.new("s", {}), category="sweatshirt")

        result = retriever.search(state, "something comfy", 2)

        hoodie = next(
            (
                candidate
                for candidate in result.candidates
                if candidate.product_id == "HOODIE_TOP"
            ),
            None,
        )
        self.assertIsNotNone(hoodie)
        self.assertIn(
            "synonym", [route_name for route_name, _rank in hoodie.route_ranks]
        )

        direct_state = replace(ShoppingState.new("s2", {}), category="hoodie")
        direct = retriever.search(direct_state, "something comfy", 2)
        for candidate in direct.candidates:
            self.assertNotIn(
                "synonym",
                [route_name for route_name, _rank in candidate.route_ranks],
            )


class RetrievalWeightsTest(unittest.TestCase):
    def test_default_weights_match_recorded_tuning_result(self) -> None:
        report_path = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "evaluations"
            / "weight-tuning.json"
        )
        best = json.loads(report_path.read_text(encoding="utf-8"))["best"]["weights"]
        weights = RetrievalWeights()
        for name, value in best.items():
            self.assertEqual(getattr(weights, name), value)

    def test_custom_weights_change_ranking(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )
        state = replace(
            ShoppingState.new("s", {}),
            category="boots",
            preferences={"material": ("leather",)},
        )

        default_retriever = CatalogRetriever(catalog_path)
        self.addCleanup(default_retriever.close)
        default_first = default_retriever.search(
            state, "black winter footwear", 2
        ).recommendations[0]
        self.assertEqual(default_first, "LEATHER_BOOT")

        rating_led = CatalogRetriever(
            catalog_path,
            weights=RetrievalWeights(
                confirmed_attribute_boost=0.0, category_boost=0.0, rating_coef=50.0
            ),
        )
        self.addCleanup(rating_led.close)
        rating_led_first = rating_led.search(
            state, "black winter footwear", 2
        ).recommendations[0]
        self.assertEqual(rating_led_first, "SYNTH_BOOT")


class BudgetMatchTest(unittest.TestCase):
    def test_exact_identifier_extraction_accepts_labeled_and_model_shorthand(self) -> None:
        self.assertEqual(
            _exact_identifiers(["Item model number: 5006715."]),
            ("5006715",),
        )
        self.assertEqual(
            _exact_identifiers(["Please find model AB-1234 for me."]),
            ("ab 1234",),
        )
        self.assertEqual(
            _exact_identifiers(
                [
                    "budget around $59.99",
                    "Date First Available: 2021",
                    "show me a model shirt",
                    "this model is comfortable",
                ]
            ),
            (),
        )

    def test_material_color_and_size_match_description(self) -> None:
        product = {
            "title": "",
            "details": "",
            "features": "",
            "description": "100% Cotton cups. Colors: White and Black. Sizes: B 36-42.",
            "price": None,
        }

        self.assertTrue(
            CatalogRetriever._preference_matches("material", "cotton", product)
        )
        self.assertTrue(CatalogRetriever._preference_matches("color", "white", product))
        self.assertTrue(CatalogRetriever._preference_matches("size", "36", product))

    def test_around_budget_matches_price_band(self) -> None:
        product = {"price": 55.0}
        self.assertTrue(
            CatalogRetriever._preference_matches("budget", "around $60", product)
        )
        self.assertTrue(
            CatalogRetriever._preference_matches("budget", "budget around $60", product)
        )

    def test_around_budget_rejects_far_price(self) -> None:
        self.assertFalse(
            CatalogRetriever._preference_matches("budget", "around $60", {"price": 200.0})
        )
        self.assertFalse(
            CatalogRetriever._preference_matches("budget", "around $60", {"price": 10.0})
        )

    def test_under_budget_still_matches(self) -> None:
        self.assertTrue(
            CatalogRetriever._preference_matches("budget", "under $60", {"price": 55.0})
        )


if __name__ == "__main__":
    unittest.main()
