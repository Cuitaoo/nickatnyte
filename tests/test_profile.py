import os
import unittest

from starter.profile import (
    PROFILE_EXPANSIONS,
    match_profile,
    max_boost,
    profile_tags,
)


class ProfileTagsTest(unittest.TestCase):
    def test_normalizes_and_deduplicates(self):
        got = profile_tags({"preference_tags": ["Comfort", "comfort", " FIT "]})
        self.assertEqual(got, ("comfort", "fit"))

    def test_missing_profile(self):
        self.assertEqual(profile_tags(None), ())
        self.assertEqual(profile_tags({}), ())


class ExpansionCoverageTest(unittest.TestCase):
    def test_every_observed_tag_has_an_entry(self):
        """The datasets draw from a closed vocabulary; all of it must be known."""
        observed = {
            "fit", "material", "comfort", "style", "durability",
            "performance", "warmth", "weather", "general shopping",
        }
        self.assertEqual(observed - set(PROFILE_EXPANSIONS), set())

    def test_general_shopping_expands_to_nothing(self):
        self.assertEqual(PROFILE_EXPANSIONS["general shopping"], frozenset())


class MaxBoostTest(unittest.TestCase):
    def tearDown(self):
        for n in ("TECHJAM_PROFILE_MAX_BUYING", "TECHJAM_PROFILE_MAX_BROWSING"):
            os.environ.pop(n, None)

    def test_buying_is_weakest_browsing_strongest(self):
        self.assertLess(max_boost("buying"), max_boost("unknown"))
        self.assertLess(max_boost("unknown"), max_boost("browsing"))

    def test_never_rivals_a_stated_preference(self):
        """confirmed_attribute_boost is ~1.30; the profile must stay far below."""
        for mode in ("buying", "browsing", "unknown"):
            self.assertLess(max_boost(mode), 0.5)

    def test_env_overrides(self):
        os.environ["TECHJAM_PROFILE_MAX_BROWSING"] = "0.25"
        self.assertAlmostEqual(max_boost("browsing"), 0.25)


class MatchProfileTest(unittest.TestCase):
    def test_expanded_vocabulary_matches_where_the_literal_tag_does_not(self):
        """The whole point: 'comfort' should match 'cushioned'."""
        corpus = "mens running shoe with cushioned insole and breathable mesh"
        boost, tags = match_profile(corpus, ("comfort",), "browsing")
        self.assertGreater(boost, 0.0)
        self.assertEqual(tags, ("comfort",))

    def test_literal_tag_absent_from_corpus_still_scores_zero_when_unrelated(self):
        boost, tags = match_profile("plain steel watch band", ("warmth",), "browsing")
        self.assertEqual(boost, 0.0)
        self.assertEqual(tags, ())

    def test_multiple_tags_accumulate(self):
        corpus = "durable reinforced canvas jacket, insulated fleece lining"
        one, _ = match_profile(corpus, ("durability",), "browsing")
        two, tags = match_profile(corpus, ("durability", "warmth"), "browsing")
        self.assertGreater(two, one)
        self.assertEqual(set(tags), {"durability", "warmth"})

    def test_ceiling_applies(self):
        corpus = " ".join(
            term for expansion in PROFILE_EXPANSIONS.values() for term in expansion
        )
        all_tags = tuple(PROFILE_EXPANSIONS)
        boost, _ = match_profile(corpus, all_tags, "buying")
        self.assertLessEqual(boost, max_boost("buying"))

    def test_buying_is_capped_below_browsing_for_the_same_product(self):
        corpus = "cushioned breathable relaxed cotton, durable reinforced"
        buying, _ = match_profile(corpus, ("comfort", "durability"), "buying")
        browsing, _ = match_profile(corpus, ("comfort", "durability"), "browsing")
        self.assertLess(buying, browsing)

    def test_general_shopping_never_matches(self):
        boost, tags = match_profile("anything at all", ("general shopping",), "browsing")
        self.assertEqual(boost, 0.0)
        self.assertEqual(tags, ())

    def test_empty_inputs(self):
        self.assertEqual(match_profile("", ("comfort",), "browsing"), (0.0, ()))
        self.assertEqual(match_profile("cushioned", (), "browsing"), (0.0, ()))

    def test_unknown_tag_is_ignored_not_crashed(self):
        boost, tags = match_profile("cushioned sole", ("nonsense",), "browsing")
        self.assertEqual(boost, 0.0)
        self.assertEqual(tags, ())

    def test_matched_tags_are_reported_for_explainability(self):
        corpus = "relaxed fit cotton tee, soft and breathable"
        _, tags = match_profile(corpus, ("fit", "comfort", "weather"), "browsing")
        self.assertIn("fit", tags)
        self.assertIn("comfort", tags)
        self.assertNotIn("weather", tags)


if __name__ == "__main__":
    unittest.main()
