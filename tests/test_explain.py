import unittest
from dataclasses import dataclass, field

from starter.explain import NO_EVIDENCE, evidence_for, explain, explain_top


@dataclass(frozen=True)
class _Cand:
    product_id: str = "p1"
    matched_attributes: frozenset = frozenset()
    matched_profile_tags: tuple = ()
    score_components: tuple = ()


class _State:
    def __init__(self, category=None, preferences=None):
        self.category = category
        self.preferences = preferences or {}


class EvidenceTest(unittest.TestCase):
    def test_identifier_leads(self):
        c = _Cand(
            matched_attributes=frozenset({"material"}),
            score_components=(("exact_identifier", 6.0), ("category", 1.0)),
        )
        state = _State("men jeans", {"material": ("cotton",)})
        self.assertEqual(evidence_for(c, state)[0], "the exact model number you gave")

    def test_category_only_when_it_actually_scored(self):
        state = _State("men jeans", {})
        scored = _Cand(score_components=(("category", 1.0),))
        self.assertIn("men jeans", evidence_for(scored, state))
        unscored = _Cand(score_components=(("fusion", 1.0),))
        self.assertNotIn("men jeans", evidence_for(unscored, state))

    def test_only_matched_attributes_are_claimed(self):
        """Never claim a preference the product did not actually match."""
        c = _Cand(matched_attributes=frozenset({"material"}))
        state = _State(preferences={"material": ("cotton",), "color": ("black",)})
        got = evidence_for(c, state)
        self.assertIn("cotton", got)
        self.assertNotIn("black", got)

    def test_zero_component_is_not_evidence(self):
        c = _Cand(score_components=(("exact_identifier", 0.0),))
        self.assertEqual(evidence_for(c, _State()), [])

    def test_no_candidate(self):
        self.assertEqual(evidence_for(None, _State()), [])


class ExplainTest(unittest.TestCase):
    def test_single_reason(self):
        c = _Cand(matched_attributes=frozenset({"material"}))
        state = _State(preferences={"material": ("cotton",)})
        self.assertEqual(explain(c, state), "Recommended because it matches cotton.")

    def test_two_reasons_use_and(self):
        c = _Cand(matched_attributes=frozenset({"material", "style"}))
        state = _State(preferences={"material": ("cotton",), "style": ("relaxed fit",)})
        self.assertEqual(
            explain(c, state), "Recommended because it matches cotton and relaxed fit."
        )

    def test_three_reasons_use_oxford_comma(self):
        c = _Cand(
            matched_attributes=frozenset({"material", "style", "color"}),
            score_components=(("category", 1.0),),
        )
        state = _State(
            "jeans",
            {"material": ("cotton",), "style": ("relaxed",), "color": ("black",)},
        )
        self.assertIn(", ", explain(c, state))
        self.assertIn(" and ", explain(c, state))

    def test_profile_is_appended_never_leading(self):
        c = _Cand(
            matched_attributes=frozenset({"material"}),
            matched_profile_tags=("comfort",),
        )
        state = _State(preferences={"material": ("cotton",)})
        got = explain(c, state)
        self.assertTrue(got.startswith("Recommended because it matches cotton"))
        self.assertIn("your profile's comfort preference", got)

    def test_profile_alone_still_reads_correctly(self):
        c = _Cand(matched_profile_tags=("comfort", "fit"))
        got = explain(c, _State())
        self.assertEqual(
            got, "Recommended because it matches your profile's comfort and fit preference."
        )

    def test_no_evidence_falls_back(self):
        self.assertEqual(explain(_Cand(), _State()), NO_EVIDENCE)

    def test_never_mentions_scores(self):
        c = _Cand(
            matched_attributes=frozenset({"material"}),
            score_components=(("fusion", 2.47), ("rating", 0.008)),
        )
        got = explain(c, _State(preferences={"material": ("cotton",)}))
        self.assertNotIn("2.47", got)
        self.assertNotIn("fusion", got)


class ExplainTopTest(unittest.TestCase):
    def test_explains_the_leader_not_another_candidate(self):
        lead = _Cand("A", matched_attributes=frozenset({"material"}))
        other = _Cand("B", matched_attributes=frozenset({"color"}))
        state = _State(preferences={"material": ("cotton",), "color": ("black",)})
        got = explain_top(["A", "B"], (other, lead), state, "fallback")
        self.assertIn("cotton", got)
        self.assertNotIn("black", got)

    def test_falls_back_when_nothing_returned(self):
        self.assertEqual(explain_top([], (), _State(), "fallback"), "fallback")

    def test_falls_back_when_leader_has_no_evidence(self):
        self.assertEqual(
            explain_top(["A"], (_Cand("A"),), _State(), "fallback"), "fallback"
        )

    def test_falls_back_when_leader_is_missing_from_candidates(self):
        self.assertEqual(explain_top(["Z"], (_Cand("A"),), _State(), "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
