import unittest

from starter.audience import (
    UNKNOWN,
    apply_audience_guardrail,
    audience_penalty,
    normalize_audience,
    product_audience,
    requested_audience,
)


class _State:
    def __init__(self, category=None, preferences=None):
        self.category = category
        self.preferences = preferences or {}


class NormalizeAudienceTest(unittest.TestCase):
    def test_catalog_case_variants_collapse(self):
        for raw in ("Womens", "womens", "WOMENS", " Womens "):
            self.assertEqual(normalize_audience(raw), ("f", "adult"))
        for raw in ("Mens", "mens"):
            self.assertEqual(normalize_audience(raw), ("m", "adult"))

    def test_hyphenated_departments(self):
        self.assertEqual(normalize_audience("unisex-adult"), ("n", "adult"))
        self.assertEqual(normalize_audience("unisex-child"), ("n", "child"))
        self.assertEqual(normalize_audience("baby-girls"), ("f", "baby"))

    def test_unrecognized_is_unknown(self):
        self.assertEqual(normalize_audience("novelty"), UNKNOWN)
        self.assertEqual(normalize_audience(""), UNKNOWN)
        self.assertEqual(normalize_audience(None), UNKNOWN)


class ProductAudienceTest(unittest.TestCase):
    def test_reads_department_from_flattened_details(self):
        meta = {"details": "product dimensions 1.97 inches department womens date first available"}
        self.assertEqual(product_audience(meta), ("f", "adult"))

    def test_falls_back_to_categories(self):
        meta = {"details": "item weight 2 pounds", "categories": "clothing, shoes & jewelry men clothing jeans"}
        self.assertEqual(product_audience(meta), ("m", "adult"))

    def test_falls_back_to_title_possessive(self):
        meta = {"details": "package dimensions 3 x 2", "categories": "clothing", "title": "levi's men's 505 regular fit jeans"}
        self.assertEqual(product_audience(meta), ("m", "adult"))

    def test_department_wins_over_title(self):
        meta = {
            "details": "department womens",
            "title": "boyfriend jeans men's cut inspired",
        }
        self.assertEqual(product_audience(meta), ("f", "adult"))

    def test_no_signal_is_unknown(self):
        self.assertEqual(product_audience({"details": "item weight 2 pounds"}), UNKNOWN)
        self.assertEqual(product_audience({}), UNKNOWN)


class RequestedAudienceTest(unittest.TestCase):
    def test_reads_state_category(self):
        self.assertEqual(requested_audience(_State("men jeans")), ("m", "adult"))

    def test_falls_back_to_latest_message(self):
        state = _State("jeans")
        self.assertEqual(requested_audience(state, "actually I need women's"), ("f", "adult"))

    def test_falls_back_to_preferences(self):
        state = _State("jeans", {"style": ("mens", "relaxed fit")})
        self.assertEqual(requested_audience(state), ("m", "adult"))

    def test_conflicting_audiences_are_unknown(self):
        self.assertEqual(requested_audience(_State("men and women jeans")), UNKNOWN)

    def test_no_audience_is_unknown(self):
        self.assertEqual(requested_audience(_State("jeans"), "something soft"), UNKNOWN)

    def test_style_names_are_not_audiences(self):
        """'boy shorts' is womenswear; reading it as boys' inverts the guardrail."""
        self.assertEqual(requested_audience(_State("panties boy shorts")), UNKNOWN)
        self.assertEqual(requested_audience(_State("boyshort briefs")), UNKNOWN)
        self.assertEqual(requested_audience(_State("mom jeans")), UNKNOWN)
        self.assertEqual(requested_audience(_State("baby doll dress")), UNKNOWN)

    def test_style_name_does_not_block_a_real_audience(self):
        self.assertEqual(
            requested_audience(_State("women panties boy shorts")), ("f", "adult")
        )


class AudiencePenaltyTest(unittest.TestCase):
    def test_same_audience_is_free(self):
        self.assertEqual(audience_penalty(("m", "adult"), ("m", "adult")), 0.0)

    def test_cross_gender_adult_is_full(self):
        self.assertEqual(audience_penalty(("m", "adult"), ("f", "adult")), 1.0)
        self.assertEqual(audience_penalty(("f", "adult"), ("m", "adult")), 1.0)

    def test_same_gender_different_age_is_half(self):
        self.assertEqual(audience_penalty(("m", "adult"), ("m", "child")), 0.5)
        self.assertEqual(audience_penalty(("f", "adult"), ("f", "child")), 0.5)

    def test_unisex_is_never_penalized(self):
        self.assertEqual(audience_penalty(("m", "adult"), ("n", "adult")), 0.0)
        self.assertEqual(audience_penalty(("n", "adult"), ("f", "adult")), 0.0)

    def test_unknown_either_side_is_free(self):
        self.assertEqual(audience_penalty(UNKNOWN, ("f", "adult")), 0.0)
        self.assertEqual(audience_penalty(("m", "adult"), UNKNOWN), 0.0)

    def test_penalty_is_symmetric(self):
        pairs = [("m", "adult"), ("f", "adult"), ("m", "child"), ("f", "child"), ("n", "adult")]
        for left in pairs:
            for right in pairs:
                self.assertEqual(
                    audience_penalty(left, right),
                    audience_penalty(right, left),
                    f"{left} vs {right}",
                )


class ApplyGuardrailTest(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "W1": {"details": "department womens"},
            "W2": {"details": "department womens"},
            "M1": {"details": "department mens"},
            "B1": {"details": "department boys"},
            "U1": {"details": "department unisex-adult"},
            "X1": {"details": "item weight 2 pounds"},
        }

    def test_demotes_wrong_gender_below_match(self):
        ordered = ["W1", "W2", "M1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.5, top_n=20
        )
        self.assertEqual(result[0], "M1")

    def test_unknown_request_is_a_no_op(self):
        ordered = ["W1", "M1", "B1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("jeans"), penalty=0.5, top_n=20
        )
        self.assertEqual(result, ordered)

    def test_unisex_and_unknown_products_hold_position(self):
        ordered = ["U1", "X1", "W1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.5, top_n=20
        )
        self.assertEqual(result[:2], ["U1", "X1"])

    def test_boys_penalized_less_than_womens(self):
        ordered = ["W1", "B1", "M1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.4, top_n=20
        )
        self.assertLess(result.index("B1"), result.index("W1"))

    def test_tail_beyond_window_is_untouched(self):
        ordered = ["W1", "M1", "W2", "B1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.5, top_n=2
        )
        self.assertEqual(result[2:], ["W2", "B1"])

    def test_ordering_is_stable_within_equal_penalty(self):
        ordered = ["W1", "W2"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.5, top_n=20
        )
        self.assertEqual(result, ["W1", "W2"])

    def test_zero_penalty_disables(self):
        ordered = ["W1", "M1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.0, top_n=20
        )
        self.assertEqual(result, ordered)

    def test_empty_input(self):
        self.assertEqual(
            apply_audience_guardrail([], self.metadata, _State("men jeans")), []
        )

    def test_soft_not_a_filter(self):
        """A wrong-audience product far ahead still survives a small penalty."""
        ordered = ["W1"] + [f"pad{i}" for i in range(18)] + ["M1"]
        result = apply_audience_guardrail(
            ordered, self.metadata, _State("men jeans"), penalty=0.15, top_n=20
        )
        self.assertIn("W1", result)
        self.assertLess(result.index("W1"), 10)


if __name__ == "__main__":
    unittest.main()
