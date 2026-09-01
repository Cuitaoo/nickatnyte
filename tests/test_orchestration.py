import unittest
from dataclasses import dataclass

from starter.orchestration import (
    ACTION_ASK_ONLY,
    ACTION_EXHAUSTED,
    ACTION_RECOMMEND,
    ACTION_RECOMMEND_ASKING,
    build_decision,
    classify_action,
    constraint_split,
    removed_split,
    route_summary,
    top_margin,
)


@dataclass(frozen=True)
class _Cand:
    product_id: str
    score: float
    route_ranks: tuple = ()


class _State:
    def __init__(self, **kw):
        self.intent_mode = kw.get("intent_mode", "unknown")
        self.preferences = kw.get("preferences", {})
        self.removed_preferences = kw.get("removed_preferences", {})
        self.user_profile = kw.get("user_profile", {})


class ClassifyActionTest(unittest.TestCase):
    def test_recommend_now(self):
        self.assertEqual(classify_action(5, None, False), ACTION_RECOMMEND)

    def test_recommend_while_asking(self):
        self.assertEqual(classify_action(5, "color", False), ACTION_RECOMMEND_ASKING)

    def test_ask_only_when_deferred(self):
        self.assertEqual(classify_action(5, "color", True), ACTION_ASK_ONLY)

    def test_ask_only_when_nothing_returned_but_asking(self):
        self.assertEqual(classify_action(0, "color", False), ACTION_ASK_ONLY)

    def test_exhausted(self):
        self.assertEqual(classify_action(0, None, False), ACTION_EXHAUSTED)


class ConstraintSplitTest(unittest.TestCase):
    def test_stated_preferences_are_hard_profile_tags_are_soft(self):
        state = _State(
            preferences={"material": ("cotton",), "color": ("black", "navy")},
            user_profile={"preference_tags": ["comfort", "fit"]},
        )
        hard, soft = constraint_split(state)
        self.assertEqual(hard, ("color=black/navy", "material=cotton"))
        self.assertEqual(soft, ("comfort", "fit"))

    def test_empty_values_are_dropped(self):
        hard, _ = constraint_split(_State(preferences={"color": ()}))
        self.assertEqual(hard, ())

    def test_removed_constraints_are_reported(self):
        state = _State(removed_preferences={"material": ("wool",)})
        self.assertEqual(removed_split(state), ("material=wool",))


class RouteSummaryTest(unittest.TestCase):
    def test_collects_unique_sorted_routes(self):
        cands = (
            _Cand("a", 1.0, (("category", 1), ("attribute", 3))),
            _Cand("b", 0.9, (("category", 2),)),
        )
        routes, vector = route_summary(cands)
        self.assertEqual(routes, ("attribute", "category"))
        self.assertFalse(vector)

    def test_detects_vector_participation(self):
        cands = (_Cand("a", 1.0, (("vector_feature", 1),)),)
        routes, vector = route_summary(cands)
        self.assertTrue(vector)
        self.assertEqual(routes, ("vector_feature",))

    def test_empty(self):
        self.assertEqual(route_summary(()), ((), False))


class TopMarginTest(unittest.TestCase):
    def test_gap_between_top_two(self):
        self.assertAlmostEqual(
            top_margin((_Cand("a", 2.5), _Cand("b", 2.0), _Cand("c", 1.0))), 0.5
        )

    def test_single_or_empty_has_no_margin(self):
        self.assertEqual(top_margin((_Cand("a", 2.5),)), 0.0)
        self.assertEqual(top_margin(()), 0.0)


class BuildDecisionTest(unittest.TestCase):
    def _decision(self, **over):
        args = dict(
            state=_State(
                intent_mode="buying",
                preferences={"material": ("cotton",)},
                user_profile={"preference_tags": ["comfort"]},
            ),
            turn=3,
            track_name="buying",
            candidates=(
                _Cand("a", 2.5, (("category", 1),)),
                _Cand("b", 2.0, (("vector_feature", 1),)),
            ),
            returned_count=2,
            ask_attribute="color",
            deferred=False,
            depth_cap=5,
        )
        args.update(over)
        return build_decision(**args)

    def test_captures_the_whole_turn(self):
        d = self._decision()
        self.assertEqual(d.turn, 3)
        self.assertEqual(d.intent_route, "buying")
        self.assertEqual(d.track, "buying")
        self.assertEqual(d.action, ACTION_RECOMMEND_ASKING)
        self.assertEqual(d.hard_constraints, ("material=cotton",))
        self.assertEqual(d.soft_constraints, ("comfort",))
        self.assertEqual(d.routes_enabled, ("category", "vector_feature"))
        self.assertTrue(d.vector_used)
        self.assertEqual(d.candidate_count, 2)
        self.assertAlmostEqual(d.top_margin, 0.5)
        self.assertEqual(d.depth_cap, 5)
        self.assertEqual(d.next_question, "color")

    def test_deferral_is_recorded_as_ask_only(self):
        d = self._decision(deferred=True, returned_count=0)
        self.assertTrue(d.deferred)
        self.assertEqual(d.action, ACTION_ASK_ONLY)

    def test_describe_is_a_single_readable_line(self):
        text = self._decision().describe()
        self.assertIn("turn 3", text)
        self.assertIn("buying/buying", text)
        self.assertIn("material=cotton", text)
        self.assertIn("ask=color", text)
        self.assertNotIn("\n", text)

    def test_serializes_to_a_plain_dict(self):
        payload = self._decision().to_dict()
        self.assertEqual(payload["intent_route"], "buying")
        self.assertEqual(payload["next_question"], "color")


if __name__ == "__main__":
    unittest.main()
