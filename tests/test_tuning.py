from __future__ import annotations

import unittest

from tools.tune_weights import sample_weights, stratified_split


SAMPLES = [
    {"sample_id": f"public_{index:04d}", "scenario_type": scenario}
    for index, scenario in enumerate(
        ["buying"] * 8 + ["browsing"] * 8 + ["boundary"] * 2 + ["intent_override"] * 2
    )
]


class StratifiedSplitTest(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        first_train, first_holdout = stratified_split(SAMPLES, holdout_fraction=0.25, seed=7)
        second_train, second_holdout = stratified_split(SAMPLES, holdout_fraction=0.25, seed=7)
        self.assertEqual(
            [item["sample_id"] for item in first_train],
            [item["sample_id"] for item in second_train],
        )
        self.assertEqual(
            [item["sample_id"] for item in first_holdout],
            [item["sample_id"] for item in second_holdout],
        )

    def test_split_is_stratified_and_disjoint(self) -> None:
        train, holdout = stratified_split(SAMPLES, holdout_fraction=0.25, seed=7)
        self.assertEqual(len(train) + len(holdout), len(SAMPLES))
        train_ids = {item["sample_id"] for item in train}
        holdout_ids = {item["sample_id"] for item in holdout}
        self.assertFalse(train_ids & holdout_ids)
        holdout_scenarios = {item["scenario_type"] for item in holdout}
        self.assertIn("buying", holdout_scenarios)
        self.assertIn("browsing", holdout_scenarios)


class SampleWeightsTest(unittest.TestCase):
    def test_sampled_weights_stay_in_multiplier_band(self) -> None:
        import random

        from starter.retrieval import RetrievalWeights

        defaults = RetrievalWeights()
        sampled = sample_weights(random.Random(3))
        for name in defaults.__dataclass_fields__:
            default_value = getattr(defaults, name)
            value = getattr(sampled, name)
            if default_value == 0:
                self.assertEqual(value, 0)
            else:
                ratio = value / default_value
                self.assertGreaterEqual(ratio, 0.5)
                self.assertLessEqual(ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
