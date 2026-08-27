from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from starter.cross_encoder_reranker import CrossEncoderReranker
from starter.state import ShoppingState


METADATA = {
    "A": {"title": "alpha", "store": "s", "categories": "", "features": "", "details": "", "description": ""},
    "B": {"title": "beta", "store": "s", "categories": "", "features": "", "details": "", "description": ""},
    "C": {"title": "gamma", "store": "s", "categories": "", "features": "", "details": "", "description": ""},
}


class FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return self._scores[: len(pairs)]


class CrossEncoderRerankerTest(unittest.TestCase):
    def _reranker(self, scores: list[float], weight: float = 0.35) -> CrossEncoderReranker:
        reranker = CrossEncoderReranker(max_candidates=10, weight=weight)
        reranker._model = FakeModel(scores)
        return reranker

    def test_blend_reorders_near_ties_only(self) -> None:
        # A and B are near-tied; C is far behind. The model prefers B strongly.
        reranker = self._reranker([0.0, 10.0, 5.0])
        result = reranker.rerank(
            ShoppingState.new("s", {}),
            "message",
            ["A", "B", "C"],
            {"A": 2.00, "B": 1.90, "C": 0.50},
            METADATA,
        )
        # minmax gives A=0, B=1, C=0.5 -> B: 1.9+0.35 = 2.25 > A: 2.0; C stays last.
        self.assertEqual(result, ["B", "A", "C"])

    def test_blend_cannot_overcome_large_score_gaps(self) -> None:
        reranker = self._reranker([0.0, 10.0, 0.0])
        result = reranker.rerank(
            ShoppingState.new("s", {}),
            "message",
            ["A", "B", "C"],
            {"A": 3.00, "B": 1.00, "C": 0.50},
            METADATA,
        )
        self.assertEqual(result[0], "A")

    def test_tail_beyond_max_candidates_is_preserved(self) -> None:
        reranker = CrossEncoderReranker(max_candidates=2, weight=0.35)
        reranker._model = FakeModel([0.0, 10.0])
        result = reranker.rerank(
            ShoppingState.new("s", {}),
            "message",
            ["A", "B", "C"],
            {"A": 1.00, "B": 0.99, "C": 0.50},
            METADATA,
        )
        self.assertEqual(result, ["B", "A", "C"])

    def test_model_failure_returns_original_order(self) -> None:
        class ExplodingModel:
            def predict(self, *args, **kwargs):
                raise RuntimeError("boom")

        reranker = CrossEncoderReranker()
        reranker._model = ExplodingModel()
        result = reranker.rerank(
            ShoppingState.new("s", {}),
            "message",
            ["A", "B", "C"],
            {"A": 2.0, "B": 1.0, "C": 0.5},
            METADATA,
        )
        self.assertEqual(result, ["A", "B", "C"])

    def test_from_environment_defaults_off(self) -> None:
        environment = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertIsNone(CrossEncoderReranker.from_environment())
        with patch.dict(
            os.environ,
            {**environment, "TECHJAM_RERANK_ENABLED": "true"},
            clear=True,
        ):
            self.assertIsNotNone(CrossEncoderReranker.from_environment())


if __name__ == "__main__":
    unittest.main()
