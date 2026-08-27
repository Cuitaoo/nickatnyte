from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from starter.reranker import (
    CandidateReranker,
    InvalidRerank,
    RerankOutput,
    RerankResult,
)
from starter.retrieval import RankedCandidate
from starter.state import ShoppingState


METADATA = {
    "A": {"title": "alpha shirt", "details": "cotton", "price": 10.0},
    "B": {"title": "beta shirt", "details": "polyester", "price": 12.0},
    "C": {"title": "gamma shirt", "details": "linen", "price": 14.0},
    "D": {"title": "delta shirt", "details": "wool", "price": 16.0},
}


def candidates(*ids: str) -> tuple[RankedCandidate, ...]:
    return tuple(
        RankedCandidate(product_id=product_id, score=1.0, route_ranks=())
        for product_id in ids
    )


class FakeMessage:
    usage_metadata = {"input_tokens": 100, "output_tokens": 20}
    response_metadata: dict = {}


class FakeStructuredModel:
    def __init__(self, order: list[int]) -> None:
        self._order = order
        self.last_input = None

    def invoke(self, messages: object) -> dict:
        self.last_input = messages
        return {
            "raw": FakeMessage(),
            "parsed": RerankOutput(order=self._order),
            "parsing_error": None,
        }


class FakeModel:
    def __init__(self, order: list[int]) -> None:
        self.structured = FakeStructuredModel(order)

    def with_structured_output(self, schema: object, **kwargs: object) -> FakeStructuredModel:
        return self.structured


class RerankerTest(unittest.TestCase):
    def test_reorders_head_and_keeps_tail(self) -> None:
        reranker = CandidateReranker(FakeModel([2, 0, 1]))
        result = reranker.rerank(
            ShoppingState.new("s", {}),
            "message",
            candidates("A", "B", "C", "D"),
            METADATA,
            limit=3,
        )
        self.assertIsInstance(result, RerankResult)
        self.assertEqual(result.ordering, ("C", "A", "B", "D"))
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 20)

    def test_non_permutation_raises_invalid_rerank(self) -> None:
        reranker = CandidateReranker(FakeModel([0, 0, 1]))
        with self.assertRaises(InvalidRerank):
            reranker.rerank(
                ShoppingState.new("s", {}),
                "message",
                candidates("A", "B", "C"),
                METADATA,
                limit=3,
            )

    def test_from_environment_disabled(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_ENABLED": "false", "OPENAI_API_KEY": "sk-test"},
            clear=False,
        ):
            self.assertIsNone(CandidateReranker.from_environment())
        with patch.dict(
            os.environ,
            {
                "OPENAI_ENABLED": "true",
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_RERANK_ENABLED": "false",
            },
            clear=False,
        ):
            self.assertIsNone(CandidateReranker.from_environment())
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            self.assertIsNone(CandidateReranker.from_environment())


class AgentRerankIntegrationTest(unittest.TestCase):
    def test_agent_uses_rerank_ordering_and_reports_usage(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from starter.agent import Agent

        products = [
            {
                "parent_asin": pid,
                "title": f"{name} cotton shirt",
                "categories": ["Men", "Clothing", "Shirts"],
                "features": ["soft"],
                "details": {"color": "red"},
                "description": ["shirt"],
                "store": "Basics",
                "price": 20.0,
                "average_rating": rating,
                "rating_number": 100,
            }
            for pid, name, rating in (
                ("S1", "alpha", 4.9),
                ("S2", "beta", 4.5),
                ("S3", "gamma", 4.1),
            )
        ]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

        class ReverseReranker:
            def rerank(self, state, message, cands, metadata, limit=20):
                ordering = tuple(
                    item.product_id for item in reversed(cands[:limit])
                ) + tuple(item.product_id for item in cands[limit:])
                return RerankResult(
                    ordering=ordering, prompt_tokens=7, completion_tokens=3
                )

        agent = Agent(catalog_path, interpreter=None, reranker=ReverseReranker())
        self.addCleanup(agent.close)
        agent.reset("s", {})
        baseline = Agent(catalog_path, interpreter=None, reranker=None)
        self.addCleanup(baseline.close)
        baseline.reset("s", {})

        base_response = baseline.respond("s", "cotton shirt", 1, 3)
        response = agent.respond("s", "cotton shirt", 1, 3)

        base_ids = [item["parent_asin"] for item in base_response["recommendations"]]
        ids = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(ids, list(reversed(base_ids)))
        self.assertEqual(response["usage"]["prompt_tokens"], 7)
        self.assertEqual(response["usage"]["completion_tokens"], 3)

    def test_agent_keeps_original_order_when_reranker_fails(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from starter.agent import Agent

        products = [
            {
                "parent_asin": "S1",
                "title": "alpha cotton shirt",
                "categories": ["Men", "Clothing", "Shirts"],
                "features": ["soft"],
                "details": {"color": "red"},
                "description": ["shirt"],
                "store": "Basics",
                "price": 20.0,
                "average_rating": 4.9,
                "rating_number": 100,
            }
        ]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        catalog_path = Path(directory.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

        class FailingReranker:
            def rerank(self, *args, **kwargs):
                raise RuntimeError("boom")

        agent = Agent(catalog_path, interpreter=None, reranker=FailingReranker())
        self.addCleanup(agent.close)
        agent.reset("s", {})
        response = agent.respond("s", "cotton shirt", 1, 3)
        ids = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(ids, ["S1"])


if __name__ == "__main__":
    unittest.main()
