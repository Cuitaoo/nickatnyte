from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from starter.retrieval import CatalogRetriever
from starter.vector_index import (
    VectorCatalogIndex,
    clear_memory_index_cache,
)


PRODUCTS = [
    {
        "parent_asin": "SHOE_1",
        "title": "Waterproof Walking Shoe",
        "categories": ["Women", "Shoes", "Walking"],
        "features": ["waterproof membrane"],
        "details": {"color": "black"},
        "description": ["comfortable commuter shoe"],
        "store": "Example",
    },
    {
        "parent_asin": "BAG_1",
        "title": "Lightweight Laptop Backpack",
        "categories": ["Bags", "Backpacks"],
        "features": ["padded laptop sleeve"],
        "details": {"color": "blue"},
        "description": ["daily university commute"],
        "store": "Example",
    },
]


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.max_seq_length = 512
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        return np.asarray(
            [
                [
                    float(len(str(text))),
                    float("shoe" in str(text).lower()),
                    float("backpack" in str(text).lower()),
                ]
                for text in texts
            ],
            dtype="float32",
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 3


class InMemoryVectorIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_index_cache()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(clear_memory_index_cache)
        self.catalog = Path(self.directory.name) / "catalog.jsonl"
        self.catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )

    def test_runtime_build_keeps_embeddings_in_ram_and_writes_nothing(self) -> None:
        model = FakeEmbeddingModel()
        with patch(
            "starter.vector_index._load_embedding_model", return_value=model
        ):
            index = VectorCatalogIndex.build_in_memory(
                self.catalog,
                model_name="fake-model",
                batch_size=2,
                max_seq_length=32,
            )

        self.assertEqual(index.config["storage"], "memory")
        self.assertEqual(index.config["row_count"], 2)
        self.assertEqual(index.parent_asins, ["SHOE_1", "BAG_1"])
        self.assertEqual(index.category_embeddings.shape, (2, 3))
        self.assertEqual(index.feature_embeddings.shape, (2, 3))
        self.assertFalse(index.category_embeddings.flags.writeable)
        self.assertFalse(index.feature_embeddings.flags.writeable)
        self.assertIs(index._embedding_model(), model)
        self.assertEqual(model.max_seq_length, 32)
        self.assertEqual(
            sorted(path.name for path in Path(self.directory.name).iterdir()),
            ["catalog.jsonl"],
        )

    def test_process_cache_builds_a_catalog_only_once(self) -> None:
        model = FakeEmbeddingModel()
        with patch(
            "starter.vector_index._load_embedding_model", return_value=model
        ) as loader:
            first = VectorCatalogIndex.load_in_memory_cached(
                self.catalog, model_name="fake-model", batch_size=2
            )
            second = VectorCatalogIndex.load_in_memory_cached(
                self.catalog, model_name="fake-model", batch_size=2
            )

        self.assertIs(first, second)
        loader.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "TECHJAM_VECTOR_ENABLED": "true",
            "TECHJAM_VECTOR_INDEX_MODE": "memory",
            "TECHJAM_VECTOR_MIN_CATALOG_SIZE": "1",
            "TECHJAM_VECTOR_LOCAL_ONLY": "true",
        },
    )
    def test_retriever_selects_runtime_memory_mode(self) -> None:
        sentinel = object()
        with patch.object(
            VectorCatalogIndex,
            "load_in_memory_cached",
            return_value=sentinel,
        ) as loader:
            retriever = CatalogRetriever(self.catalog)
        self.addCleanup(retriever.close)

        self.assertIs(retriever.vector_index, sentinel)
        self.assertEqual(retriever.vector_index_status, "memory")
        loader.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "TECHJAM_VECTOR_ENABLED": "true",
            "TECHJAM_VECTOR_INDEX_MODE": "memory",
            "TECHJAM_VECTOR_MIN_CATALOG_SIZE": "1",
        },
    )
    def test_runtime_embedding_failure_keeps_lexical_retrieval_available(self) -> None:
        with patch.object(
            VectorCatalogIndex,
            "load_in_memory_cached",
            side_effect=RuntimeError("model unavailable"),
        ):
            retriever = CatalogRetriever(self.catalog)
        self.addCleanup(retriever.close)

        self.assertIsNone(retriever.vector_index)
        self.assertEqual(retriever.vector_index_status, "unavailable:memory")
        self.assertIn("model unavailable", retriever.vector_index_error or "")


if __name__ == "__main__":
    unittest.main()
