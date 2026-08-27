from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from starter.state import ShoppingState


DEFAULT_VECTOR_INDEX_DIR = Path("data/vector_index")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def catalog_sha256(catalog_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(catalog_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_embedding_text(product: dict[str, Any]) -> str:
    parts = [
        f"Title: {product.get('title') or ''}",
        f"Brand: {product.get('store') or ''}",
        f"Categories: {flatten_text(product.get('categories'))}",
        f"Features: {flatten_text(product.get('features'))}",
        f"Details: {flatten_text(product.get('details'))}",
        f"Description: {flatten_text(product.get('description'))}",
    ]
    if product.get("price") not in (None, ""):
        parts.append(f"Price: {product['price']}")
    if product.get("average_rating") not in (None, ""):
        parts.append(f"Rating: {product['average_rating']}")
    return "\n".join(part for part in parts if part.strip())


def query_embedding_text(state: ShoppingState, latest_message: str) -> str:
    parts = [f"Latest shopper message: {latest_message}"]
    if state.category:
        parts.append(f"Category: {state.category}")
    for attribute, values in sorted(state.preferences.items()):
        if values:
            parts.append(f"{attribute}: {', '.join(values)}")
    if state.search_terms:
        parts.append(f"Search terms: {', '.join(state.search_terms)}")
    if state.user_profile:
        profile_terms = state.user_profile.get("preference_tags") or []
        if profile_terms:
            parts.append(f"User profile preferences: {', '.join(map(str, profile_terms))}")
    return "\n".join(parts)


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


class VectorCatalogIndex:
    def __init__(self, index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR) -> None:
        self.index_dir = Path(index_dir)
        self.config = json.loads((self.index_dir / "config.json").read_text(encoding="utf-8"))
        self.parent_asins = json.loads(
            (self.index_dir / "parent_asins.json").read_text(encoding="utf-8")
        )
        self._np = _import_numpy()
        self.embeddings = self._np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self._model = None

    @classmethod
    def load_if_available(
        cls,
        catalog_path: str | Path,
        index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    ) -> "VectorCatalogIndex | None":
        directory = Path(index_dir)
        required = ("config.json", "parent_asins.json", "embeddings.npy")
        if not all((directory / name).exists() for name in required):
            return None
        try:
            instance = cls(directory)
            expected_hash = catalog_sha256(catalog_path)
            if instance.config.get("catalog_sha256") != expected_hash:
                return None
            return instance
        except Exception:
            return None

    def search(self, state: ShoppingState, latest_message: str, top_k: int) -> list[str]:
        if top_k <= 0:
            return []
        model = self._embedding_model()
        query = query_embedding_text(state, latest_message)
        vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_vector = self._np.asarray(vector, dtype="float32")[0]
        scores = self.embeddings @ query_vector
        limit = min(top_k, len(self.parent_asins))
        if limit <= 0:
            return []
        candidate_indexes = self._np.argpartition(-scores, limit - 1)[:limit]
        ranked_indexes = candidate_indexes[self._np.argsort(-scores[candidate_indexes])]
        return [self.parent_asins[int(index)] for index in ranked_indexes]

    def _embedding_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_path = self.config.get("model_path")
            if model_path:
                path = self.index_dir / str(model_path)
                self._model = SentenceTransformer(str(path), local_files_only=True)
            else:
                self._model = SentenceTransformer(
                    str(self.config["model_name"]), local_files_only=True
                )
        return self._model


def build_vector_index(
    catalog_path: str | Path = "data/catalog.jsonl",
    index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> dict[str, Any]:
    np = _import_numpy()
    from sentence_transformers import SentenceTransformer

    catalog = Path(catalog_path)
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)

    parent_asins: list[str] = []
    texts: list[str] = []
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asins.append(str(product["parent_asin"]))
            texts.append(product_embedding_text(product))

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(embeddings, dtype="float32")
    np.save(directory / "embeddings.npy", matrix)
    model.save(str(directory / "model"))
    (directory / "parent_asins.json").write_text(
        json.dumps(parent_asins, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "model_name": model_name,
        "catalog_path": str(catalog),
        "catalog_sha256": catalog_sha256(catalog),
        "row_count": len(parent_asins),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "model_path": "model",
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Vector search requires numpy and sentence-transformers. "
            "Install optional dependencies from requirements-vector.txt."
        ) from exc
    return np
