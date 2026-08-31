from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from starter.state import ShoppingState
from starter.query_expansion import ScenarioHypothesis


DEFAULT_VECTOR_INDEX_DIR = Path("data/vector_index")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "a",
        "actually",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "looking",
        "me",
        "my",
        "need",
        "no",
        "of",
        "on",
        "or",
        "preference",
        "please",
        "show",
        "some",
        "still",
        "that",
        "the",
        "this",
        "to",
        "want",
        "with",
        "would",
        "you",
    }
)
FEATURE_ATTRIBUTES = (
    "feature",
    "use_case",
    "material",
    "style",
    "size",
    "brand",
    "color",
    "budget",
    "other",
)


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


def category_embedding_text(product: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            f"Title: {product.get('title') or ''}",
            f"Categories: {flatten_text(product.get('categories'))}",
        )
        if part.strip()
    )


def feature_embedding_text(product: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            f"Features: {_clip_text(flatten_text(product.get('features')), 48)}",
            f"Details: {_clip_text(flatten_text(product.get('details')), 64)}",
            f"Description: {_clip_text(flatten_text(product.get('description')), 48)}",
            f"Brand: {_clip_text(product.get('store') or '', 16)}",
        )
        if part.strip()
    )


def full_query_embedding_text(state: ShoppingState, latest_message: str) -> str:
    parts = [f"Latest shopper message: {latest_message}"]
    if state.category:
        parts.append(f"Category: {state.category}")
    for attribute, values in sorted(state.preferences.items()):
        if values:
            parts.append(f"{attribute}: {', '.join(values)}")
    for attribute, values in sorted(state.removed_preferences.items()):
        if values:
            parts.append(f"avoid {attribute}: {', '.join(values)}")
    if state.search_terms:
        parts.append(f"Search terms: {', '.join(state.search_terms)}")
    if state.user_profile:
        profile_terms = state.user_profile.get("preference_tags") or []
        if profile_terms:
            parts.append(f"User profile preferences: {', '.join(map(str, profile_terms))}")
    return "\n".join(parts)


def category_query_embedding_text(state: ShoppingState, latest_message: str) -> str:
    """Return only the explicit product/category signal for title recall."""

    # Raw-message context, such as a country name or broad shopper goal, must
    # not leak into this embedding. It belongs either to confirmed state or the
    # separate, optional scenario-only feature route.
    del latest_message
    return str(state.category or "").strip()


def feature_query_embedding_text(state: ShoppingState, latest_message: str) -> str:
    parts: list[str] = []
    for attribute in FEATURE_ATTRIBUTES:
        parts.extend(state.preferences.get(attribute, ()))
    parts.extend(state.search_terms)
    for attribute, values in sorted(state.removed_preferences.items()):
        if values:
            parts.append(f"avoid {attribute}: {', '.join(values)}")
    return " ".join(_dedupe(parts))


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _clip_text(value: object, max_terms: int) -> str:
    return " ".join(_content_terms([value], limit=max_terms))


def _content_terms(values: Any, limit: int = 48) -> list[str]:
    result: list[str] = []
    for value in values:
        for token in TOKEN_RE.findall(str(value).lower()):
            if len(token) <= 1 or token in STOPWORDS or token in result:
                continue
            result.append(token)
            if len(result) >= limit:
                return result
    return result


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split()).lower()
        if not normalized or normalized in seen:
            continue
        result.append(str(value))
        seen.add(normalized)
    return result


def _route_allowed(route: str, allowed_routes: set[str] | None) -> bool:
    if allowed_routes is None or route in allowed_routes:
        return True
    return route.startswith("vector_scenario_") and "vector_scenario" in allowed_routes


class VectorCatalogIndex:
    def __init__(self, index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR) -> None:
        self.index_dir = Path(index_dir)
        self.config = json.loads(
            (self.index_dir / "config.json").read_text(encoding="utf-8")
        )
        self.parent_asins = json.loads(
            (self.index_dir / "parent_asins.json").read_text(encoding="utf-8")
        )
        self._np = _import_numpy()
        self.embeddings = self._load_optional_matrix("embeddings.npy")
        self.category_embeddings = self._load_optional_matrix("category_embeddings.npy")
        self.feature_embeddings = self._load_optional_matrix("feature_embeddings.npy")
        self._model = None

    @classmethod
    def load_if_available(
        cls,
        catalog_path: str | Path,
        index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    ) -> "VectorCatalogIndex | None":
        directory = Path(index_dir)
        required = ("config.json", "parent_asins.json")
        if not all((directory / name).exists() for name in required):
            return None
        try:
            instance = cls(directory)
            if (
                instance.embeddings is None
                and instance.category_embeddings is None
                and instance.feature_embeddings is None
            ):
                return None
            if instance.config.get("catalog_sha256") != catalog_sha256(catalog_path):
                return None
            return instance
        except Exception:
            return None

    def search(self, state: ShoppingState, latest_message: str, top_k: int) -> list[str]:
        matrix = self.embeddings
        if matrix is None:
            matrix = self.category_embeddings
        if matrix is None:
            matrix = self.feature_embeddings
        if matrix is None:
            return []
        return self._search_matrix(
            matrix,
            full_query_embedding_text(state, latest_message),
            top_k,
        )

    def search_routes(
        self,
        state: ShoppingState,
        latest_message: str,
        top_k: int,
        allowed_routes: set[str] | None = None,
        scenario_hypotheses: tuple[ScenarioHypothesis, ...] = (),
    ) -> dict[str, list[str]]:
        return {
            route: [product_id for product_id, _score in rows]
            for route, rows in self.search_routes_with_scores(
                state,
                latest_message,
                top_k,
                allowed_routes,
                scenario_hypotheses=scenario_hypotheses,
            ).items()
        }

    def search_routes_with_scores(
        self,
        state: ShoppingState,
        latest_message: str,
        top_k: int,
        allowed_routes: set[str] | None = None,
        scenario_hypotheses: tuple[ScenarioHypothesis, ...] = (),
    ) -> dict[str, list[tuple[str, float]]]:
        queries: list[tuple[str, str, Any]] = []
        category_query = category_query_embedding_text(state, latest_message)
        if (
            category_query
            and self.category_embeddings is not None
            and _route_allowed("vector_category", allowed_routes)
        ):
            queries.append(("vector_category", category_query, self.category_embeddings))
        feature_query = feature_query_embedding_text(state, latest_message)
        if (
            feature_query
            and self.feature_embeddings is not None
            and _route_allowed("vector_feature", allowed_routes)
        ):
            queries.append(("vector_feature", feature_query, self.feature_embeddings))
        if (
            not queries
            and self.embeddings is not None
            and _route_allowed("vector", allowed_routes)
        ):
            queries.append(
                (
                    "vector",
                    full_query_embedding_text(state, latest_message),
                    self.embeddings,
                )
            )
        scenario_matrix = self.feature_embeddings
        if scenario_matrix is None:
            scenario_matrix = self.embeddings
        if scenario_matrix is not None and _route_allowed(
            "vector_scenario", allowed_routes
        ):
            for index, hypothesis in enumerate(scenario_hypotheses, start=1):
                scenario_query = hypothesis.scenario_query.strip()
                if scenario_query:
                    queries.append(
                        (f"vector_scenario_{index}", scenario_query, scenario_matrix)
                    )
        if not queries:
            return {}

        model = self._embedding_model()
        vectors = model.encode(
            [query for _route, query, _matrix in queries],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        results: dict[str, list[tuple[str, float]]] = {}
        for index, (route, _query, matrix) in enumerate(queries):
            results[route] = self._rank_matrix_with_scores(
                matrix, vectors[index], top_k
            )
        return results

    def _load_optional_matrix(self, filename: str):
        path = self.index_dir / filename
        if not path.exists():
            return None
        return self._np.load(path, mmap_mode="r")

    def _search_matrix(self, matrix: Any, query: str, top_k: int) -> list[str]:
        if top_k <= 0:
            return []
        model = self._embedding_model()
        vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return self._rank_matrix(matrix, vector[0], top_k)

    def _rank_matrix(self, matrix: Any, vector: Any, top_k: int) -> list[str]:
        return [
            product_id
            for product_id, _score in self._rank_matrix_with_scores(
                matrix, vector, top_k
            )
        ]

    def _rank_matrix_with_scores(
        self, matrix: Any, vector: Any, top_k: int
    ) -> list[tuple[str, float]]:
        query_vector = self._np.asarray(vector, dtype="float32")
        scores = matrix @ query_vector
        limit = min(top_k, len(self.parent_asins))
        if limit <= 0:
            return []
        candidate_indexes = self._np.argpartition(-scores, limit - 1)[:limit]
        ranked_indexes = candidate_indexes[self._np.argsort(-scores[candidate_indexes])]
        return [
            (self.parent_asins[int(index)], float(scores[int(index)]))
            for index in ranked_indexes
        ]

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
            if self.config.get("max_seq_length"):
                self._model.max_seq_length = int(self.config["max_seq_length"])
        return self._model


def build_vector_index(
    catalog_path: str | Path = "data/catalog.jsonl",
    index_dir: str | Path = DEFAULT_VECTOR_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    include_full: bool = False,
    max_seq_length: int = 128,
) -> dict[str, Any]:
    np = _import_numpy()
    from sentence_transformers import SentenceTransformer

    catalog = Path(catalog_path)
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)

    parent_asins: list[str] = []
    full_texts: list[str] = []
    category_texts: list[str] = []
    feature_texts: list[str] = []
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asins.append(str(product["parent_asin"]))
            if include_full:
                full_texts.append(product_embedding_text(product))
            category_texts.append(category_embedding_text(product))
            feature_texts.append(feature_embedding_text(product))

    model = SentenceTransformer(model_name)
    model.max_seq_length = min(int(getattr(model, "max_seq_length", 512)), max_seq_length)
    category_embeddings = model.encode(
        category_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    feature_embeddings = model.encode(
        feature_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    category_matrix = np.asarray(category_embeddings, dtype="float32")
    feature_matrix = np.asarray(feature_embeddings, dtype="float32")
    if include_full:
        full_embeddings = model.encode(
            full_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        full_matrix = np.asarray(full_embeddings, dtype="float32")
        np.save(directory / "embeddings.npy", full_matrix)
    np.save(directory / "category_embeddings.npy", category_matrix)
    np.save(directory / "feature_embeddings.npy", feature_matrix)
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
        "embedding_dim": int(category_matrix.shape[1])
        if category_matrix.ndim == 2
        else 0,
        "model_path": "model",
        "max_seq_length": model.max_seq_length,
        "routes": [
            *(["vector"] if include_full else []),
            "vector_category",
            "vector_feature",
        ],
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
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
