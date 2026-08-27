from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from starter.cross_encoder_reranker import CrossEncoderReranker
from starter.state import ShoppingState
from starter.vector_index import VectorCatalogIndex


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
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
        "of",
        "on",
        "or",
        "please",
        "show",
        "some",
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


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def lexical_terms(values: Iterable[str], limit: int = 48) -> list[str]:
    result: list[str] = []
    for value in values:
        for token in TOKEN_RE.findall(value.lower()):
            if len(token) <= 1 or token in STOPWORDS or token in result:
                continue
            result.append(token)
            if len(result) >= limit:
                return result
    return result


def fts_expression(terms: Iterable[str], columns: tuple[str, ...] = ()) -> str:
    clauses: list[str] = []
    for term in terms:
        escaped = term.replace('"', '""')
        if columns:
            clauses.append(" OR ".join(f'{column}:"{escaped}"' for column in columns))
        else:
            clauses.append(f'"{escaped}"')
    return " OR ".join(f"({clause})" for clause in clauses)


def phrase_terms(values: Iterable[str]) -> list[str]:
    phrases: list[str] = []
    for value in values:
        terms = lexical_terms([value], limit=12)
        if len(terms) < 2:
            continue
        phrase = " ".join(terms)
        if phrase not in phrases:
            phrases.append(phrase)
        for first, second in zip(terms, terms[1:]):
            bigram = f"{first} {second}"
            if bigram not in phrases:
                phrases.append(bigram)
    return phrases


@dataclass(frozen=True)
class RouteSpec:
    name: str
    terms: list[str]
    columns: tuple[str, ...]
    weight: float
    limit: int


class CatalogRetriever:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.metadata: dict[str, dict[str, Any]] = {}
        self._fallback_ids: list[str] = []
        self.vector_index = None
        self.reranker = CrossEncoderReranker.from_environment()
        self._build_index()
        self._load_vector_index()

    def close(self) -> None:
        self.connection.close()

    def _load_vector_index(self) -> None:
        enabled = os.getenv("TECHJAM_VECTOR_ENABLED", "auto").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return
        index_dir = os.getenv("TECHJAM_VECTOR_INDEX_DIR", "data/vector_index")
        self.vector_index = VectorCatalogIndex.load_if_available(self.catalog_path, index_dir)

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                fields = {
                    "title": flatten_text(product.get("title")),
                    "categories": flatten_text(product.get("categories")),
                    "features": flatten_text(product.get("features")),
                    "details": flatten_text(product.get("details")),
                    "store": flatten_text(product.get("store")),
                    "description": flatten_text(product.get("description")),
                }
                corpus = " ".join(fields.values()).lower()
                self.metadata[parent_asin] = {
                    "title": fields["title"].lower(),
                    "categories": fields["categories"].lower(),
                    "features": fields["features"].lower(),
                    "details": fields["details"].lower(),
                    "store": fields["store"].lower(),
                    "description": fields["description"].lower(),
                    "corpus": corpus,
                    "price": self._number(product.get("price")),
                    "average_rating": self._number(product.get("average_rating")) or 0.0,
                    "rating_number": self._number(product.get("rating_number")) or 0.0,
                }
                batch.append(
                    (
                        parent_asin,
                        fields["title"],
                        fields["categories"],
                        fields["features"],
                        fields["details"],
                        fields["store"],
                        fields["description"],
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self._fallback_ids = sorted(
            self.metadata,
            key=lambda product_id: (
                -self.metadata[product_id]["average_rating"],
                -self.metadata[product_id]["rating_number"],
                product_id,
            ),
        )

    @staticmethod
    def _number(value: object) -> float | None:
        try:
            return None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            return None

    def _route(
        self,
        terms: list[str],
        limit: int,
        columns: tuple[str, ...] = (),
    ) -> list[str]:
        expression = fts_expression(terms, columns)
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [str(row[0]) for row in rows]

    def search(
        self,
        state: ShoppingState,
        latest_message: str,
        top_k: int,
    ) -> list[str]:
        if top_k <= 0:
            return []
        state_values: list[str] = []
        if state.category:
            state_values.append(state.category)
        for values in state.preferences.values():
            state_values.extend(values)
        state_values.extend(state.search_terms)
        state_terms = lexical_terms(state_values)
        latest_terms = lexical_terms([latest_message])

        if not state_terms and not latest_terms:
            return self._fallback_ids[:top_k]

        candidate_limit = max(150, top_k * 20)
        category_terms = lexical_terms([state.category] if state.category else [])
        preference_values = [
            value
            for values in state.preferences.values()
            for value in values
        ]
        preference_terms = lexical_terms(preference_values)
        relaxed_terms = lexical_terms(
            [
                *([state.category] if state.category else []),
                *state.search_terms,
                latest_message,
            ],
            limit=24,
        )
        route_specs = [
            RouteSpec(
                "category_only",
                category_terms,
                ("categories",),
                0.25,
                candidate_limit,
            ),
            RouteSpec(
                "title_category",
                state_terms,
                ("title", "categories"),
                1.25,
                candidate_limit,
            ),
            RouteSpec(
                "full_state",
                state_terms,
                (),
                0.85,
                candidate_limit,
            ),
            RouteSpec(
                "feature_details",
                preference_terms or state_terms,
                ("features", "details", "description"),
                0.35,
                candidate_limit,
            ),
            RouteSpec(
                "latest_message",
                latest_terms,
                (),
                1.50,
                candidate_limit,
            ),
            RouteSpec(
                "exact_phrase",
                phrase_terms([latest_message, *preference_values]),
                ("title", "features", "details", "description"),
                0.45,
                max(50, top_k * 10),
            ),
            RouteSpec(
                "relaxed",
                relaxed_terms,
                (),
                0.20,
                candidate_limit,
            ),
        ]
        routes = [
            (self._route(spec.terms, spec.limit, spec.columns), spec.weight)
            for spec in route_specs
            if spec.terms
        ]
        if self.vector_index is not None:
            try:
                routes.append(
                    (self.vector_index.search(state, latest_message, 30), 0.05)
                )
            except Exception:
                pass
        scores: defaultdict[str, float] = defaultdict(float)
        route_hits: defaultdict[str, int] = defaultdict(int)
        for identifiers, route_weight in routes:
            for rank, product_id in enumerate(identifiers, start=1):
                scores[product_id] += route_weight / (8.0 + rank)
                route_hits[product_id] += 1

        if not scores:
            return self._fallback_ids[:top_k]

        latest_phrase = " ".join(latest_terms)
        latest_override = bool(
            re.search(
                r"\b(actually|instead|changed my mind|ignore|what i need is)\b",
                latest_message,
                re.IGNORECASE,
            )
        )
        state_weight = 0.15 if latest_override else 1.0
        for product_id in list(scores):
            product = self.metadata[product_id]
            corpus = product["corpus"]
            title_and_category = f"{product['title']} {product['categories']}"
            scores[product_id] += 0.12 * max(0, route_hits[product_id] - 1)
            if state.category and state.category.lower() in product["categories"]:
                scores[product_id] += 1.8 * state_weight
            if latest_phrase and latest_phrase in title_and_category:
                scores[product_id] += 1.5
            for attribute, values in state.preferences.items():
                for value in values:
                    if self._preference_matches(attribute, value, product):
                        scores[product_id] += 2.4 * state_weight
            for values in state.removed_preferences.values():
                for value in values:
                    if value in corpus:
                        scores[product_id] -= 3.0
            scores[product_id] += min(product["average_rating"], 5.0) * 0.002
            scores[product_id] += math.log1p(product["rating_number"]) * 0.0002

        ranked = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))
        if self.reranker is not None:
            ranked = self.reranker.rerank(
                state,
                latest_message,
                ranked,
                dict(scores),
                self.metadata,
            )
        return ranked[:top_k]

    @staticmethod
    def _preference_matches(
        attribute: str, value: str, product: dict[str, Any]
    ) -> bool:
        if attribute == "budget":
            match = NUMBER_RE.search(value)
            price = product["price"]
            if not match or price is None:
                return False
            budget = float(match.group(0))
            if any(term in value for term in ("under", "below", "less than", "up to")):
                return price <= budget
        return value.lower() in product["corpus"]
