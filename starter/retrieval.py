from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from starter.state import ShoppingState


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


OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore|what i need is)\b", re.IGNORECASE
)
ROUTE_LIMIT = 150
CANDIDATE_LIMIT = 500
# Fusion contributes at most ``weight / (RRF_OFFSET + rank)`` per route. A large
# offset flattens that band until the additive boosts below decide every ordering
# and BM25 rank stops mattering, so keep it small enough that lexical evidence
# still separates candidates that share the same boosts.
RRF_OFFSET = 8.0


@dataclass(frozen=True)
class AttributeDiagnostic:
    attribute: str
    coverage: float
    disagreement: float
    relevance: float


@dataclass(frozen=True)
class RankedCandidate:
    product_id: str
    score: float
    route_ranks: tuple[tuple[str, int], ...]
    matched_attributes: frozenset[str] = frozenset()
    score_components: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class SearchResult:
    recommendations: tuple[str, ...]
    candidates: tuple[RankedCandidate, ...]
    diagnostics: dict[str, AttributeDiagnostic]

    @classmethod
    def empty(cls) -> "SearchResult":
        return cls(recommendations=(), candidates=(), diagnostics={})


@dataclass(frozen=True)
class _RouteSpec:
    name: str
    terms: tuple[str, ...]
    columns: tuple[str, ...]
    weight: float
    exact_phrase: bool = False


def _is_override(message: str) -> bool:
    return bool(OVERRIDE_RE.search(message))


def _phrase_expression(values: Iterable[str], columns: tuple[str, ...]) -> str:
    phrases: list[str] = []
    for value in values:
        tokens = lexical_terms([value])
        if len(tokens) < 2:
            continue
        candidates = [" ".join(tokens)]
        candidates.extend(
            f"{first} {second}" for first, second in zip(tokens, tokens[1:])
        )
        for phrase in candidates:
            if phrase not in phrases:
                phrases.append(phrase)
    return fts_expression(phrases, columns)


def _route_specs(state: ShoppingState, latest_message: str) -> tuple[_RouteSpec, ...]:
    category_values = [state.category] if state.category else []
    functional_values = [
        *state.preferences.get("feature", ()),
        *state.preferences.get("use_case", ()),
        latest_message,
    ]
    attribute_values = [
        value
        for attribute in ("material", "color", "size", "style", "brand", "budget")
        for value in state.preferences.get(attribute, ())
    ]
    latest_values = [latest_message]
    relaxed_values = [
        *category_values,
        *functional_values,
        *attribute_values,
        *state.search_terms,
    ]

    return tuple(
        route
        for route in (
            _RouteSpec(
                "category",
                tuple(lexical_terms(category_values)),
                ("title", "categories"),
                1.40,
            ),
            _RouteSpec(
                "feature_use_case",
                tuple(lexical_terms(functional_values)),
                ("features", "details", "description", "categories"),
                1.35,
            ),
            _RouteSpec(
                "exact_phrase",
                tuple(
                    value
                    for value in [*functional_values, latest_message]
                    if len(lexical_terms([value])) >= 2
                ),
                ("title", "features", "details", "description"),
                1.60,
                True,
            ),
            _RouteSpec(
                "attribute",
                tuple(lexical_terms(attribute_values)),
                ("title", "details", "store", "description"),
                1.25,
            ),
            _RouteSpec("relaxed", tuple(lexical_terms(relaxed_values)), (), 0.80),
            _RouteSpec(
                "latest_message",
                tuple(lexical_terms(latest_values)),
                (),
                2.20 if _is_override(latest_message) else 1.50,
            ),
        )
        if route.terms
    )


CATEGORY_BOOST = 1.8
CONFIRMED_ATTRIBUTE_BOOST = 2.4
EXACT_PHRASE_BOOST = 1.5
REMOVED_ATTRIBUTE_PENALTY = 4.0
# The general profile only breaks ties, so its total contribution must stay below
# what a single top-ranked route hit is worth (see the regression test).
MAX_PROFILE_BOOST = 0.03
PROFILE_TERM_BOOST = 0.005
DIAGNOSTIC_POOL = 100

ATTRIBUTE_LEXICONS = {
    "material": frozenset(
        {
            "cotton",
            "polyester",
            "nylon",
            "leather",
            "wool",
            "spandex",
            "silk",
            "rayon",
            "fabric",
            "synthetic",
        }
    ),
    "color": frozenset(
        {
            "black",
            "white",
            "blue",
            "red",
            "pink",
            "green",
            "brown",
            "gray",
            "grey",
            "purple",
            "yellow",
            "orange",
        }
    ),
    "use_case": frozenset(
        {
            "hiking",
            "running",
            "gym",
            "winter",
            "outdoor",
            "work",
            "walking",
            "travel",
            "marathon",
        }
    ),
}
ATTRIBUTE_RELEVANCE_PRIORS = {
    "category": 0.75,
    "feature": 0.85,
    "use_case": 0.80,
    "style": 0.65,
    "material": 0.65,
    "color": 0.55,
    "size": 0.50,
    "brand": 0.30,
    "budget": 0.30,
    "other": 0.0,
}


def _attribute_text(product: dict[str, Any], attribute: str) -> str:
    if attribute == "category":
        return product["categories"]
    if attribute in {"material", "color", "size"}:
        return f"{product['title']} {product['details']} {product['features']}"
    if attribute == "style":
        return f"{product['title']} {product['categories']} {product['details']}"
    if attribute == "brand":
        return f"{product['store']} {product['details']}"
    if attribute == "feature":
        return f"{product['features']} {product['details']} {product['description']}"
    if attribute == "use_case":
        return f"{product['categories']} {product['features']} {product['description']}"
    return product["corpus"]


def _attribute_signature(product: dict[str, Any], attribute: str) -> str:
    if attribute in ATTRIBUTE_LEXICONS:
        tokens = {token.lower() for token in TOKEN_RE.findall(_attribute_text(product, attribute))}
        return " ".join(sorted(ATTRIBUTE_LEXICONS[attribute] & tokens))
    if attribute == "budget":
        price = product["price"]
        return "" if price is None else f"band{int(price // 25)}"
    return " ".join(lexical_terms([_attribute_text(product, attribute)])[:4])


def _profile_terms(state: ShoppingState) -> list[str]:
    profile = state.user_profile or {}
    values = [str(profile.get("summary", ""))]
    values.extend(str(tag) for tag in profile.get("preference_tags", []) or [])
    return lexical_terms(values, limit=16)


def select_diverse_recommendations(
    candidates: tuple[RankedCandidate, ...], top_k: int
) -> tuple[str, ...]:
    if top_k <= 0 or not candidates:
        return ()
    locked_count = min(6, top_k, len(candidates))
    selected = list(candidates[:locked_count])
    remaining = list(candidates[locked_count:])
    threshold_index = min(len(candidates) - 1, max(top_k - 1, top_k * 3 - 1))
    minimum_score = candidates[threshold_index].score

    while len(selected) < min(top_k, len(candidates)):
        eligible = [item for item in remaining if item.score >= minimum_score]
        if not eligible:
            eligible = remaining
        route_counts: defaultdict[str, int] = defaultdict(int)
        for item in selected:
            for route, _ in item.route_ranks:
                route_counts[route] += 1
        choice = min(
            eligible,
            key=lambda item: (
                min((route_counts[route] for route, _ in item.route_ranks), default=0),
                -item.score,
                item.product_id,
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(item.product_id for item in selected)


class CatalogRetriever:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.metadata: dict[str, dict[str, Any]] = {}
        self._fallback_ids: list[str] = []
        self._build_index()

    def close(self) -> None:
        self.connection.close()

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

    def _run_route(self, route: _RouteSpec) -> list[str]:
        if route.exact_phrase:
            expression = _phrase_expression(route.terms, route.columns)
        else:
            expression = fts_expression(route.terms, route.columns)
        if not expression:
            return []
        rows = self._match(expression)
        if rows is None:
            reduced = fts_expression(lexical_terms(route.terms)[:12])
            rows = self._match(reduced) if reduced else []
            if rows is None:
                return []
        return rows

    def _match(self, expression: str) -> list[str] | None:
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, ROUTE_LIMIT),
            ).fetchall()
        except sqlite3.Error:
            return None
        return [str(row[0]) for row in rows]

    def _fallback_result(self, top_k: int) -> SearchResult:
        identifiers = self._fallback_ids[:top_k]
        candidates = tuple(
            RankedCandidate(
                product_id=product_id,
                score=float(len(identifiers) - rank + 1),
                route_ranks=(("fallback", rank),),
            )
            for rank, product_id in enumerate(identifiers, start=1)
        )
        return SearchResult(
            recommendations=tuple(identifiers), candidates=candidates, diagnostics={}
        )

    def search(
        self,
        state: ShoppingState,
        latest_message: str,
        top_k: int,
    ) -> SearchResult:
        if top_k <= 0:
            return SearchResult.empty()

        routes = _route_specs(state, latest_message)
        if not routes:
            return self._fallback_result(top_k)

        scores: defaultdict[str, float] = defaultdict(float)
        route_ranks: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for route in routes:
            for rank, product_id in enumerate(self._run_route(route), start=1):
                if product_id not in scores and len(scores) >= CANDIDATE_LIMIT:
                    continue
                scores[product_id] += route.weight / (RRF_OFFSET + rank)
                route_ranks[product_id].append((route.name, rank))

        if not scores:
            return self._fallback_result(top_k)

        latest_phrase = " ".join(lexical_terms([latest_message]))
        state_weight = 0.15 if _is_override(latest_message) else 1.0
        profile_terms = _profile_terms(state)
        components: dict[str, defaultdict[str, float]] = {}
        matched: dict[str, set[str]] = {}
        for product_id in list(scores):
            product = self.metadata[product_id]
            parts: defaultdict[str, float] = defaultdict(float)
            parts["fusion"] = scores[product_id]
            parts["fusion"] += 0.12 * max(0, len(route_ranks[product_id]) - 1)
            hits: set[str] = set()

            if state.category and state.category.lower() in product["categories"]:
                parts["category"] += CATEGORY_BOOST * state_weight
            title_and_category = f"{product['title']} {product['categories']}"
            if latest_phrase and latest_phrase in title_and_category:
                parts["exact_phrase"] += EXACT_PHRASE_BOOST
            for attribute, values in state.preferences.items():
                for value in values:
                    if self._preference_matches(attribute, value, product):
                        parts[f"preference:{attribute}"] += (
                            CONFIRMED_ATTRIBUTE_BOOST * state_weight
                        )
                        hits.add(attribute)
            for attribute, values in state.removed_preferences.items():
                for value in values:
                    if self._preference_matches(attribute, value, product):
                        parts[f"removed:{attribute}"] -= REMOVED_ATTRIBUTE_PENALTY
            if profile_terms:
                corpus = product["corpus"]
                matched_terms = sum(1 for term in profile_terms if term in corpus)
                if matched_terms:
                    parts["profile"] += min(
                        MAX_PROFILE_BOOST, PROFILE_TERM_BOOST * matched_terms
                    )
            parts["rating"] += min(product["average_rating"], 5.0) * 0.002
            parts["rating_count"] += math.log1p(product["rating_number"]) * 0.0002

            components[product_id] = parts
            matched[product_id] = hits
            scores[product_id] = sum(parts.values())

        candidates = tuple(
            RankedCandidate(
                product_id=product_id,
                score=scores[product_id],
                route_ranks=tuple(sorted(route_ranks[product_id])),
                matched_attributes=frozenset(matched[product_id]),
                score_components=tuple(
                    sorted(
                        (key, value)
                        for key, value in components[product_id].items()
                        if value
                    )
                ),
            )
            for product_id in sorted(scores, key=lambda item: (-scores[item], item))
        )
        return SearchResult(
            recommendations=select_diverse_recommendations(candidates, top_k),
            candidates=candidates,
            diagnostics=self._diagnostics(candidates),
        )

    def _diagnostics(
        self, candidates: tuple[RankedCandidate, ...]
    ) -> dict[str, AttributeDiagnostic]:
        pool = candidates[:DIAGNOSTIC_POOL]
        diagnostics: dict[str, AttributeDiagnostic] = {}
        if not pool:
            return diagnostics
        for attribute, prior in ATTRIBUTE_RELEVANCE_PRIORS.items():
            if attribute == "other":
                continue
            signatures = [
                _attribute_signature(self.metadata[item.product_id], attribute)
                for item in pool
            ]
            present = [signature for signature in signatures if signature]
            covered = len(present)
            coverage = covered / len(pool)
            disagreement = min(1.0, len(set(present)) / max(2, covered))
            relevance = min(1.0, prior + 0.15 * coverage)
            diagnostics[attribute] = AttributeDiagnostic(
                attribute=attribute,
                coverage=coverage,
                disagreement=disagreement,
                relevance=relevance,
            )
        return diagnostics

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
            return False
        return value.lower() in _attribute_text(product, attribute)
