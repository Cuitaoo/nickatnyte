from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from starter.audience import apply_audience_guardrail
from starter.constraints import (
    classify_constraints,
    staged_filter,
    staged_filter_enabled,
)
from starter.cross_encoder_reranker import CrossEncoderReranker
from starter.profile import match_profile, profile_tags, semantic_profile_enabled
from starter.query_expansion import ScenarioHypothesis
from starter.state import ShoppingState
from starter.synonyms import expand_terms
from starter.tracks import (
    apply_track,
    diversity_enabled,
    browsing_track_enabled,
    dual_track_enabled,
    resolve_track,
    select_diverse_recommendations,
)
from starter.vector_index import (
    VectorCatalogIndex,
    category_query_embedding_text,
    feature_query_embedding_text,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
IDENTIFIER_LABEL_RE = re.compile(
    r"\b(?:"
    r"(?:item\s+model\s+number|model\s+(?:number|no)|style\s+(?:number|no)|"
    r"part\s+(?:number|no)|mpn|sku)\s*[:#-]?\s*"
    # Shorthand needs a digit so ordinary phrases such as "model shirt" do
    # not become an identifier route.
    r"|model\s+(?=[a-z0-9._/-]*\d)"
    r")([a-z0-9][a-z0-9._/-]{2,})",
    re.IGNORECASE,
)
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
EXACT_IDENTIFIER_BOOST = 6.0


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
    matched_profile_tags: tuple[str, ...] = ()


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


def _is_product_change(state: ShoppingState, message: str) -> bool:
    if state.last_update_type == "product_change":
        return True
    # Direct retriever callers do not pass through the state updater. Keep the
    # legacy signal for those turn-zero calls without overriding structured state.
    return state.turn == 0 and _is_override(message)


def _has_browsing_signal(message: str) -> bool:
    return bool(
        re.search(r"\b(browsing|exploring|not sure|just looking)\b", message.lower())
    )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(os.getenv(name, str(default)))
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _signature_disagreement(signatures: Iterable[str]) -> float:
    """Return Gini impurity for the observed non-empty attribute signatures."""
    counts = Counter(signature for signature in signatures if signature)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


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


def _exact_identifiers(values: Iterable[str]) -> tuple[str, ...]:
    identifiers: list[str] = []
    for value in values:
        for match in IDENTIFIER_LABEL_RE.finditer(str(value)):
            identifier = match.group(1).strip(" .,:;")
            tokens = TOKEN_RE.findall(identifier.lower())
            if not tokens:
                continue
            normalized = " ".join(tokens)
            # Avoid treating ordinary years or prices as product identifiers.
            if len(tokens) == 1 and tokens[0].isdigit() and len(tokens[0]) < 4:
                continue
            if normalized not in identifiers:
                identifiers.append(normalized)
    return tuple(identifiers)


def _route_specs(
    state: ShoppingState, latest_message: str, weights: "RetrievalWeights"
) -> tuple[_RouteSpec, ...]:
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
    identifiers = _exact_identifiers(
        [
            latest_message,
            *functional_values,
            *attribute_values,
            *state.search_terms,
        ]
    )
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
                weights.route_category,
            ),
            _RouteSpec(
                "feature_use_case",
                tuple(lexical_terms(functional_values)),
                ("features", "details", "description", "categories"),
                weights.route_feature_use_case,
            ),
            _RouteSpec(
                "exact_phrase",
                tuple(
                    value
                    for value in [*functional_values, *state.search_terms, latest_message]
                    if len(lexical_terms([value])) >= 2
                ),
                ("title", "features", "details", "description"),
                weights.route_exact_phrase,
                True,
            ),
            _RouteSpec(
                "identifier",
                identifiers,
                ("title", "features", "details", "description"),
                weights.route_exact_phrase,
            ),
            _RouteSpec(
                "attribute",
                tuple(lexical_terms(attribute_values)),
                ("title", "details", "store", "description"),
                weights.route_attribute,
            ),
            _RouteSpec(
                "relaxed", tuple(lexical_terms(relaxed_values)), (), weights.route_relaxed
            ),
            _RouteSpec(
                "latest_message",
                tuple(lexical_terms(latest_values)),
                (),
                weights.route_latest_override
                if _is_product_change(state, latest_message)
                else weights.route_latest,
            ),
        )
        if route.terms
    )


def _synonym_route(
    state: ShoppingState, latest_message: str, weights: "RetrievalWeights"
) -> _RouteSpec | None:
    """Rescue route: expand shopper vocabulary only when direct lexical routes
    miss, so template-matching sessions are unaffected."""
    source_values = [state.category] if state.category else []
    source_values.append(latest_message)
    terms = expand_terms(lexical_terms(source_values))
    if not terms:
        return None
    return _RouteSpec("synonym", terms, ("title", "categories"), weights.route_synonym)


CATEGORY_BOOST = 1.8
CONFIRMED_ATTRIBUTE_BOOST = 2.4
EXACT_PHRASE_BOOST = 1.5
REMOVED_ATTRIBUTE_PENALTY = 4.0
# The general profile only breaks ties, so its total contribution must stay below
# what a single top-ranked route hit is worth (see the regression test).
MAX_PROFILE_BOOST = 0.03
PROFILE_TERM_BOOST = 0.005
DIAGNOSTIC_POOL = 100


@dataclass(frozen=True)
class RetrievalWeights:
    """Tunable ranking constants.

    Defaults are the best random-search configuration from
    ``tools/tune_weights.py`` (seed 7, 20 trials, stratified 150/50 split;
    see docs/evaluations/weight-tuning.json): train 0.8176 / holdout 0.8352
    vs the hand-tuned baseline's 0.8103 / 0.8058.
    """

    rrf_offset: float = 5.049346763678743
    category_boost: float = 1.0586227003338649
    confirmed_attribute_boost: float = 1.3021929603979996
    exact_phrase_boost: float = 2.1756228415861134
    removed_attribute_penalty: float = 2.3927678713168463
    multi_route_bonus: float = 0.08457270764590544
    rating_coef: float = 0.0017193930814524492
    rating_count_coef: float = 0.0003346942922951214
    route_category: float = 0.7827315103345402
    route_feature_use_case: float = 1.258176402376759
    route_exact_phrase: float = 1.7135065698695175
    route_attribute: float = 2.1268167707488765
    route_relaxed: float = 1.2454193337901538
    route_latest: float = 2.484458494106324
    route_latest_override: float = 1.6181503400897523
    route_synonym: float = 0.4890644682366305

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
    "feature": frozenset(
        {
            "adjustable",
            "breathable",
            "closure",
            "comfort",
            "cushioning",
            "durable",
            "insulated",
            "lightweight",
            "lining",
            "pocket",
            "pockets",
            "sole",
            "stretch",
            "waterproof",
            "windproof",
            "zipper",
        }
    ),
    "style": frozenset(
        {
            "athletic",
            "casual",
            "classic",
            "formal",
            "loose",
            "maxi",
            "mini",
            "modern",
            "relaxed",
            "slim",
            "sporty",
            "vintage",
        }
    ),
    "size": frozenset(
        {
            "xs",
            "s",
            "m",
            "l",
            "xl",
            "xxl",
            "small",
            "medium",
            "large",
            "wide",
            "narrow",
            "petite",
            "plus",
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


def _has_attribute_metadata(attribute: str, product: dict[str, Any]) -> bool:
    """Whether the product says anything at all about this attribute.

    Used by the staged filter so that missing catalog metadata never excludes
    a product - only a product that has the field and contradicts it is cut.
    """
    if attribute == "budget":
        return product.get("price") is not None
    return bool(_attribute_text(product, attribute).strip())


def _attribute_text(product: dict[str, Any], attribute: str) -> str:
    if attribute == "category":
        return product["categories"]
    if attribute in {"material", "color", "size"}:
        return (
            f"{product['title']} {product['details']} "
            f"{product['features']} {product['description']}"
        )
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
    if attribute == "category":
        return " ".join(lexical_terms([product["categories"]])[-3:])
    if attribute == "brand":
        return " ".join(lexical_terms([product["store"]])[:2])
    return ""


def _profile_terms(state: ShoppingState) -> list[str]:
    profile = state.user_profile or {}
    values = [str(profile.get("summary", ""))]
    values.extend(str(tag) for tag in profile.get("preference_tags", []) or [])
    return lexical_terms(values, limit=16)


class CatalogRetriever:
    def __init__(
        self,
        catalog_path: str | Path,
        weights: RetrievalWeights | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.weights = weights or RetrievalWeights()
        self.connection = sqlite3.connect(":memory:")
        self.metadata: dict[str, dict[str, Any]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._fallback_ids: list[str] = []
        self.cross_encoder_reranker = CrossEncoderReranker.from_environment()
        # Final business guardrail: demote wrong-audience products after
        # semantic reranking. Soft by design, so a strong exact match wins.
        # Damp constraint values that appear across most of the catalogue.
        # "Imported" and "Zipper closure" are true of nearly everything and
        # should not carry the ranking force of a distinctive term.
        self.idf_weighting = _env_bool("TECHJAM_IDF_WEIGHTING", False)
        self.idf_min_scale = _env_float("TECHJAM_IDF_MIN_SCALE", 0.35, 0.0, 1.0)
        self.idf_damp = _env_float("TECHJAM_IDF_DAMP", 1.5, 0.0, 10.0)
        self.audience_guardrail = _env_bool("TECHJAM_AUDIENCE_GUARDRAIL", False)
        self.audience_penalty = _env_float("TECHJAM_AUDIENCE_PENALTY", 0.15, 0.0, 1.0)
        self.audience_top_n = _env_int("TECHJAM_AUDIENCE_TOP_N", 20, 1, 200)
        self.vector_index: VectorCatalogIndex | None = None
        self.vector_top_k = _env_int("TECHJAM_VECTOR_TOP_K", 30, 1, 200)
        self.vector_weight = _env_float("TECHJAM_VECTOR_WEIGHT", 0.0, 0.0, 2.0)
        self.vector_category_weight = _env_float(
            "TECHJAM_VECTOR_CATEGORY_WEIGHT", 0.0, 0.0, 2.0
        )
        self.vector_feature_weight = _env_float(
            "TECHJAM_VECTOR_FEATURE_WEIGHT", 0.05, 0.0, 2.0
        )
        # Scenario vectors are a distinct, low-weight RRF branch. They never
        # replace lexical/category evidence or turn inferred language into a
        # confirmed preference.
        self.scenario_vector_weight = _env_float(
            "TECHJAM_SCENARIO_VECTOR_WEIGHT", 0.25, 0.0, 2.0
        )
        self.vector_recall_only = _env_bool("TECHJAM_VECTOR_RECALL_ONLY", True)
        self.vector_max_doc_frequency = _env_int(
            "TECHJAM_VECTOR_MAX_DOC_FREQUENCY", 750, 1, 50_000
        )
        self.vector_min_rare_terms = _env_int(
            "TECHJAM_VECTOR_MIN_RARE_TERMS", 1, 1, 8
        )
        self.vector_policy = (
            os.getenv("TECHJAM_VECTOR_POLICY", "adaptive").strip().lower()
            or "adaptive"
        )
        self.vector_low_confidence_candidate_limit = _env_int(
            "TECHJAM_VECTOR_LOW_CONFIDENCE_CANDIDATES", 40, 1, CANDIDATE_LIMIT
        )
        self.vector_high_confidence_route_count = _env_int(
            "TECHJAM_VECTOR_HIGH_CONFIDENCE_ROUTES", 3, 1, 12
        )
        self.vector_min_similarity = _env_float(
            "TECHJAM_VECTOR_MIN_SIMILARITY", 0.45, -1.0, 1.0
        )
        self._build_index()
        self._load_vector_index()

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
                self._document_frequency.update(
                    {
                        token
                        for token in TOKEN_RE.findall(corpus)
                        if len(token) > 1 and token not in STOPWORDS
                    }
                )
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

    def _load_vector_index(self) -> None:
        if not _env_bool("TECHJAM_VECTOR_ENABLED", False):
            return
        index_dir = os.getenv("TECHJAM_VECTOR_INDEX_DIR", "data/vector_index")
        self.vector_index = VectorCatalogIndex.load_if_available(
            self.catalog_path, index_dir
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

    def fallback(self, top_k: int) -> SearchResult:
        """Return the catalog's deterministic popularity fallback."""
        return self._fallback_result(max(0, int(top_k)))

    def search(
        self,
        state: ShoppingState,
        latest_message: str,
        top_k: int,
        *,
        scenario_hypotheses: tuple[ScenarioHypothesis, ...] = (),
    ) -> SearchResult:
        if top_k <= 0:
            return SearchResult.empty()

        weights = self.weights
        # Quality signal, env-overridable so it can be swept without editing
        # the tuned defaults. public_0144 is the motivating case: the target
        # has 147 ratings at 4.3 while products above it have 2, 3 and 4.
        quality = _env_float("TECHJAM_RATING_COUNT_COEF", -1.0, -1.0, 1.0)
        if quality >= 0.0:
            weights = replace(weights, rating_count_coef=quality)
        track = (
            resolve_track(state.intent_mode, _is_product_change(state, latest_message))
            if dual_track_enabled()
            else None
        )
        if track is not None:
            weights = apply_track(weights, track)
        routes = _route_specs(state, latest_message, weights)
        if not routes:
            return self._fallback_result(top_k)

        scores: defaultdict[str, float] = defaultdict(float)
        route_ranks: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        category_rows = 0
        ran_category_route = False
        for route in routes:
            rows = self._run_route(route)
            if route.name == "category":
                ran_category_route = True
                category_rows = len(rows)
            for rank, product_id in enumerate(rows, start=1):
                scores[product_id] += route.weight / (weights.rrf_offset + rank)
                route_ranks[product_id].append((route.name, rank))

        if not scores or (ran_category_route and category_rows == 0):
            synonym_route = _synonym_route(state, latest_message, weights)
            if synonym_route is not None:
                for rank, product_id in enumerate(
                    self._run_route(synonym_route), start=1
                ):
                    scores[product_id] += synonym_route.weight / (
                        weights.rrf_offset + rank
                    )
                    route_ranks[product_id].append((synonym_route.name, rank))

        if self.vector_index is not None and (
            self.vector_weight > 0
            or self.vector_category_weight > 0
            or self.vector_feature_weight > 0
            or scenario_hypotheses
        ):
            allowed_vector_routes = self._allowed_vector_routes(
                state, latest_message, scores, route_ranks
            )
            if scenario_hypotheses and self.vector_policy not in {
                "0",
                "false",
                "no",
                "off",
            }:
                allowed_vector_routes.add("vector_scenario")
            try:
                if allowed_vector_routes:
                    search_with_scores = getattr(
                        self.vector_index, "search_routes_with_scores", None
                    )
                    if callable(search_with_scores):
                        if scenario_hypotheses:
                            vector_routes = search_with_scores(
                                state,
                                latest_message,
                                self.vector_top_k,
                                allowed_routes=allowed_vector_routes,
                                scenario_hypotheses=scenario_hypotheses,
                            )
                        else:
                            vector_routes = search_with_scores(
                                state,
                                latest_message,
                                self.vector_top_k,
                                allowed_routes=allowed_vector_routes,
                            )
                    else:
                        if scenario_hypotheses:
                            vector_routes = self.vector_index.search_routes(
                                state,
                                latest_message,
                                self.vector_top_k,
                                allowed_routes=allowed_vector_routes,
                                scenario_hypotheses=scenario_hypotheses,
                            )
                        else:
                            vector_routes = self.vector_index.search_routes(
                                state,
                                latest_message,
                                self.vector_top_k,
                                allowed_routes=allowed_vector_routes,
                            )
                else:
                    vector_routes = {}
            except Exception:
                vector_routes = {}
            for route_name, vector_rows in vector_routes.items():
                scenario_route = route_name.startswith("vector_scenario_")
                route_weight = (
                    self.scenario_vector_weight
                    if scenario_route
                    else 0.0
                    if self.vector_recall_only
                    else self._vector_route_weight(route_name)
                )
                if (
                    route_weight <= 0
                    and self.vector_recall_only is False
                    and not scenario_route
                ):
                    continue
                for rank, row in enumerate(vector_rows, start=1):
                    product_id, similarity = self._vector_row(row)
                    if (
                        similarity is not None
                        and similarity < self.vector_min_similarity
                    ):
                        continue
                    if product_id not in self.metadata:
                        continue
                    scores[product_id] += route_weight / (weights.rrf_offset + rank)
                    route_ranks[product_id].append((route_name, rank))

        if not scores:
            return self._fallback_result(top_k)
        if len(scores) > CANDIDATE_LIMIT:
            retained = set(
                sorted(scores, key=lambda item: (-scores[item], item))[
                    :CANDIDATE_LIMIT
                ]
            )
            scores = defaultdict(
                float, {product_id: scores[product_id] for product_id in retained}
            )
            route_ranks = defaultdict(
                list,
                {
                    product_id: route_ranks[product_id]
                    for product_id in retained
                },
            )

        latest_phrase = " ".join(lexical_terms([latest_message]))
        latest_lower = latest_message.lower()
        exact_identifiers = _exact_identifiers(
            [
                latest_message,
                *(
                    value
                    for values in state.preferences.values()
                    for value in values
                ),
                *state.search_terms,
            ]
        )
        session_profile_tags = profile_tags(state.user_profile)
        profile_hits: dict[str, tuple[str, ...]] = {}
        override_message = _is_product_change(state, latest_message)
        category_weight = (
            0.15
            if override_message
            and state.category
            and state.category.lower() not in latest_lower
            else 1.0
        )
        profile_terms = _profile_terms(state)
        components: dict[str, defaultdict[str, float]] = {}
        matched: dict[str, set[str]] = {}
        for product_id in list(scores):
            product = self.metadata[product_id]
            parts: defaultdict[str, float] = defaultdict(float)
            parts["fusion"] = scores[product_id]
            parts["fusion"] += weights.multi_route_bonus * max(
                0, len(route_ranks[product_id]) - 1
            )
            hits: set[str] = set()

            if state.category and state.category.lower() in product["categories"]:
                parts["category"] += weights.category_boost * category_weight
            title_and_category = f"{product['title']} {product['categories']}"
            if latest_phrase and latest_phrase in title_and_category:
                parts["exact_phrase"] += weights.exact_phrase_boost
            for identifier in exact_identifiers:
                if self._identifier_matches(identifier, product):
                    parts["exact_identifier"] += EXACT_IDENTIFIER_BOOST
                    hits.add("feature")
            for attribute, values in state.preferences.items():
                for value in values:
                    if self._preference_matches(attribute, value, product):
                        preference_weight = (
                            0.15
                            if override_message and value.lower() not in latest_lower
                            else 1.0
                        ) * self._value_idf_scale(value)
                        parts[f"preference:{attribute}"] += (
                            weights.confirmed_attribute_boost * preference_weight
                        )
                        hits.add(attribute)
            for attribute, values in state.removed_preferences.items():
                for value in values:
                    if self._preference_matches(attribute, value, product):
                        parts[f"removed:{attribute}"] -= weights.removed_attribute_penalty
            if semantic_profile_enabled():
                # Expanded tags against catalog vocabulary, scaled by intent and
                # recorded by name so the personalization can be explained.
                boost, tags = match_profile(
                    product["corpus"],
                    session_profile_tags,
                    state.intent_mode,
                    self._term_idf_scale,
                )
                if boost:
                    parts["profile"] += boost
                    profile_hits[product_id] = tags
            elif profile_terms:
                corpus = product["corpus"]
                matched_terms = sum(1 for term in profile_terms if term in corpus)
                if matched_terms:
                    parts["profile"] += min(
                        MAX_PROFILE_BOOST, PROFILE_TERM_BOOST * matched_terms
                    )
            parts["rating"] += min(product["average_rating"], 5.0) * weights.rating_coef
            parts["rating_count"] += (
                math.log1p(product["rating_number"]) * weights.rating_count_coef
            )

            components[product_id] = parts
            matched[product_id] = hits
            scores[product_id] = sum(parts.values())

        candidates = tuple(
            RankedCandidate(
                product_id=product_id,
                score=scores[product_id],
                route_ranks=tuple(sorted(route_ranks[product_id])),
                matched_attributes=frozenset(matched[product_id]),
                matched_profile_tags=profile_hits.get(product_id, ()),
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
        self.last_relaxed_constraints: tuple[str, ...] = ()

        def _apply_staged_filter(pool: tuple) -> tuple:
            if not (staged_filter_enabled() and state.intent_mode == "buying"):
                return pool
            filtered, relaxed = staged_filter(
                pool,
                classify_constraints(state),
                self.metadata,
                self._preference_matches,
                _has_attribute_metadata,
            )
            self.last_relaxed_constraints = relaxed
            return filtered or pool

        # Placement matters: ahead of the cross-encoder the reranker sees a
        # different candidate distribution than the one its weights were tuned
        # on; behind it, filtering only removes violators from an ordering the
        # reranker produced normally.
        if not _env_bool("TECHJAM_STAGED_FILTER_AFTER_RERANK", True):
            candidates = _apply_staged_filter(candidates)

        if self.cross_encoder_reranker is not None:
            ranked_ids = [candidate.product_id for candidate in candidates]
            reranked_ids = self.cross_encoder_reranker.rerank(
                state,
                latest_message,
                ranked_ids,
                dict(scores),
                self.metadata,
            )
            candidate_by_id = {candidate.product_id: candidate for candidate in candidates}
            candidates = tuple(
                candidate_by_id[product_id]
                for product_id in reranked_ids
                if product_id in candidate_by_id
            )
        if _env_bool("TECHJAM_STAGED_FILTER_AFTER_RERANK", True):
            candidates = _apply_staged_filter(candidates)
        if self.audience_guardrail:
            candidate_by_id = {candidate.product_id: candidate for candidate in candidates}
            guarded_ids = apply_audience_guardrail(
                [candidate.product_id for candidate in candidates],
                self.metadata,
                state,
                latest_message,
                penalty=self.audience_penalty,
                top_n=self.audience_top_n,
            )
            candidates = tuple(
                candidate_by_id[product_id]
                for product_id in guarded_ids
                if product_id in candidate_by_id
            )
        return SearchResult(
            recommendations=select_diverse_recommendations(
                candidates,
                top_k,
                self.metadata,
                track.category_cap if track is not None and diversity_enabled() else 0,
            ),
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
            disagreement = _signature_disagreement(present)
            relevance = min(1.0, prior + 0.15 * coverage)
            diagnostics[attribute] = AttributeDiagnostic(
                attribute=attribute,
                coverage=coverage,
                disagreement=disagreement,
                relevance=relevance,
            )
        return diagnostics

    def _vector_route_weight(self, route_name: str) -> float:
        if route_name == "vector_category":
            return self.vector_category_weight
        if route_name == "vector_feature":
            return self.vector_feature_weight
        return self.vector_weight

    @staticmethod
    def _vector_row(row: object) -> tuple[str, float | None]:
        if isinstance(row, tuple) and row:
            product_id = str(row[0])
            try:
                return product_id, float(row[1])
            except (IndexError, TypeError, ValueError):
                return product_id, None
        return str(row), None

    def _allowed_vector_routes(
        self,
        state: ShoppingState,
        latest_message: str,
        scores: dict[str, float],
        route_ranks: dict[str, list[tuple[str, int]]],
    ) -> set[str]:
        policy = self.vector_policy
        if policy in {"0", "false", "no", "off"}:
            return set()
        return {
            route_name
            for route_name in (
                "vector_category",
                "vector_feature",
                "vector",
            )
            if self._vector_route_weight(route_name) > 0
            and self._should_use_vector_route(
                route_name,
                state,
                latest_message,
                scores,
                route_ranks,
            )
        }

    def _should_use_vector_route(
        self,
        route_name: str,
        state: ShoppingState,
        latest_message: str,
        scores: dict[str, float],
        route_ranks: dict[str, list[tuple[str, int]]],
    ) -> bool:
        if route_name == "vector_category":
            query = category_query_embedding_text(state, latest_message)
        elif route_name == "vector_feature":
            query = feature_query_embedding_text(state, latest_message)
        else:
            query = latest_message
        if not query:
            return False

        policy = self.vector_policy
        if policy == "always":
            return True

        rare_terms = [
            term
            for term in lexical_terms([query], limit=32)
            if 0 < self._document_frequency.get(term, 0) <= self.vector_max_doc_frequency
        ]
        has_rare_terms = len(rare_terms) >= self.vector_min_rare_terms
        if policy in {"rare", "rare_terms", "rare-term"}:
            return has_rare_terms
        if policy not in {"adaptive", "production"}:
            return has_rare_terms

        browsing = state.intent_mode == "browsing" or _has_browsing_signal(
            latest_message
        )
        override = _is_product_change(state, latest_message)
        strongest_route_count = max(
            (len(ranks) for ranks in route_ranks.values()), default=0
        )
        lexical_low_confidence = (
            len(scores) <= self.vector_low_confidence_candidate_limit
            or strongest_route_count < self.vector_high_confidence_route_count
        )

        if route_name == "vector_category":
            return browsing and lexical_low_confidence
        if route_name == "vector_feature":
            return has_rare_terms or browsing or override or lexical_low_confidence
        return lexical_low_confidence

    def _term_idf_scale(self, term: str) -> float:
        """Rarity of a single term in [0, 1]; 1.0 means highly distinctive."""
        total = len(self.metadata) or 1
        frequency = self._document_frequency.get(term, 0)
        if frequency <= 0:
            return 1.0
        return max(self.idf_min_scale, 1.0 - (frequency / total) * self.idf_damp)

    def _value_idf_scale(self, value: str) -> float:
        """How much ranking force a constraint value deserves.

        A phrase is only as discriminating as its rarest word: "100% Cotton"
        earns its weight from "cotton", not from "100". Values whose rarest
        token still appears across much of the catalogue are damped toward
        `idf_min_scale`, never to zero - a common constraint is weak evidence,
        not wrong evidence.
        """
        if not self.idf_weighting:
            return 1.0
        total = len(self.metadata) or 1
        frequencies = [
            self._document_frequency.get(term, 0)
            for term in lexical_terms([value])
        ]
        frequencies = [f for f in frequencies if f > 0]
        if not frequencies:
            return 1.0
        ratio = min(frequencies) / total
        return max(self.idf_min_scale, 1.0 - ratio * self.idf_damp)

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
            if any(term in value for term in ("around", "about", "approximately")):
                return 0.75 * budget <= price <= 1.25 * budget
            return False
        return value.lower() in _attribute_text(product, attribute).lower()

    @staticmethod
    def _identifier_matches(identifier: str, product: dict[str, Any]) -> bool:
        tokens = TOKEN_RE.findall(identifier.lower())
        if not tokens:
            return False
        needle = " ".join(tokens)
        return needle in " ".join(TOKEN_RE.findall(product["corpus"].lower()))
