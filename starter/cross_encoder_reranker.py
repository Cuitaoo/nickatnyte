from __future__ import annotations

import math
from typing import Any

from starter.state import ShoppingState
from starter import config


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        max_candidates: int = 10,
        batch_size: int = 16,
        weight: float = 0.65,
        local_files_only: bool = True,
        text_format: str = "legacy",
    ) -> None:
        self.model_name = model_name
        self.max_candidates = max(0, int(max_candidates))
        self.batch_size = max(1, int(batch_size))
        self.weight = max(0.0, float(weight))
        self.local_files_only = local_files_only
        self.text_format = text_format.strip().lower() or "legacy"
        self._model = None

    @classmethod
    def from_environment(cls) -> "CrossEncoderReranker | None":
        enabled = config.getenv("TECHJAM_RERANK_ENABLED", "false").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        return cls(
            config.getenv("TECHJAM_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()
            or DEFAULT_RERANK_MODEL,
            max_candidates=_bounded_int(
                config.getenv("TECHJAM_RERANK_TOP_N", "10"), minimum=5, maximum=80
            ),
            batch_size=_bounded_int(
                config.getenv("TECHJAM_RERANK_BATCH_SIZE", "16"), minimum=1, maximum=64
            ),
            weight=_bounded_float(
                config.getenv("TECHJAM_RERANK_WEIGHT", "0.65"), minimum=0.0, maximum=2.0
            ),
            local_files_only=config.getenv("TECHJAM_RERANK_LOCAL_ONLY", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            text_format=config.getenv("TECHJAM_RERANK_TEXT_FORMAT", "legacy"),
        )

    def rerank(
        self,
        state: ShoppingState,
        latest_message: str,
        ranked_ids: list[str],
        scores: dict[str, float],
        metadata: dict[str, dict[str, Any]],
    ) -> list[str]:
        if self.max_candidates <= 0 or self.weight <= 0.0 or not ranked_ids:
            return ranked_ids
        candidates = [item for item in ranked_ids[: self.max_candidates] if item in metadata]
        if len(candidates) <= 1:
            return ranked_ids
        query = _query_text(state, latest_message, self.text_format)
        pairs = [
            (query, _product_text(metadata[product_id], self.text_format))
            for product_id in candidates
        ]
        try:
            raw_scores = self._cross_encoder().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception:
            return ranked_ids
        adjusted = dict(scores)
        normalized = _minmax([float(score) for score in raw_scores])
        for product_id, score in zip(candidates, normalized):
            adjusted[product_id] = adjusted.get(product_id, 0.0) + self.weight * score
        reranked_head = sorted(candidates, key=lambda item: (-adjusted[item], item))
        candidate_set = set(candidates)
        return reranked_head + [item for item in ranked_ids if item not in candidate_set]

    def _cross_encoder(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        return self._model


def _query_text(
    state: ShoppingState, latest_message: str, text_format: str = "legacy"
) -> str:
    if text_format not in {"structured", "legacy"}:
        text_format = "legacy"
    if text_format == "legacy":
        return _legacy_query_text(state, latest_message)

    must_match: list[str] = []
    for attribute, values in sorted(state.preferences.items()):
        if values:
            must_match.append(f"{attribute}: {', '.join(values)}")
    avoid: list[str] = []
    for attribute, values in sorted(state.removed_preferences.items()):
        if values:
            avoid.append(f"{attribute}: {', '.join(values)}")

    parts = [f"Find clothing product for shopper request: {latest_message}"]
    if state.category:
        parts.append(f"Target product category: {state.category}")
    if must_match:
        parts.append(f"Must match shopper preferences: {'; '.join(must_match)}")
    if avoid:
        parts.append(f"Must avoid: {'; '.join(avoid)}")
    if state.search_terms:
        parts.append(f"Additional search terms: {', '.join(state.search_terms)}")
    return "\n".join(parts)


def _legacy_query_text(state: ShoppingState, latest_message: str) -> str:
    parts = [latest_message]
    if state.category:
        parts.append(f"category: {state.category}")
    for attribute, values in sorted(state.preferences.items()):
        if values:
            parts.append(f"{attribute}: {', '.join(values)}")
    for attribute, values in sorted(state.removed_preferences.items()):
        if values:
            parts.append(f"avoid {attribute}: {', '.join(values)}")
    if state.search_terms:
        parts.append(f"search terms: {', '.join(state.search_terms)}")
    return "\n".join(parts)


def _product_text(product: dict[str, Any], text_format: str = "legacy") -> str:
    if text_format not in {"structured", "legacy"}:
        text_format = "legacy"
    if text_format == "legacy":
        return _legacy_product_text(product)

    parts = [
        f"Product title: {product.get('title', '')}",
        f"Product category: {product.get('categories', '')}",
        f"Brand: {product.get('store', '')}",
        f"Product features: {product.get('features', '')}",
        f"Product details: {product.get('details', '')}",
        f"Product description: {product.get('description', '')}",
    ]
    return "\n".join(part for part in parts if part.strip())


def _legacy_product_text(product: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"title: {product.get('title', '')}",
            f"brand: {product.get('store', '')}",
            f"categories: {product.get('categories', '')}",
            f"features: {product.get('features', '')}",
            f"details: {product.get('details', '')}",
            f"description: {product.get('description', '')}",
        ]
    )


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _bounded_int(value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed = minimum
    return min(maximum, max(minimum, parsed))


def _bounded_float(value: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError:
        parsed = minimum
    return min(maximum, max(minimum, parsed))
