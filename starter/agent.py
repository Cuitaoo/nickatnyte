from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from starter.llm_agent import (
    Interpretation,
    InvalidInterpretation,
    PreferenceInterpreter,
)
from starter.preference_tool import apply_preference_patch, parse_preference_fallback
from starter.questions import choose_clarification
from starter.reranker import CandidateReranker
from starter.retrieval import CatalogRetriever, RetrievalWeights
from starter.state import ShoppingState


_AUTO_INTERPRETER = object()
_AUTO_RERANKER = object()


class Agent:
    """Conversational shopping agent with LLM-assisted preference memory."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        interpreter: object = _AUTO_INTERPRETER,
        openai_enabled: bool | None = None,
        weights: RetrievalWeights | None = None,
        reranker: object = _AUTO_RERANKER,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = CatalogRetriever(self.catalog_path, weights=weights)
        self._sessions: dict[str, ShoppingState] = {}
        self.defer_low_confidence_recommendations = _env_bool(
            "TECHJAM_DEFER_LOW_CONFIDENCE_RECOMMENDATIONS", True
        )
        if interpreter is not _AUTO_INTERPRETER:
            self.interpreter = interpreter
        elif openai_enabled is False:
            self.interpreter = None
        else:
            self.interpreter = PreferenceInterpreter.from_environment()
        if reranker is not _AUTO_RERANKER:
            self.reranker = reranker
        elif openai_enabled is False or interpreter is not _AUTO_INTERPRETER:
            self.reranker = None
        else:
            self.reranker = CandidateReranker.from_environment()

    def close(self) -> None:
        self.retriever.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not session_id:
            raise ValueError("session_id must not be blank")
        self._sessions[session_id] = ShoppingState.new(session_id, user_profile)
        reset_method = getattr(self.interpreter, "reset", None)
        if callable(reset_method):
            try:
                reset_method(session_id)
            except Exception:
                pass

    def session_state(self, session_id: str) -> ShoppingState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise RuntimeError("reset must be called before respond") from exc

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self.session_state(session_id)
        previous_category = state.category
        prompt_tokens = 0
        completion_tokens = 0

        if self.interpreter is not None:
            try:
                interpretation = self.interpreter.interpret(user_message, state)
                state, prompt_tokens, completion_tokens = self._validated_interpretation(
                    session_id, interpretation
                )
            except InvalidInterpretation as exc:
                prompt_tokens = exc.prompt_tokens
                completion_tokens = exc.completion_tokens
                state = self._fallback_state(user_message, state)
            except Exception:
                state = self._fallback_state(user_message, state)
        else:
            state = self._fallback_state(user_message, state)

        requested_count = min(10, max(0, int(top_k)))
        try:
            search_result = self.retriever.search(
                state, str(user_message), requested_count
            )
        except Exception:
            search_result = self.retriever.fallback(requested_count)

        if self.reranker is not None and len(search_result.candidates) > 1:
            try:
                rerank = self.reranker.rerank(
                    state,
                    str(user_message),
                    search_result.candidates,
                    self.retriever.metadata,
                )
            except Exception:
                rerank = None
            if rerank is not None:
                prompt_tokens += max(0, int(rerank.prompt_tokens))
                completion_tokens += max(0, int(rerank.completion_tokens))
                search_result = replace(
                    search_result, recommendations=tuple(rerank.ordering)
                )

        identifiers = [
            product_id
            for product_id in dict.fromkeys(search_result.recommendations)
            if product_id in self.retriever.metadata
        ][:requested_count]

        message, ask_attribute = choose_clarification(
            state,
            int(turn),
            search_result.diagnostics,
            str(user_message),
        )
        if self.defer_low_confidence_recommendations and _should_defer_recommendations(
            state,
            int(turn),
            ask_attribute,
            search_result,
            previous_category,
        ):
            identifiers = []
        depth = _recommendation_depth(int(turn), state.intent_mode)
        if depth is not None and ask_attribute is not None:
            identifiers = identifiers[:depth]
        asked_attributes = list(state.asked_attributes)
        if ask_attribute and ask_attribute not in asked_attributes:
            asked_attributes.append(ask_attribute)
        state = replace(
            state,
            asked_attributes=tuple(asked_attributes),
            previous_ask_attribute=ask_attribute,
            latest_recommendations=tuple(identifiers),
            turn=int(turn),
            prompt_tokens=state.prompt_tokens + prompt_tokens,
            completion_tokens=state.completion_tokens + completion_tokens,
        )
        self._sessions[session_id] = state

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin} for parent_asin in identifiers
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    @staticmethod
    def _fallback_state(message: str, state: ShoppingState) -> ShoppingState:
        patch = parse_preference_fallback(str(message), state)
        return apply_preference_patch(state, patch)

    @staticmethod
    def _validated_interpretation(
        session_id: str, interpretation: Any
    ) -> tuple[ShoppingState, int, int]:
        if not isinstance(interpretation, Interpretation):
            raise TypeError("interpreter returned an invalid result")
        if interpretation.state.session_id != session_id:
            raise ValueError("interpreter returned state for another session")
        return (
            interpretation.state,
            max(0, int(interpretation.prompt_tokens)),
            max(0, int(interpretation.completion_tokens)),
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


def _should_defer_recommendations(
    state: ShoppingState,
    turn: int,
    ask_attribute: str | None,
    search_result: object,
    previous_category: str | None = None,
) -> bool:
    max_turn = _env_int("TECHJAM_DEFER_MAX_TURN", 3, 1, 10)
    max_non_category_preferences = _env_int(
        "TECHJAM_DEFER_MAX_NON_CATEGORY_PREFERENCES", 1, 0, 8
    )
    min_candidate_count = _env_int("TECHJAM_DEFER_MIN_CANDIDATE_COUNT", 2, 1, 10)
    min_top_route_count = _env_int(
        "TECHJAM_DEFER_MIN_TOP_ROUTE_COUNT_TO_RECOMMEND", 99, 1, 99
    )
    if (
        turn > max_turn
        or ask_attribute is None
        or (
            state.intent_mode == "buying"
            and not _env_bool("TECHJAM_DEFER_INCLUDE_BUYING", False)
        )
        or _category_changed(previous_category, state.category)
    ):
        return False
    candidates = getattr(search_result, "candidates", ())[:10]
    if len(candidates) < min_candidate_count:
        return False
    if any(
        route_name == "fallback"
        for candidate in candidates
        for route_name, _rank in getattr(candidate, "route_ranks", ())
    ):
        return False
    non_category_preferences = sum(
        len(values)
        for attribute, values in state.preferences.items()
        if attribute != "category"
    )
    if non_category_preferences > max_non_category_preferences:
        return False
    top_candidate = candidates[0] if candidates else None
    top_route_count = len(getattr(top_candidate, "route_ranks", ()))
    if top_route_count >= min_top_route_count:
        return False
    allowed = {"browsing", "unknown"}
    if _env_bool("TECHJAM_DEFER_INCLUDE_BUYING", False):
        allowed.add("buying")
    return state.intent_mode in allowed


def _category_changed(previous: str | None, current: str | None) -> bool:
    if not previous or not current:
        return False
    return previous.strip().lower() != current.strip().lower()


def _recommendation_depth(turn: int, intent_mode: str = "unknown") -> int | None:
    """Per-turn cap on returned candidates.

    The evaluator locks reciprocal rank at the FIRST turn the target appears,
    so shipping a deep list early can permanently bank a poor rank. Buying
    sessions are disclosed one hard constraint up front, which makes them
    commit early at a mediocre rank; TECHJAM_DEPTH_SCHEDULE_BUYING gives them
    a separate schedule. Format: comma-separated depths from turn 1, last
    value repeats. Unset disables truncation.
    """
    raw = ""
    if intent_mode == "buying":
        raw = os.getenv("TECHJAM_DEPTH_SCHEDULE_BUYING", "").strip()
    if not raw:
        raw = os.getenv("TECHJAM_DEPTH_SCHEDULE", "").strip()
    if not raw:
        return None
    try:
        depths = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        return None
    if not depths:
        return None
    index = max(1, turn) - 1
    return max(1, min(10, depths[min(index, len(depths) - 1)]))
