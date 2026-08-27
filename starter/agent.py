from __future__ import annotations

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
from starter.retrieval import CatalogRetriever
from starter.state import ShoppingState


_AUTO_INTERPRETER = object()


class Agent:
    """Conversational shopping agent with LLM-assisted preference memory."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        interpreter: object = _AUTO_INTERPRETER,
        openai_enabled: bool | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = CatalogRetriever(self.catalog_path)
        self._sessions: dict[str, ShoppingState] = {}
        if interpreter is not _AUTO_INTERPRETER:
            self.interpreter = interpreter
        elif openai_enabled is False:
            self.interpreter = None
        else:
            self.interpreter = PreferenceInterpreter.from_environment()

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
            identifiers = self.retriever.search(
                state, str(user_message), requested_count
            )
        except Exception:
            identifiers = []
        identifiers = list(dict.fromkeys(str(value) for value in identifiers))[
            :requested_count
        ]

        candidate_products = [
            self.retriever.metadata[parent_asin]
            for parent_asin in identifiers
            if parent_asin in self.retriever.metadata
        ]
        message, ask_attribute = choose_clarification(
            state, int(turn), candidate_products
        )
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
