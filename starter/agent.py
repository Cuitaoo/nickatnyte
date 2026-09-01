from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from starter.llm_agent import (
    Interpretation,
    InvalidInterpretation,
    PreferenceInterpreter,
)
from starter.orchestration import StrategyDecision, build_decision
from starter.explain import explain_top, explanations_enabled
from starter.confidence import (
    assess,
    confidence_depth,
    depth_mode,
    overload_steering_enabled,
    choose_strategy,
    controller_enabled,
    withholds_recommendations,
)
from starter.profile import (
    profile_reorder_enabled,
    profile_tags,
    reorder_by_profile,
)
from starter.profile_memory import ProfileUpdate, distill_profile_updates
from starter.preference_tool import (
    PreferenceParseDecision,
    apply_preference_patch,
    canonicalize_model_patch,
    parse_preference_fallback,
    preference_parse_decision,
)
from starter.questions import choose_clarification
from starter.query_expansion import ScenarioHypothesis, query_expansion_mode
from starter.reranker import CandidateReranker
from starter.retrieval import CatalogRetriever, RetrievalWeights, _is_product_change
from starter.tracks import dual_track_enabled, resolve_track
from starter.state import ShoppingState
from starter import config


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
        llm_gate_enabled: bool | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = CatalogRetriever(self.catalog_path, weights=weights)
        self._sessions: dict[str, ShoppingState] = {}
        self._decisions: dict[str, StrategyDecision] = {}
        self._parse_decisions: dict[str, PreferenceParseDecision] = {}
        self._state_update_diagnostics: dict[str, dict[str, Any]] = {}
        self._profile_updates: dict[str, tuple[ProfileUpdate, ...]] = {}
        self.defer_low_confidence_recommendations = _env_bool(
            "TECHJAM_DEFER_LOW_CONFIDENCE_RECOMMENDATIONS", True
        )
        auto_interpreter = interpreter is _AUTO_INTERPRETER
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
        self.llm_gate_enabled = (
            _env_bool("OPENAI_AMBIGUITY_GATE_ENABLED", True)
            if llm_gate_enabled is None and auto_interpreter
            else bool(llm_gate_enabled)
            if llm_gate_enabled is not None
            else False
        )

    def close(self) -> None:
        self.retriever.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not session_id:
            raise ValueError("session_id must not be blank")
        self._sessions[session_id] = ShoppingState.new(session_id, user_profile)
        self._parse_decisions.pop(session_id, None)
        self._state_update_diagnostics.pop(session_id, None)
        self._profile_updates.pop(session_id, None)
        reset_method = getattr(self.interpreter, "reset", None)
        if callable(reset_method):
            try:
                reset_method(session_id)
            except Exception:
                pass

    def last_decision(self, session_id: str) -> StrategyDecision | None:
        """The strategy record for this session's most recent turn.

        Deliberately not returned in the response payload: the submission
        contract is exactly message/ask_attribute/recommendations/usage, and
        widening it is not worth the risk. Read it from here for logging,
        debugging, and explaining a turn.
        """
        return self._decisions.get(session_id)

    def last_parse_decision(
        self, session_id: str
    ) -> PreferenceParseDecision | None:
        """Return the latest deterministic LLM-routing decision for diagnostics."""
        return self._parse_decisions.get(session_id)

    def last_state_update_diagnostic(self, session_id: str) -> dict[str, Any] | None:
        """Return how the latest state patch was selected and canonicalized."""
        return self._state_update_diagnostics.get(session_id)

    def profile_updates(self, session_id: str) -> tuple[ProfileUpdate, ...]:
        """Return persistence-ready deltas observed in this anonymous session.

        The official evaluator has no stable user identity, so the Agent never
        writes these itself. An authenticated application layer may persist them
        under its own opaque user key.
        """

        return self._profile_updates.get(session_id, ())

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
        state_before_update = state
        previous_category = state.category
        prompt_tokens = 0
        completion_tokens = 0
        scenario_hypotheses: tuple[ScenarioHypothesis, ...] = ()
        parse_decision = preference_parse_decision(user_message, state)
        self._parse_decisions[session_id] = parse_decision
        use_interpreter = bool(
            self.interpreter is not None
            and (not self.llm_gate_enabled or parse_decision.use_llm)
        )
        update_diagnostic: dict[str, Any] = {
            "llm_requested": use_interpreter,
            "deterministic_patch": parse_decision.patch.model_dump(mode="json"),
            "model_patch": None,
            "applied_patch": None,
            "fallback_error": None,
            "scenario_hypotheses": [],
        }

        if use_interpreter:
            try:
                interpretation = self.interpreter.interpret(user_message, state)
                interpreted_state, prompt_tokens, completion_tokens = (
                    self._validated_interpretation(session_id, interpretation)
                )
                scenario_hypotheses = interpretation.scenario_hypotheses
                update_diagnostic["scenario_hypotheses"] = [
                    {
                        "scenario_query": item.scenario_query,
                        "basis": item.basis,
                        "confidence": item.confidence,
                    }
                    for item in scenario_hypotheses
                ]
                if interpretation.patch is not None:
                    update_diagnostic["model_patch"] = (
                        interpretation.patch.model_dump(mode="json")
                    )
                    canonical_patch = canonicalize_model_patch(
                        user_message,
                        state,
                        interpretation.patch,
                        parse_decision.patch,
                    )
                    update_diagnostic["applied_patch"] = (
                        canonical_patch.model_dump(mode="json")
                    )
                    state = apply_preference_patch(state, canonical_patch)
                else:
                    update_diagnostic["applied_patch"] = "raw_interpreted_state"
                    state = interpreted_state
            except InvalidInterpretation as exc:
                prompt_tokens = exc.prompt_tokens
                completion_tokens = exc.completion_tokens
                update_diagnostic["fallback_error"] = type(exc).__name__
                update_diagnostic["applied_patch"] = (
                    parse_decision.patch.model_dump(mode="json")
                )
                state = apply_preference_patch(state, parse_decision.patch)
            except Exception as exc:
                update_diagnostic["fallback_error"] = type(exc).__name__
                update_diagnostic["applied_patch"] = (
                    parse_decision.patch.model_dump(mode="json")
                )
                state = apply_preference_patch(state, parse_decision.patch)
        else:
            update_diagnostic["applied_patch"] = (
                parse_decision.patch.model_dump(mode="json")
            )
            state = apply_preference_patch(state, parse_decision.patch)
        self._state_update_diagnostics[session_id] = update_diagnostic

        observed_updates = distill_profile_updates(
            str(user_message), state_before_update, state, int(turn)
        )
        if observed_updates:
            existing_updates = list(self._profile_updates.get(session_id, ()))
            for update in observed_updates:
                if update not in existing_updates:
                    existing_updates.append(update)
            self._profile_updates[session_id] = tuple(existing_updates)

        requested_count = min(10, max(0, int(top_k)))
        try:
            if scenario_hypotheses and query_expansion_mode() == "recall":
                search_result = self.retriever.search(
                    state,
                    str(user_message),
                    requested_count,
                    scenario_hypotheses=scenario_hypotheses,
                )
            else:
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

        # Assessed before the question is chosen: over-generality should steer
        # which question gets asked, not merely be recorded afterwards.
        signals = assess(search_result.candidates, state)
        steer = signals.is_overloaded and overload_steering_enabled()
        message, ask_attribute = choose_clarification(
            state,
            int(turn),
            search_result.diagnostics,
            steer,
        )
        rule_would_defer = bool(
            self.defer_low_confidence_recommendations
            and _should_defer_recommendations(
                state,
                int(turn),
                ask_attribute,
                search_result,
                previous_category,
            )
        )
        strategy = None
        if controller_enabled():
            strategy = choose_strategy(signals, ask_attribute, rule_would_defer)
            deferred = withholds_recommendations(strategy)
        else:
            deferred = rule_would_defer
        if deferred:
            identifiers = []
        mode = depth_mode()
        if mode == "confidence":
            # Return everything or nothing: withhold until the ranking has
            # separated, and ask a question in the meantime.
            depth = 10 if (signals.is_confident or int(turn) >= 5) else 0
        elif mode == "hybrid":
            depth = confidence_depth(signals, int(turn))
        else:
            depth = _recommendation_depth(int(turn), state.intent_mode)
        applied_depth = None
        if depth is not None and ask_attribute is not None:
            identifiers = identifiers[:depth]
            applied_depth = depth
            if depth == 0:
                deferred = True

        # Business guardrail final pass. Runs after the depth cap, so the
        # returned set is already frozen: this can only permute it.
        reordered_tags: tuple[str, ...] = ()
        if profile_reorder_enabled() and len(identifiers) > 1:
            identifiers, reordered_tags = reorder_by_profile(
                identifiers,
                self.retriever.metadata,
                profile_tags(state.user_profile),
                state.intent_mode,
            )

        track = (
            resolve_track(state.intent_mode, _is_product_change(state, str(user_message)))
            if dual_track_enabled()
            else None
        )
        decision = build_decision(
            state=state,
            turn=int(turn),
            track_name=track.name if track is not None else "neutral",
            candidates=search_result.candidates,
            returned_count=len(identifiers),
            ask_attribute=ask_attribute,
            deferred=deferred,
            depth_cap=applied_depth,
        )
        if reordered_tags:
            decision = replace(decision, profile_tags_matched=reordered_tags)
        if strategy is not None:
            decision = replace(decision, action=strategy)
        self._decisions[session_id] = decision
        # Transparent explanation, generated from recorded match evidence only.
        # The response contract requires `message` to be a string and nothing
        # parses its content, so this is where a shopper-facing reason belongs.
        if explanations_enabled() and identifiers:
            message = explain_top(
                identifiers, search_result.candidates, state, message
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = config.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(config.getenv(name, str(default)))
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
        raw = config.getenv("TECHJAM_DEPTH_SCHEDULE_BUYING", "").strip()
    if not raw:
        raw = config.getenv("TECHJAM_DEPTH_SCHEDULE", "").strip()
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
