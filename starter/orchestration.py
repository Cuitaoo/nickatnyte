"""One explicit strategy decision per turn.

The agent already adapts at runtime - it picks an intent track, enables vector
routes on low lexical confidence, defers weak recommendations, and chooses a
clarification attribute. But those choices were scattered across three modules
and left no trace, so "adaptive orchestration" was real behaviour that nothing
could inspect, test, or explain.

This module collects the choices already made into one record. It decides
nothing itself: adding it cannot change a ranking, which is what makes it safe
to leave switched on. What it buys is that every turn can now answer "what did
you do, and why".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# What the agent did with this turn's candidates.
ACTION_RECOMMEND = "recommend_now"
ACTION_RECOMMEND_ASKING = "recommend_while_asking"
ACTION_ASK_ONLY = "ask_only"
ACTION_EXHAUSTED = "no_candidates"


@dataclass(frozen=True)
class StrategyDecision:
    """Everything the agent decided this turn, in one inspectable record."""

    turn: int
    action: str
    intent_route: str = "unknown"
    track: str = "neutral"
    hard_constraints: tuple[str, ...] = ()
    soft_constraints: tuple[str, ...] = ()
    removed_constraints: tuple[str, ...] = ()
    routes_enabled: tuple[str, ...] = ()
    vector_used: bool = False
    candidate_count: int = 0
    returned_count: int = 0
    top_margin: float = 0.0
    deferred: bool = False
    depth_cap: int | None = None
    next_question: str | None = None
    # Profile tags that actually matched the top candidate. Personalization
    # that cannot be explained should not be applied.
    profile_tags_matched: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        """A one-line human summary, for logs and for explaining to judges."""
        parts = [f"turn {self.turn}", f"{self.intent_route}/{self.track}", self.action]
        if self.hard_constraints:
            parts.append("hard=" + ",".join(self.hard_constraints))
        if self.soft_constraints:
            parts.append("soft=" + ",".join(self.soft_constraints))
        if self.removed_constraints:
            parts.append("removed=" + ",".join(self.removed_constraints))
        if self.profile_tags_matched:
            parts.append("profile=" + ",".join(self.profile_tags_matched))
        if self.routes_enabled:
            parts.append("routes=" + ",".join(self.routes_enabled))
        parts.append(f"pool={self.candidate_count}")
        if self.candidate_count > 1:
            parts.append(f"margin={self.top_margin:.3f}")
        if self.depth_cap is not None:
            parts.append(f"depth={self.depth_cap}")
        if self.next_question:
            parts.append(f"ask={self.next_question}")
        return " | ".join(parts)


def classify_action(
    returned_count: int, ask_attribute: str | None, deferred: bool
) -> str:
    """Name what the turn actually did with its candidates."""
    if deferred or (returned_count == 0 and ask_attribute):
        return ACTION_ASK_ONLY
    if returned_count == 0:
        return ACTION_EXHAUSTED
    if ask_attribute:
        return ACTION_RECOMMEND_ASKING
    return ACTION_RECOMMEND


def constraint_split(state: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate what the shopper stated from what the profile merely suggests.

    Confirmed preferences are hard: the shopper said them. Profile tags are
    soft: they are an aggregate prior and must never act as a constraint.
    Genuine hard/soft typing *within* stated preferences (must-have versus
    nice-to-have) is a larger change - see the buying path in the backlog.
    """
    hard = tuple(
        f"{attribute}={'/'.join(values)}"
        for attribute, values in sorted((state.preferences or {}).items())
        if values
    )
    profile = state.user_profile or {}
    soft = tuple(str(tag) for tag in (profile.get("preference_tags") or []))
    return hard, soft


def removed_split(state: Any) -> tuple[str, ...]:
    return tuple(
        f"{attribute}={'/'.join(values)}"
        for attribute, values in sorted((state.removed_preferences or {}).items())
        if values
    )


def route_summary(candidates: Any) -> tuple[tuple[str, ...], bool]:
    """Which retrieval routes actually contributed, and did vector fire."""
    names: list[str] = []
    for candidate in candidates or ():
        for route_name, _rank in getattr(candidate, "route_ranks", ()):
            if route_name not in names:
                names.append(route_name)
    vector_used = any(name.startswith("vector") for name in names)
    return tuple(sorted(names)), vector_used


def top_margin(candidates: Any) -> float:
    """Score gap between the top two candidates.

    A small margin means the ranking cannot separate its own best guesses,
    which is the signal a confidence controller would act on.
    """
    items = list(candidates or ())
    if len(items) < 2:
        return 0.0
    return float(items[0].score) - float(items[1].score)


def build_decision(
    *,
    state: Any,
    turn: int,
    track_name: str,
    candidates: Any,
    returned_count: int,
    ask_attribute: str | None,
    deferred: bool,
    depth_cap: int | None,
) -> StrategyDecision:
    hard, soft = constraint_split(state)
    routes, vector_used = route_summary(candidates)
    items = list(candidates or ())
    matched_tags = tuple(getattr(items[0], "matched_profile_tags", ()) if items else ())
    return StrategyDecision(
        turn=turn,
        action=classify_action(returned_count, ask_attribute, deferred),
        intent_route=getattr(state, "intent_mode", "unknown"),
        track=track_name,
        hard_constraints=hard,
        soft_constraints=soft,
        removed_constraints=removed_split(state),
        routes_enabled=routes,
        vector_used=vector_used,
        candidate_count=len(list(candidates or ())),
        returned_count=returned_count,
        top_margin=top_margin(candidates),
        deferred=deferred,
        depth_cap=depth_cap,
        next_question=ask_attribute,
        profile_tags_matched=matched_tags,
    )
