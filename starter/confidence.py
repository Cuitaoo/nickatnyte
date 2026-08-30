"""Confidence-driven strategy switching.

Deferral today is a rule about turn number, intent, and how many preferences
have been collected. It never looks at whether the ranking is any good. So a
session where the top candidate is streets ahead of the rest is withheld for
the same three turns as one where the top ten are indistinguishable.

That costs on the metric that pays for it. TechnicalScore weights Efficiency at
0.20 and Efficiency is `(11 - MTTC) / 10`, so every turn of delay before the
target first appears costs 0.02. Withholding a list that already contains the
target is the most expensive thing the agent can do.

This module measures how well-separated the ranking actually is and lets a
confident turn recommend immediately. The signals are the ones the rubric asks
for - top-1/top-2 margin, candidate pool size, route agreement, and how much of
the shopper's stated constraints the leader satisfies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Strategies the controller can select.
RECOMMEND_NOW = "recommend_now"
RECOMMEND_WHILE_ASKING = "recommend_while_asking"
ASK_ONLY = "ask_only"
# Selected but not yet actioned: both need a second retrieval pass, which costs
# latency. The controller names them so the decision is visible and testable.
BROADEN_RETRIEVAL = "broaden_retrieval"
RELAX_CONSTRAINT = "relax_constraint"


@dataclass(frozen=True)
class ConfidenceSignals:
    """How separable this turn's ranking is."""

    margin: float = 0.0
    pool_size: int = 0
    route_agreement: int = 0
    constraint_coverage: float = 0.0
    has_fallback: bool = False

    @property
    def is_confident(self) -> bool:
        """Well-separated leader, backed by more than one route."""
        return (
            self.margin >= min_margin()
            and self.route_agreement >= min_route_agreement()
            and not self.has_fallback
        )

    @property
    def is_starved(self) -> bool:
        """Too few candidates to rank at all - broadening is the only move."""
        return self.pool_size <= max_starved_pool()


def controller_enabled() -> bool:
    return _env_bool("TECHJAM_CONFIDENCE_CONTROLLER", False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(os.getenv(name, str(default)))
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def min_margin() -> float:
    """Score gap that counts as a well-separated leader.

    Observed margins collapse to ~0.003 once a session is genuinely ambiguous
    and sit above 0.2 when it is not, so the threshold has a wide gap to land
    in.
    """
    return _env_float("TECHJAM_CONFIDENCE_MIN_MARGIN", 0.05, 0.0, 10.0)


def min_route_agreement() -> int:
    return _env_int("TECHJAM_CONFIDENCE_MIN_ROUTES", 2, 1, 12)


def max_starved_pool() -> int:
    return _env_int("TECHJAM_CONFIDENCE_STARVED_POOL", 3, 0, 50)


def assess(candidates: Any, state: Any) -> ConfidenceSignals:
    """Measure separability of the current ranking."""
    items = list(candidates or ())
    if not items:
        return ConfidenceSignals()

    top = items[0]
    margin = float(top.score) - float(items[1].score) if len(items) > 1 else float(top.score)
    route_ranks = getattr(top, "route_ranks", ())
    has_fallback = any(
        name == "fallback"
        for candidate in items[:10]
        for name, _rank in getattr(candidate, "route_ranks", ())
    )

    stated = {
        attribute
        for attribute, values in (getattr(state, "preferences", None) or {}).items()
        if values and attribute != "category"
    }
    matched = set(getattr(top, "matched_attributes", frozenset()))
    coverage = len(stated & matched) / len(stated) if stated else 0.0

    return ConfidenceSignals(
        margin=margin,
        pool_size=len(items),
        route_agreement=len(route_ranks),
        constraint_coverage=coverage,
        has_fallback=has_fallback,
    )


def choose_strategy(
    signals: ConfidenceSignals,
    ask_attribute: str | None,
    rule_would_defer: bool,
) -> str:
    """Pick this turn's strategy from measured confidence.

    The controller only ever *releases* a turn the existing rule would have
    withheld, and only when the ranking is demonstrably well separated. It
    never withholds a turn the rule would have released, so it cannot make
    MTTC worse than the rule alone.
    """
    if signals.is_starved:
        return BROADEN_RETRIEVAL if ask_attribute else RECOMMEND_NOW

    if rule_would_defer:
        if signals.is_confident:
            # The ranking has separated. Withholding it only delays the hit.
            return RECOMMEND_WHILE_ASKING if ask_attribute else RECOMMEND_NOW
        return ASK_ONLY

    return RECOMMEND_WHILE_ASKING if ask_attribute else RECOMMEND_NOW


def withholds_recommendations(strategy: str) -> bool:
    return strategy in {ASK_ONLY, BROADEN_RETRIEVAL}
