"""Dual-track routing: one pipeline, two operating points.

A shopper who opens with "I'm looking for men jeans, a key requirement is 100%
cotton" has already handed over a hard constraint, and the job is precision -
lock that constraint and do not drift. A shopper who opens with "I'm looking
for women dresses, but I'm still exploring" has handed over almost nothing,
and the job is coverage - show a spread they can react to.

Until now both ran the same additive score with the same weights. This module
resolves the intent into a TrackProfile of multipliers over RetrievalWeights,
plus a diversity setting that only the browsing track uses.

The tracks are two operating points on one pipeline rather than two code
paths: everything stays measurable against the same weights, and an intent
misread degrades to a slightly mistuned ranking instead of a different system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from starter import config


@dataclass(frozen=True)
class TrackProfile:
    """Multipliers over RetrievalWeights, plus browsing-only diversity."""

    name: str
    confirmed_attribute: float = 1.0
    exact_phrase: float = 1.0
    category: float = 1.0
    route_relaxed: float = 1.0
    route_synonym: float = 1.0
    route_attribute: float = 1.0
    profile_scale: float = 1.0
    # Max results per leaf category. 0 disables, so the list is a plain
    # truncation as before.
    category_cap: int = 0


# Buying: the shopper stated a hard constraint, so weight confirmed
# attributes and exact phrases harder and damp the routes that widen the pool.
BUYING = TrackProfile(
    name="buying",
    confirmed_attribute=1.30,
    exact_phrase=1.20,
    category=1.10,
    route_relaxed=0.70,
    route_synonym=0.70,
    route_attribute=1.10,
    profile_scale=0.5,
    category_cap=0,
)

# Browsing: little has been stated, so widen the pool, lean on the profile,
# and spread the returned list across leaf categories instead of returning ten
# near-duplicates of one.
BROWSING = TrackProfile(
    name="browsing",
    confirmed_attribute=0.90,
    exact_phrase=1.00,
    category=1.00,
    route_relaxed=1.25,
    route_synonym=1.30,
    route_attribute=0.95,
    profile_scale=2.0,
    category_cap=3,
)

NEUTRAL = TrackProfile(name="neutral")


def dual_track_enabled() -> bool:
    return _env_bool("TECHJAM_DUAL_TRACK", False)


def diversity_enabled() -> bool:
    return _env_bool("TECHJAM_DUAL_TRACK_DIVERSITY", False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = config.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def browsing_track_enabled() -> bool:
    return _env_bool("TECHJAM_DUAL_TRACK_BROWSING", False)


def resolve_track(intent_mode: str, is_product_change: bool = False) -> TrackProfile:
    """Pick the operating point for this turn's intent.

    A product-change turn stays neutral. `preference_tool` classifies an
    override as buying intent, so tightening would fire exactly when the
    category context has just been rewritten and the confirmed attributes may
    belong to the product the shopper has abandoned.

    The browsing profile is opt-in: browsing is already the strongest scenario,
    so it is left untouched unless deliberately switched on.
    """
    if is_product_change:
        return NEUTRAL
    if intent_mode == "buying":
        return BUYING
    if intent_mode == "browsing" and browsing_track_enabled():
        return BROWSING
    return NEUTRAL


def track_strength() -> float:
    """How far to move from neutral toward the track. 0 disables, 1 is full."""
    try:
        parsed = float(config.getenv("TECHJAM_DUAL_TRACK_STRENGTH", "1.0"))
    except ValueError:
        parsed = 1.0
    return max(0.0, min(1.0, parsed))


def _blend(multiplier: float, strength: float) -> float:
    """Interpolate a multiplier between neutral (1.0) and its full value."""
    return 1.0 + (multiplier - 1.0) * strength


def apply_track(weights: Any, track: TrackProfile, strength: float | None = None) -> Any:
    """Return a copy of `weights` scaled by the track's multipliers.

    Tightening trades Hit@10 for MRR: a sharper buying track finds the target
    higher when it finds it, but can lose one outright. `strength` scales the
    whole profile so that trade can be tuned rather than taken whole.
    """
    if track is NEUTRAL:
        return weights
    if strength is None:
        strength = track_strength()
    if strength <= 0.0:
        return weights
    return replace(
        weights,
        confirmed_attribute_boost=weights.confirmed_attribute_boost
        * _blend(track.confirmed_attribute, strength),
        exact_phrase_boost=weights.exact_phrase_boost * _blend(track.exact_phrase, strength),
        category_boost=weights.category_boost * _blend(track.category, strength),
        route_relaxed=weights.route_relaxed * _blend(track.route_relaxed, strength),
        route_synonym=weights.route_synonym * _blend(track.route_synonym, strength),
        route_attribute=weights.route_attribute * _blend(track.route_attribute, strength),
    )


def leaf_category(meta: dict[str, Any]) -> str:
    """Approximate a product's leaf category.

    `categories` reaches us as one flattened lowercase path, so the leaf is
    the tail of the string. Two tokens keep "denim jeans" and "cargo pants"
    apart without splitting a single category across spellings.
    """
    if not meta:
        return ""
    raw = str(meta.get("categories") or "").strip()
    if not raw:
        return str(meta.get("store") or "")
    return " ".join(raw.split()[-2:])


def select_diverse_recommendations(
    candidates: tuple,
    top_k: int,
    metadata: dict[str, dict[str, Any]] | None = None,
    category_cap: int = 0,
) -> tuple[str, ...]:
    """Take the top `top_k`, optionally capping results per leaf category.

    Capped products are not dropped, only deferred: once the ranking runs out
    of fresh categories they backfill in their original order, so a shopper
    never sees a shorter list for the sake of variety.
    """
    if top_k <= 0 or not candidates:
        return ()
    if category_cap <= 0 or not metadata:
        return tuple(item.product_id for item in candidates[:top_k])

    seen: dict[str, int] = {}
    picked: list[str] = []
    deferred: list[str] = []

    for item in candidates:
        if len(picked) >= top_k:
            break
        key = leaf_category(metadata.get(item.product_id) or {})
        if key and seen.get(key, 0) >= category_cap:
            deferred.append(item.product_id)
            continue
        seen[key] = seen.get(key, 0) + 1
        picked.append(item.product_id)

    for product_id in deferred:
        if len(picked) >= top_k:
            break
        picked.append(product_id)

    return tuple(picked[:top_k])
