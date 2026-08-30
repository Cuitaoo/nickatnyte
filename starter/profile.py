"""Safe personalization from the aggregate session profile.

The evaluator hands us a per-session profile - a handful of preference tags, a
summary, a rating style. There is no long-term user database to build and none
is wanted: the profile is an aggregate prior, not a statement of intent.

The previous treatment was a plain lexical overlap between the tag strings and
the product text, capped at 0.03. That almost never fired, because a shopper
whose profile says "comfort" is not looking for products containing the word
"comfort" - they are looking for cushioned, breathable, relaxed, soft. This
module closes that gap by expanding each tag into the vocabulary the catalog
actually uses, and reports which tags matched so the personalization can be
explained rather than merely applied.

Three rules, from the competition guidance:

- buying: weak tie-breaker only, because the shopper has stated what they want
- browsing: a real reranking signal, because they have not
- never a hard constraint, and never able to change the category

The boost is bounded well below `confirmed_attribute_boost`, so a stated
preference always outranks an inferred one.
"""

from __future__ import annotations

import os

# The public and synthetic sets draw from a closed tag vocabulary. These
# expansions are generic apparel vocabulary, not target-specific terms, so they
# transfer to any set built from the same catalogue.
PROFILE_EXPANSIONS: dict[str, frozenset[str]] = {
    "comfort": frozenset(
        {
            "comfort", "comfortable", "cushioned", "soft", "breathable", "relaxed",
            "cozy", "padded", "lightweight", "plush", "stretch",
        }
    ),
    "fit": frozenset(
        {
            "fit", "fitted", "relaxed", "slim", "regular", "tapered", "stretch",
            "adjustable", "true to size", "tailored",
        }
    ),
    "material": frozenset(
        {
            "cotton", "polyester", "leather", "wool", "denim", "linen", "nylon",
            "spandex", "blend", "fabric", "suede", "canvas",
        }
    ),
    "style": frozenset(
        {
            "classic", "casual", "modern", "vintage", "printed", "floral",
            "striped", "solid", "trendy", "elegant", "everyday",
        }
    ),
    "durability": frozenset(
        {
            "durable", "durability", "rugged", "reinforced", "sturdy", "heavy duty",
            "long lasting", "tough", "workwear", "abrasion",
        }
    ),
    "performance": frozenset(
        {
            "performance", "athletic", "moisture wicking", "quick dry", "breathable",
            "stretch", "training", "active", "sport",
        }
    ),
    "warmth": frozenset(
        {
            "warm", "warmth", "insulated", "fleece", "thermal", "lined", "cozy",
            "quilted", "down",
        }
    ),
    "weather": frozenset(
        {
            "waterproof", "water resistant", "windproof", "weatherproof", "rain",
            "all weather", "repellent",
        }
    ),
    # Deliberately empty: too vague to mean anything about a product.
    "general shopping": frozenset(),
}

# Ceilings on the total profile contribution, by intent. Both sit far below
# RetrievalWeights.confirmed_attribute_boost (~1.30) so a stated preference can
# never be outweighed by an inferred one.
MAX_BOOST_BUYING = 0.04
MAX_BOOST_BROWSING = 0.18
MAX_BOOST_UNKNOWN = 0.08

# Per matched tag, before the ceiling applies.
BOOST_PER_TAG = 0.045


def semantic_profile_enabled() -> bool:
    return _env_bool("TECHJAM_PROFILE_SEMANTIC", False)


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


def profile_tags(user_profile: dict | None) -> tuple[str, ...]:
    """The session's preference tags, normalized and deduplicated."""
    if not user_profile:
        return ()
    raw = user_profile.get("preference_tags") or []
    seen: list[str] = []
    for tag in raw:
        normalized = str(tag).strip().lower()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return tuple(seen)


def max_boost(intent_mode: str) -> float:
    """Personalization ceiling for this intent.

    Buying is deliberately near-zero: the shopper has told us what they want,
    so the profile may only break ties. Browsing is where it earns its keep.
    """
    if intent_mode == "buying":
        return _env_float("TECHJAM_PROFILE_MAX_BUYING", MAX_BOOST_BUYING, 0.0, 1.0)
    if intent_mode == "browsing":
        return _env_float("TECHJAM_PROFILE_MAX_BROWSING", MAX_BOOST_BROWSING, 0.0, 1.0)
    return _env_float("TECHJAM_PROFILE_MAX_UNKNOWN", MAX_BOOST_UNKNOWN, 0.0, 1.0)


def match_profile(
    corpus: str, tags: tuple[str, ...], intent_mode: str = "unknown"
) -> tuple[float, tuple[str, ...]]:
    """Score a product against the profile and name the tags that matched.

    Returns the bounded boost and the matching tag names. The names are the
    point: personalization that cannot be explained should not be applied.
    """
    if not corpus or not tags:
        return 0.0, ()

    matched: list[str] = []
    for tag in tags:
        expansion = PROFILE_EXPANSIONS.get(tag)
        if not expansion:
            continue
        if any(term in corpus for term in expansion):
            matched.append(tag)

    if not matched:
        return 0.0, ()

    per_tag = _env_float("TECHJAM_PROFILE_PER_TAG", BOOST_PER_TAG, 0.0, 1.0)
    boost = min(max_boost(intent_mode), per_tag * len(matched))
    return boost, tuple(matched)
