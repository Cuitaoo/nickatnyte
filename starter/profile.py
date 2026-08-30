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
from typing import Callable

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

# Positional denominator for the reorder. Fixed rather than the list length so
# one position is always worth 1/10 and the ceiling reads as a stable number of
# places, however few products came back.
REORDER_SLOTS = 10

# The scoring boosts are sized for the additive retrieval score; in the
# reorder they are read in position units, where one place costs 1/10. Without
# a gain a single matched tag (0.045) cannot climb even one place and the pass
# silently does nothing. At 2.5, browsing's 0.18 ceiling buys ~4.5 places and
# buying's 0.015 buys ~0.4 - a genuine tie-breaker, exactly as intended.
REORDER_GAIN = 2.5


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
    corpus: str,
    tags: tuple[str, ...],
    intent_mode: str = "unknown",
    idf_scale: "Callable[[str], float] | None" = None,
) -> tuple[float, tuple[str, ...]]:
    """Score a product against the profile and name the tags that matched.

    Returns the bounded boost and the matching tag names. The names are the
    point: personalization that cannot be explained should not be applied.

    `idf_scale` maps a term to how rare it is, in [0, 1]. Without it every tag
    counts the same, which is why the first version of this failed: "comfort"
    matches 51.9% of the catalogue and "material" 60.8%, so the commonest tags
    - also the commonest in the data - acted as a near-uniform boost that added
    noise rather than ranking. With it, a tag earns its weight from the rarest
    expansion term it actually matched on, so "weather" matching "waterproof"
    counts for far more than "comfort" matching "soft".
    """
    if not corpus or not tags:
        return 0.0, ()

    matched: list[str] = []
    total = 0.0
    per_tag = _env_float("TECHJAM_PROFILE_PER_TAG", BOOST_PER_TAG, 0.0, 1.0)
    for tag in tags:
        expansion = PROFILE_EXPANSIONS.get(tag)
        if not expansion:
            continue
        hits = [term for term in expansion if term in corpus]
        if not hits:
            continue
        matched.append(tag)
        # The rarest term that matched is the informative one.
        weight = max((idf_scale(term) for term in hits), default=1.0) if idf_scale else 1.0
        total += per_tag * weight

    if not matched:
        return 0.0, ()
    return min(max_boost(intent_mode), total), tuple(matched)


def profile_reorder_enabled() -> bool:
    return _env_bool("TECHJAM_PROFILE_REORDER", False)


def reorder_by_profile(
    identifiers: list[str],
    metadata: dict,
    tags: tuple[str, ...],
    intent_mode: str = "unknown",
    idf_scale: "Callable[[str], float] | None" = None,
) -> tuple[list[str], tuple[str, ...]]:
    """Reorder an already-selected list without changing what is in it.

    Applied after selection and after the depth cap, so the returned set is
    frozen before this runs. Hit@10 and MTTC therefore cannot move - the same
    products appear on the same turn - and only the rank of the target within
    the list can change. That confines the whole feature to the 0.30 MRR term
    and removes its exposure to the 0.50 Hit@10 and 0.20 Efficiency terms.

    Position i is scored (REORDER_SLOTS - i) / REORDER_SLOTS, normalized by a
    fixed denominator rather than the list length: with the length, a short
    list spaces positions so far apart that the ceiling can never overtake
    anything, and the feature silently does nothing. Ties keep their original
    order.

    Returns the new order and the tags that matched whatever ended up first.
    """
    if len(identifiers) < 2 or not tags:
        return identifiers, ()

    scored: list[tuple[float, int, str, tuple[str, ...]]] = []
    for index, product_id in enumerate(identifiers):
        corpus = str((metadata.get(product_id) or {}).get("corpus") or "")
        boost, matched = match_profile(corpus, tags, intent_mode, idf_scale)
        base = (REORDER_SLOTS - index) / REORDER_SLOTS
        gain = _env_float("TECHJAM_PROFILE_REORDER_GAIN", REORDER_GAIN, 0.0, 20.0)
        scored.append((base + boost * gain, index, product_id, matched))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [product_id for _, _, product_id, _ in scored], scored[0][3]
