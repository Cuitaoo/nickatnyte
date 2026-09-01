"""Curated clothing-vocabulary synonym groups for lexical-mismatch hedging.

Retrieval is purely lexical, so a shopper who says "trainers" never reaches a
catalog that says "sneaker". Expansion returns only terms NOT already present
in the input, and is used by a deliberately low-weight retrieval route so it
can surface candidates without outranking direct lexical evidence.
"""

from __future__ import annotations

from typing import Iterable

_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"sneaker", "sneakers", "trainer", "trainers", "shoe", "shoes"}),
    frozenset({"hoodie", "hoodies", "sweatshirt", "sweatshirts", "pullover"}),
    frozenset({"jacket", "jackets", "coat", "coats", "parka", "windbreaker"}),
    frozenset({"tee", "tshirt", "shirt", "shirts", "top", "tops", "blouse"}),
    frozenset({"pants", "trousers", "slacks", "chinos"}),
    frozenset({"jumper", "sweater", "sweaters", "cardigan", "knitwear"}),
    frozenset({"purse", "handbag", "tote"}),
    frozenset({"jeans", "denim"}),
    frozenset({"boot", "boots", "booties"}),
    frozenset({"sandal", "sandals", "slides"}),
    frozenset({"leggings", "tights"}),
    frozenset({"dress", "dresses", "gown", "sundress"}),
    frozenset({"cap", "hat", "beanie"}),
    frozenset({"sock", "socks", "hosiery"}),
    frozenset({"underwear", "briefs", "boxers", "panties", "lingerie"}),
    frozenset({"swimsuit", "swimwear", "bikini", "trunks"}),
    frozenset({"scarf", "scarves", "shawl"}),
    frozenset({"glove", "gloves", "mittens"}),
)

_INDEX: dict[str, frozenset[str]] = {
    term: group for group in _GROUPS for term in group
}


def expand_terms(terms: Iterable[str]) -> tuple[str, ...]:
    """Return synonyms of the given terms that are not themselves present."""
    seen = {str(term).lower() for term in terms}
    result: list[str] = []
    for term in sorted(seen):
        for synonym in sorted(_INDEX.get(term, frozenset())):
            if synonym not in seen and synonym not in result:
                result.append(synonym)
    return tuple(result)
