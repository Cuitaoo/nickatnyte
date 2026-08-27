from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


IntentMode = Literal["buying", "browsing", "unknown"]
EvidenceSource = Literal["unsolicited", "clarification", "correction"]

ALLOWED_PREFERENCE_ATTRIBUTES = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)


@dataclass(frozen=True)
class PreferenceEvidence:
    attribute: str
    values: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    source_turn: int = 0
    source_kind: EvidenceSource = "unsolicited"


@dataclass(frozen=True)
class ShoppingState:
    session_id: str
    user_profile: dict[str, Any]
    intent_mode: IntentMode = "unknown"
    category: str | None = None
    preferences: dict[str, tuple[str, ...]] = field(default_factory=dict)
    removed_preferences: dict[str, tuple[str, ...]] = field(default_factory=dict)
    no_preference_attributes: frozenset[str] = frozenset()
    search_terms: tuple[str, ...] = ()
    preference_evidence: tuple[PreferenceEvidence, ...] = ()
    asked_attributes: tuple[str, ...] = ()
    previous_ask_attribute: str | None = None
    latest_recommendations: tuple[str, ...] = ()
    turn: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def new(cls, session_id: str, user_profile: dict[str, Any]) -> "ShoppingState":
        return cls(session_id=session_id, user_profile=deepcopy(user_profile))

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "intent_mode": self.intent_mode,
            "category": self.category,
            "preferences": {key: list(values) for key, values in self.preferences.items()},
            "removed_preferences": {
                key: list(values) for key, values in self.removed_preferences.items()
            },
            "no_preference_attributes": sorted(self.no_preference_attributes),
            "search_terms": list(self.search_terms),
            "previous_ask_attribute": self.previous_ask_attribute,
        }
