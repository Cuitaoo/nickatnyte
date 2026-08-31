"""Privacy-conscious, persistence-ready long-term preference updates.

The evaluator supplies anonymous sessions, so this module never tries to infer
identity. The agent emits structured updates; an authenticated application may
persist them under its own opaque user key.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from starter.state import ShoppingState


DEFAULT_PROFILE_STORE_PATH = Path("data/long_term_user_profile_updates.json")
SCHEMA_VERSION = 1
PROFILE_ATTRIBUTES = frozenset(
    {"material", "color", "style", "brand", "feature", "use_case"}
)
DURABLE_MARKER_RE = re.compile(
    r"\b(?:usually|generally|typically|normally)\b|"
    r"\bi\s+(?:always|tend\s+to)\b|"
    r"\bmost\s+of\s+the\s+time\b|"
    r"\bmy\s+go-to\b",
    re.IGNORECASE,
)
TRANSIENT_MARKER_RE = re.compile(
    r"\b(?:for\s+this(?:\s+one)?|this\s+time|today|right\s+now|"
    r"for\s+this\s+(?:trip|event|occasion)|as\s+a\s+gift)\b",
    re.IGNORECASE,
)
NEGATED_DURABLE_RE = re.compile(
    r"\b(?:not|never|no\s+longer|don['’]?t|do\s+not)\b.{0,32}"
    r"\b(?:always|usually|generally|typically|normally|prefer|like)\b",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ProfileUpdate:
    """One validated durable preference with bounded source provenance."""

    category_scope: str
    attribute: str
    value: str
    confidence: float
    source_turn: int
    source: Literal["explicit_long_term"] = "explicit_long_term"
    polarity: Literal["prefer"] = "prefer"
    evidence_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(value: object) -> str:
    return SPACE_RE.sub(" ", str(value)).strip(" \t\r\n,.;:-").lower()


def _durable_confidence(message: str) -> float:
    lowered = message.lower()
    if re.search(r"\b(?:always|my\s+go-to)\b", lowered):
        return 0.95
    return 0.90


def _evidence_excerpt(message: str, limit: int = 160) -> str:
    """Keep only the sentence containing the durable marker, never full history."""

    compact = SPACE_RE.sub(" ", str(message)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    selected = next(
        (sentence for sentence in sentences if DURABLE_MARKER_RE.search(sentence)),
        compact,
    )
    return selected[:limit].strip()


def distill_profile_updates(
    message: str,
    before: ShoppingState,
    after: ShoppingState,
    turn: int,
) -> tuple[ProfileUpdate, ...]:
    """Emit durable updates only when explicit wording supports persistence.

    The canonical state reducer remains the source of truth for attribute/value
    extraction. A durable marker alone cannot manufacture a preference.
    """

    if (
        not DURABLE_MARKER_RE.search(message)
        or TRANSIENT_MARKER_RE.search(message)
        or NEGATED_DURABLE_RE.search(message)
    ):
        return ()

    category_scope = _normalize(after.category or before.category or "")
    if not category_scope:
        return ()

    confidence = _durable_confidence(message)
    updates: list[ProfileUpdate] = []
    seen: set[tuple[str, str]] = set()
    for evidence in after.preference_evidence:
        attribute = _normalize(evidence.attribute)
        if evidence.source_turn != turn or attribute not in PROFILE_ATTRIBUTES:
            continue
        for raw_value in evidence.values:
            value = _normalize(raw_value)
            key = (attribute, value)
            if not value or key in seen:
                continue
            seen.add(key)
            updates.append(
                ProfileUpdate(
                    category_scope=category_scope,
                    attribute=attribute,
                    value=value,
                    confidence=confidence,
                    source_turn=max(1, int(turn)),
                    evidence_excerpt=_evidence_excerpt(message),
                )
            )
    return tuple(updates)


class JsonProfileStore:
    """Small production-adapter demo using an externally supplied user key."""

    def __init__(self, path: str | Path = DEFAULT_PROFILE_STORE_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load_user(self, user_key: str) -> dict[str, Any]:
        normalized_key = self._validate_user_key(user_key)
        with self._lock:
            payload = self._read()
            user = payload["users"].get(normalized_key)
            if user is None:
                return {"preferences": []}
            return json.loads(json.dumps(user))

    def apply_updates(
        self,
        user_key: str,
        session_id: str,
        updates: Sequence[ProfileUpdate],
    ) -> dict[str, Any]:
        """Merge updates and return the resulting stored user profile."""

        normalized_key = self._validate_user_key(user_key)
        normalized_session = str(session_id).strip()
        if not normalized_session:
            raise ValueError("session_id must not be blank")

        with self._lock:
            payload = self._read()
            user = payload["users"].setdefault(normalized_key, {"preferences": []})
            preferences = user.setdefault("preferences", [])
            for update in updates:
                self._merge_update(preferences, normalized_session, update)
            preferences.sort(
                key=lambda item: (
                    item["category_scope"],
                    item["attribute"],
                    item["value"],
                )
            )
            self._write(payload)
            return json.loads(json.dumps(user))

    @staticmethod
    def _validate_user_key(user_key: str) -> str:
        normalized = str(user_key).strip()
        if not normalized:
            raise ValueError("user_key must be an opaque non-blank identifier")
        return normalized

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "users": {}}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid profile store: {self.path}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or not isinstance(payload.get("users"), dict)
        ):
            raise ValueError(f"unsupported profile store schema: {self.path}")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    @staticmethod
    def _merge_update(
        preferences: list[dict[str, Any]],
        session_id: str,
        update: ProfileUpdate,
    ) -> None:
        identity = (update.category_scope, update.attribute, update.value)
        record = next(
            (
                item
                for item in preferences
                if (
                    item.get("category_scope"),
                    item.get("attribute"),
                    item.get("value"),
                )
                == identity
            ),
            None,
        )
        if record is None:
            record = {
                "category_scope": update.category_scope,
                "attribute": update.attribute,
                "value": update.value,
                "polarity": update.polarity,
                "confidence": update.confidence,
                "support_count": 0,
                "sources": [],
                "evidence": [],
            }
            preferences.append(record)

        evidence = {
            "session_id": session_id,
            "turn": update.source_turn,
            "source": update.source,
        }
        evidence_identity = (
            evidence["session_id"],
            evidence["turn"],
            evidence["source"],
        )
        existing_evidence = next(
            (
                item
                for item in record["evidence"]
                if (
                    item.get("session_id"),
                    item.get("turn"),
                    item.get("source"),
                )
                == evidence_identity
            ),
            None,
        )
        is_new_evidence = existing_evidence is None
        if update.evidence_excerpt:
            evidence["evidence_excerpt"] = update.evidence_excerpt
        if is_new_evidence:
            record["evidence"].append(evidence)
        elif update.evidence_excerpt and existing_evidence is not None:
            existing_evidence["evidence_excerpt"] = update.evidence_excerpt
        if update.source not in record["sources"]:
            record["sources"].append(update.source)
            record["sources"].sort()
        record["evidence"].sort(
            key=lambda item: (item["session_id"], item["turn"], item["source"])
        )
        record["support_count"] = len(record["evidence"])
        if is_new_evidence:
            record["confidence"] = min(
                0.99,
                max(
                    float(record.get("confidence", 0.0)),
                    update.confidence
                    + 0.03 * max(0, record["support_count"] - 1),
                ),
            )
