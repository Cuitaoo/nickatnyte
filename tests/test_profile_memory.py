from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.preference_tool import (
    PreferencePatch,
    PreferenceValue,
    apply_preference_patch,
)
from starter.profile_memory import (
    JsonProfileStore,
    ProfileUpdate,
    distill_profile_updates,
)
from starter.state import ShoppingState


def updated_state(
    message: str = "I usually prefer cotton jeans.",
    *,
    category: str = "jeans",
    attribute: str = "material",
    value: str = "cotton",
) -> tuple[ShoppingState, ShoppingState]:
    del message
    before = ShoppingState.new("session", {})
    after = apply_preference_patch(
        before,
        PreferencePatch(
            category=category,
            set_preferences=[
                PreferenceValue(attribute=attribute, value=value),
            ],
        ),
    )
    return before, after


class ProfileDistillationTest(unittest.TestCase):
    def test_explicit_durable_preference_emits_scoped_update(self) -> None:
        before, after = updated_state()

        updates = distill_profile_updates(
            "I usually prefer cotton jeans.", before, after, 1
        )

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].category_scope, "jeans")
        self.assertEqual(updates[0].attribute, "material")
        self.assertEqual(updates[0].value, "cotton")
        self.assertEqual(updates[0].source, "explicit_long_term")
        self.assertEqual(
            updates[0].evidence_excerpt,
            "I usually prefer cotton jeans.",
        )

    def test_ordinary_session_preference_emits_nothing(self) -> None:
        before, after = updated_state()

        self.assertEqual(
            distill_profile_updates("I need cotton jeans.", before, after, 1),
            (),
        )

    def test_transient_scope_overrides_durable_marker(self) -> None:
        before, after = updated_state()

        self.assertEqual(
            distill_profile_updates(
                "I usually prefer cotton, but for this one use cotton.",
                before,
                after,
                1,
            ),
            (),
        )

    def test_negated_tendency_does_not_create_positive_memory(self) -> None:
        before, after = updated_state()

        self.assertEqual(
            distill_profile_updates(
                "I do not usually prefer cotton jeans.", before, after, 1
            ),
            (),
        )

    def test_category_scope_is_required(self) -> None:
        before, after = updated_state(category="unchanged")

        self.assertEqual(
            distill_profile_updates(
                "I usually prefer cotton products.", before, after, 1
            ),
            (),
        )

    def test_transient_attributes_are_not_persisted(self) -> None:
        before, after = updated_state(attribute="budget", value="under $50")

        self.assertEqual(
            distill_profile_updates(
                "I usually spend under $50 on jeans.", before, after, 1
            ),
            (),
        )


class JsonProfileStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profiles.json"
        self.store = JsonProfileStore(self.path)
        self.update = ProfileUpdate(
            category_scope="jeans",
            attribute="style",
            value="relaxed fit",
            confidence=0.90,
            source_turn=1,
            evidence_excerpt="I usually prefer relaxed-fit jeans.",
        )

    def test_updates_are_stored_under_external_user_key(self) -> None:
        user = self.store.apply_updates(
            "demo_user_42", "session_a", [self.update]
        )

        self.assertEqual(user["preferences"][0]["support_count"], 1)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("demo_user_42", payload["users"])
        evidence = payload["users"]["demo_user_42"]["preferences"][0]["evidence"]
        self.assertEqual(
            evidence[0]["evidence_excerpt"],
            "I usually prefer relaxed-fit jeans.",
        )

    def test_duplicate_evidence_is_idempotent(self) -> None:
        self.store.apply_updates("demo_user_42", "session_a", [self.update])
        user = self.store.apply_updates("demo_user_42", "session_a", [self.update])

        self.assertEqual(user["preferences"][0]["support_count"], 1)
        self.assertEqual(len(user["preferences"][0]["evidence"]), 1)

    def test_independent_session_increases_support(self) -> None:
        self.store.apply_updates("demo_user_42", "session_a", [self.update])
        user = self.store.apply_updates("demo_user_42", "session_b", [self.update])

        self.assertEqual(user["preferences"][0]["support_count"], 2)
        self.assertGreater(user["preferences"][0]["confidence"], 0.90)

    def test_duplicate_after_multiple_sessions_does_not_inflate_confidence(self) -> None:
        self.store.apply_updates("demo_user_42", "session_a", [self.update])
        before = self.store.apply_updates(
            "demo_user_42", "session_b", [self.update]
        )
        after = self.store.apply_updates(
            "demo_user_42", "session_b", [self.update]
        )

        self.assertEqual(
            after["preferences"][0]["confidence"],
            before["preferences"][0]["confidence"],
        )

    def test_users_remain_isolated(self) -> None:
        self.store.apply_updates("user_a", "session_a", [self.update])

        self.assertEqual(self.store.load_user("user_b"), {"preferences": []})

    def test_blank_user_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.apply_updates("", "session_a", [self.update])
