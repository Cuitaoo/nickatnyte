"""Demonstrate production profile writeback without evaluator identity leakage."""

from __future__ import annotations

import argparse
import json

from starter.preference_tool import (
    PreferencePatch,
    PreferenceValue,
    apply_preference_patch,
)
from starter.profile_memory import JsonProfileStore, distill_profile_updates
from starter.state import ShoppingState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-key", default="demo_user_42")
    parser.add_argument("--session-id", default="demo_session_1")
    parser.add_argument(
        "--output",
        default="data/long_term_user_profile_updates.json",
    )
    args = parser.parse_args()

    message = "I usually prefer relaxed-fit jeans."
    before = ShoppingState.new(args.session_id, {})
    after = apply_preference_patch(
        before,
        PreferencePatch(
            category="jeans",
            set_preferences=[
                PreferenceValue(attribute="style", value="relaxed fit"),
            ],
        ),
    )
    updates = distill_profile_updates(message, before, after, turn=1)
    stored = JsonProfileStore(args.output).apply_updates(
        args.user_key,
        args.session_id,
        updates,
    )

    print(
        json.dumps(
            {
                "user_key": args.user_key,
                "session_id": args.session_id,
                "emitted_updates": [update.to_dict() for update in updates],
                "stored_profile": stored,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
