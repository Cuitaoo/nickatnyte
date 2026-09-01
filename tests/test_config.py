"""The shipped configuration is the reference, so it gets pinned like one."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from starter import config

STARTER = Path(__file__).resolve().parent.parent / "starter"


class ShippedConfigurationTest(unittest.TestCase):
    def test_api_key_is_never_a_default(self) -> None:
        # A secret must come from the environment. Shipping one - even an
        # empty string - would let a committed value reach a scoring run.
        self.assertNotIn("OPENAI_API_KEY", config.DEFAULTS)
        for name, value in config.DEFAULTS.items():
            self.assertNotIn("sk-", value, f"{name} looks like it holds a key")

    def test_environment_still_wins(self) -> None:
        name = "TECHJAM_DEPTH_MODE"
        self.assertEqual(config.getenv(name), "hybrid")
        with patch.dict(os.environ, {name: "turn"}):
            self.assertEqual(config.getenv(name), "turn")

    def test_unknown_name_falls_through_to_the_call_site(self) -> None:
        self.assertIsNone(config.getenv("TECHJAM_NOT_A_REAL_SETTING"))
        self.assertEqual(
            config.getenv("TECHJAM_NOT_A_REAL_SETTING", "fallback"), "fallback"
        )

    def test_every_default_is_actually_read(self) -> None:
        # A default nobody reads is a lie about what the agent does.
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in STARTER.glob("*.py")
            if path.name != "config.py"
        )
        referenced = set(re.findall(r"\"((?:TECHJAM|OPENAI)_[A-Z0-9_]+)\"", source))
        orphaned = sorted(set(config.DEFAULTS) - referenced)
        self.assertEqual(orphaned, [], f"defaults nothing reads: {orphaned}")

    def test_starter_reads_configuration_through_one_door(self) -> None:
        # os.getenv in starter/ would bypass the shipped defaults silently.
        offenders = [
            path.name
            for path in STARTER.glob("*.py")
            if path.name != "config.py"
            and re.search(r"\bos\.(getenv|environ\.get)\(", path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [], f"bypassing starter.config: {offenders}")


if __name__ == "__main__":
    unittest.main()
