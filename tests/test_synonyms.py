from __future__ import annotations

import unittest

from starter.synonyms import expand_terms


class SynonymTest(unittest.TestCase):
    def test_expands_known_synonyms(self) -> None:
        self.assertIn("sneaker", expand_terms(["trainers"]))
        self.assertIn("sweatshirt", expand_terms(["hoodie"]))

    def test_returns_only_new_terms(self) -> None:
        self.assertNotIn("hoodie", expand_terms(["hoodie"]))

    def test_unknown_terms_expand_to_nothing(self) -> None:
        self.assertEqual(expand_terms(["zzzz"]), ())

    def test_expansion_is_deterministic(self) -> None:
        self.assertEqual(
            expand_terms(["hoodie", "trainers"]), expand_terms(["hoodie", "trainers"])
        )


if __name__ == "__main__":
    unittest.main()
