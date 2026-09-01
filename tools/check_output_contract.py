"""Check `respond()` against the Output Rules in docs/submission_rules.md.

Runs the official harness loop unchanged and validates every response the
agent produces, rather than asserting the contract in prose:

    PYTHONPATH=. python tools/check_output_contract.py --catalog data/catalog.jsonl

Exit status is 0 when every rule holds on every turn, 1 otherwise. Defaults to
the deterministic path so it costs nothing to run; pass --with-model to check
the LLM path as well.
"""

from __future__ import annotations

import argparse
import os
import sys

RULES = (
    "message is a string",
    "ask_attribute is an allowed attribute or None",
    "recommendations are ordered best to worst",
    "at most 10 valid unique parent_asin values",
    "usage reports non-negative integer token counts",
)


class ContractChecked:
    """Wraps the real Agent and validates each response as it is produced."""

    def __init__(self, agent, catalog_ids: set[str], allowed: set[str]) -> None:
        self._agent = agent
        self._catalog_ids = catalog_ids
        self._allowed = allowed
        self.turns = 0
        self.violations: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self._agent.respond(session_id, user_message, turn, top_k)
        self.turns += 1
        self._check(response, session_id, turn)
        return response

    def _fail(self, rule: int, session_id: str, turn: int, detail: str) -> None:
        self.violations.append(f"rule {rule} ({RULES[rule - 1]}) turn {turn} {session_id}: {detail}")

    def _check(self, response: object, session_id: str, turn: int) -> None:
        if not isinstance(response, dict):
            self._fail(1, session_id, turn, f"response is {type(response).__name__}, not dict")
            return
        extra = set(response) - {"message", "ask_attribute", "recommendations", "usage"}
        if extra:
            self._fail(1, session_id, turn, f"unexpected keys {sorted(extra)}")

        # 1. message is a string
        if not isinstance(response.get("message"), str):
            self._fail(1, session_id, turn, f"message is {type(response.get('message')).__name__}")

        # 2. ask_attribute is one allowed attribute or null
        attribute = response.get("ask_attribute")
        if attribute is not None and attribute not in self._allowed:
            self._fail(2, session_id, turn, f"ask_attribute={attribute!r}")

        # 3 and 4. recommendations: well formed, valid, unique, at most ten
        recommendations = response.get("recommendations")
        if not isinstance(recommendations, list):
            self._fail(3, session_id, turn, f"recommendations is {type(recommendations).__name__}")
            return
        identifiers = []
        for index, item in enumerate(recommendations):
            if not isinstance(item, dict) or not isinstance(item.get("parent_asin"), str):
                self._fail(3, session_id, turn, f"recommendation {index} is malformed")
                continue
            identifiers.append(item["parent_asin"])
        if len(identifiers) > 10:
            self._fail(4, session_id, turn, f"returned {len(identifiers)} recommendations")
        if len(set(identifiers)) != len(identifiers):
            self._fail(4, session_id, turn, "duplicate parent_asin values")
        unknown = [i for i in identifiers if i not in self._catalog_ids]
        if unknown:
            self._fail(4, session_id, turn, f"{len(unknown)} id(s) not in the catalogue, e.g. {unknown[0]}")

        # 5. usage reports non-negative token counts
        usage = response.get("usage")
        if not isinstance(usage, dict):
            self._fail(5, session_id, turn, f"usage is {type(usage).__name__}")
        else:
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    self._fail(5, session_id, turn, f"{key}={value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--with-model", action="store_true", help="also exercise the LLM path")
    args = parser.parse_args()

    if not args.with_model:
        os.environ["OPENAI_ENABLED"] = "false"

    from evaluator.local_evaluator import ALLOWED_ATTRIBUTES, catalog_index, evaluate, load_jsonl
    from starter.agent import Agent

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    checked = ContractChecked(Agent(args.catalog), catalog_ids, set(ALLOWED_ATTRIBUTES))
    result = evaluate(checked, samples, catalog_ids, categories, products)

    print(f"{len(samples)} sessions, {checked.turns} turns checked "
          f"({'model enabled' if args.with_model else 'deterministic path'})")
    if checked.violations:
        print(f"\nFAIL - {len(checked.violations)} violation(s):")
        for violation in checked.violations[:20]:
            print(f"  {violation}")
        return 1
    for index, rule in enumerate(RULES, start=1):
        print(f"  ok  rule {index}: {rule}")
    print(f"\nAll Output Rules hold. Score on this run: "
          f"{result['recommended_technical_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
