"""Exercise scenario-aware query expansion against varied natural phrasing.

This is not a relevance benchmark: the synthetic messages have no hidden
purchase target. It verifies the safety contract around query expansion:
which requests reach the LLM, whether hypotheses are emitted, and whether
inferred language remains temporary rather than mutating confirmed state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from starter.agent import Agent


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    message: str
    expected_llm: bool
    purpose: str


CASES: tuple[ProbeCase, ...] = (
    ProbeCase(
        "broad_destination_goal",
        "I'm looking for a good jacket for Canada.",
        True,
        "Broad destination context may add temporary cold-weather recall hints.",
    ),
    ProbeCase(
        "broad_activity_goal",
        "Search for appropriate shoes for wet weather.",
        True,
        "Scenario language is broad enough for semantic interpretation.",
    ),
    ProbeCase(
        "broad_proper_noun_goal",
        "Find a shirt for Russia.",
        True,
        "A named location is context, not an asserted product feature.",
    ),
    ProbeCase(
        "broad_student_goal",
        "Can you recommend a good backpack for commuting?",
        True,
        "Natural recommendation phrasing should not depend on a fixed prefix.",
    ),
    ProbeCase(
        "long_natural_scenario",
        "I need a good pair of shoes for a rainy commute.",
        True,
        "Semantically broad, but intentionally tests the length guard.",
    ),
    ProbeCase(
        "specific_attribute_request",
        "I'm looking for black leather boots in size 10.",
        False,
        "Explicit product evidence should use the deterministic fast path.",
    ),
    ProbeCase(
        "explicit_budget_request",
        "I need a blue shirt for work under 30 dollars.",
        False,
        "Stated category, color, use case, and budget do not need expansion.",
    ),
    ProbeCase(
        "exact_identifier",
        "Please find model AB-1234 for me.",
        False,
        "Known-item lookup must preserve the exact-ID fast path.",
    ),
    ProbeCase(
        "detailed_catalog_evidence",
        "I'm looking for women's camisoles with 100% cotton, a built-in shelf bra, and adjustable straps.",
        False,
        "Detailed evidence should not be reinterpreted as broad scenario context.",
    ),
)


def _state_summary(agent: Agent, session_id: str) -> dict[str, Any]:
    state = agent.session_state(session_id)
    return {
        "intent_mode": state.intent_mode,
        "category": state.category,
        "preferences": {key: list(values) for key, values in state.preferences.items()},
        "removed_preferences": {
            key: list(values) for key, values in state.removed_preferences.items()
        },
        "search_terms": list(state.search_terms),
    }


def _candidate_effect(
    agent: Agent,
    session_id: str,
    message: str,
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Show whether temporary expansion adds candidates before ranking."""

    if not hypotheses:
        return None
    from starter.query_expansion import ScenarioHypothesis

    state = agent.session_state(session_id)
    validated = tuple(ScenarioHypothesis(**item) for item in hypotheses)
    baseline = agent.retriever.search(state, message, 10)
    expanded = agent.retriever.search(
        state, message, 10, scenario_hypotheses=validated
    )
    baseline_ids = {candidate.product_id for candidate in baseline.candidates}
    added = [
        candidate.product_id
        for candidate in expanded.candidates
        if candidate.product_id not in baseline_ids
    ]
    return {
        "baseline_candidate_count": len(baseline.candidates),
        "expanded_candidate_count": len(expanded.candidates),
        "added_candidate_count": len(added),
        "added_candidate_ids": added[:10],
        "expanded_top_ids": list(expanded.recommendations[:5]),
    }


def run_case(agent: Agent, case: ProbeCase) -> dict[str, Any]:
    session_id = f"query-expansion-{case.case_id}"
    agent.reset(session_id, {"summary": "", "preference_tags": []})
    response = agent.respond(session_id, case.message, turn=1, top_k=10)
    parse_decision = agent.last_parse_decision(session_id)
    diagnostic = agent.last_state_update_diagnostic(session_id) or {}
    recommendations = []
    for item in response.get("recommendations", [])[:5]:
        product_id = str(item.get("parent_asin", ""))
        metadata = agent.retriever.metadata.get(product_id, {})
        recommendations.append(
            {
                "parent_asin": product_id,
                "title": metadata.get("title", ""),
            }
        )
    actual_llm = bool(diagnostic.get("llm_requested"))
    hypotheses = list(diagnostic.get("scenario_hypotheses", []))
    return {
        **asdict(case),
        "actual_llm": actual_llm,
        "route_matches_expectation": actual_llm == case.expected_llm,
        "parse_decision": {
            "use_llm": parse_decision.use_llm if parse_decision else None,
            "safe_case": parse_decision.safe_case if parse_decision else None,
            "reasons": list(parse_decision.reasons) if parse_decision else [],
            "risk_score": parse_decision.risk_score if parse_decision else None,
        },
        "diagnostic": diagnostic,
        "candidate_effect": _candidate_effect(
            agent, session_id, case.message, hypotheses
        ),
        "state": _state_summary(agent, session_id),
        "response": {
            "ask_attribute": response.get("ask_attribute"),
            "recommendation_count": len(response.get("recommendations", [])),
            "top_recommendations": recommendations,
            "usage": response.get("usage", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--output",
        default="results_query_expansion_synthetic.json",
        help="JSON report path.",
    )
    args = parser.parse_args()

    agent = Agent(Path(args.catalog))
    try:
        results = [run_case(agent, case) for case in CASES]
    finally:
        agent.close()

    report = {
        "purpose": "Synthetic safety probe for scenario-aware query expansion.",
        "case_count": len(results),
        "expected_route_matches": sum(
            item["route_matches_expectation"] for item in results
        ),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    for item in results:
        hypotheses = item["diagnostic"].get("scenario_hypotheses", [])
        print(
            f"{item['case_id']}: expected_llm={item['expected_llm']} "
            f"actual_llm={item['actual_llm']} hypotheses={len(hypotheses)} "
            f"match={item['route_matches_expectation']}"
        )


if __name__ == "__main__":
    main()
