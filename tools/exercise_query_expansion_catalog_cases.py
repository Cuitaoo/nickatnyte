"""Run catalog-grounded probes for the temporary query-expansion path.

The broad requests deliberately do not name a unique bought item. For those
cases, a passing result means that the model creates a safe, plausible temporary
hypothesis and the selected matching catalog product reaches candidate recall or
the returned Top 10. Detailed and identifier cases verify that the feature does
not run when deterministic evidence is already sufficient.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from starter.agent import Agent
from starter.query_expansion import ScenarioHypothesis


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogProbeCase:
    case_id: str
    target_id: str
    message: str
    expect_expansion: bool
    purpose: str


CASES: tuple[CatalogProbeCase, ...] = (
    CatalogProbeCase(
        "canadian_winter_jacket",
        "B01EFJSGMK",
        "I'm looking for a good men's jacket for Canada.",
        False,
        "A country alone is not enough evidence to infer weather or insulation.",
    ),
    CatalogProbeCase(
        "wet_weather_mens_shoe",
        "B07895D8RR",
        "Search for appropriate men's shoes for wet weather.",
        True,
        "Target is a waterproof, high-traction Columbia men's shoe.",
    ),
    CatalogProbeCase(
        "commuter_laptop_backpack",
        "B003Y3B0C2",
        "Can you recommend a good laptop backpack for commuting?",
        True,
        "Target has a padded laptop sleeve and organizer panel.",
    ),
    CatalogProbeCase(
        "detailed_camisole",
        "B08975T43L",
        "I'm looking for a women's cotton camisole with a shelf bra and adjustable straps.",
        False,
        "Target exactly states the product type, material, shelf bra, and straps.",
    ),
    CatalogProbeCase(
        "exact_bra_model",
        "B00CYNKSTE",
        "Please find model 5006715.",
        False,
        "Target details contain Item model number 5006715.",
    ),
)


def _target_position(ids: list[str], target_id: str) -> int | None:
    try:
        return ids.index(target_id) + 1
    except ValueError:
        return None


def _candidate_snapshot(agent: Agent, state, message: str, hypotheses=()) -> dict[str, Any]:
    result = agent.retriever.search(
        state, message, 10, scenario_hypotheses=hypotheses
    )
    candidate_ids = [item.product_id for item in result.candidates]
    return {
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "top_ids": list(result.recommendations),
    }


def run_case(agent: Agent, case: CatalogProbeCase) -> dict[str, Any]:
    session_id = f"catalog-query-expansion-{case.case_id}"
    agent.reset(session_id, {"summary": "", "preference_tags": []})
    response = agent.respond(session_id, case.message, turn=1, top_k=10)
    state = agent.session_state(session_id)
    diagnostic = agent.last_state_update_diagnostic(session_id) or {}
    decision = agent.last_parse_decision(session_id)
    hypotheses = tuple(
        ScenarioHypothesis(**item)
        for item in diagnostic.get("scenario_hypotheses", [])
    )
    baseline = _candidate_snapshot(agent, state, case.message)
    expanded = _candidate_snapshot(agent, state, case.message, hypotheses)
    baseline_ids = set(baseline["candidate_ids"])
    response_ids = [
        str(item.get("parent_asin", "")) for item in response.get("recommendations", [])
    ]
    target = agent.retriever.metadata[case.target_id]
    message_terms = set(TOKEN_RE.findall(case.message.lower()))
    inferred_terms = {
        term
        for item in hypotheses
        for term in TOKEN_RE.findall(item.scenario_query.lower())
        if term not in message_terms
    }
    return {
        **asdict(case),
        "target": {
            "title": target.get("title", ""),
            "categories": target.get("categories", ""),
        },
        "actual_llm": bool(diagnostic.get("llm_requested")),
        "expansion_emitted": bool(hypotheses),
        "route_reasons": list(decision.reasons) if decision else [],
        "hypotheses": [
            {
                "scenario_query": item.scenario_query,
                "basis": item.basis,
                "confidence": item.confidence,
            }
            for item in hypotheses
        ],
        "temporary_state_safety": {
            "generated_hypothesis_terms_in_search_terms": any(
                term in " ".join(state.search_terms).lower()
                for term in inferred_terms
            ),
            "inferred_terms": sorted(inferred_terms),
            "confirmed_preferences": {
                attribute: list(values) for attribute, values in state.preferences.items()
            },
        },
        "target_recall": {
            "in_baseline_candidates": case.target_id in baseline_ids,
            "in_expanded_candidates": case.target_id in set(expanded["candidate_ids"]),
            "added_only_by_expansion": (
                case.target_id not in baseline_ids
                and case.target_id in set(expanded["candidate_ids"])
            ),
            "baseline_candidate_rank": _target_position(
                baseline["candidate_ids"], case.target_id
            ),
            "expanded_candidate_rank": _target_position(
                expanded["candidate_ids"], case.target_id
            ),
            "baseline_top_10_rank": _target_position(baseline["top_ids"], case.target_id),
            "expanded_top_10_rank": _target_position(expanded["top_ids"], case.target_id),
            "returned_rank": _target_position(response_ids, case.target_id),
        },
        "response": {
            "ask_attribute": response.get("ask_attribute"),
            "returned_ids": response_ids,
            "usage": response.get("usage", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--output", default="results_query_expansion_catalog_cases.json"
    )
    args = parser.parse_args()

    agent = Agent(Path(args.catalog))
    try:
        results = [run_case(agent, case) for case in CASES]
    finally:
        agent.close()
    report = {
        "purpose": "Catalog-grounded query-expansion behavior probe.",
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in results:
        target = item["target_recall"]
        print(
            f"{item['case_id']}: expansion={item['expansion_emitted']} "
            f"baseline={target['baseline_top_10_rank']} "
            f"expanded={target['expanded_top_10_rank']} "
            f"returned={target['returned_rank']}"
        )


if __name__ == "__main__":
    main()
