"""Tune only the product-quality tie-breaker weights.

This keeps every retrieval/reranking setting fixed except:

- RetrievalWeights.rating_coef
- RetrievalWeights.rating_count_coef

Use this instead of broad random search when testing whether rating/popularity
should be stronger. Report validation performance, not just full-public score.

Example:
    source .env
    OPENAI_ENABLED=false python -m tools.tune_quality_weights \
        --catalog /path/to/catalog.jsonl \
        --dataset data/public_set.jsonl \
        --rating-coefs 0,0.0017,0.005,0.01,0.02 \
        --rating-count-coefs 0,0.00033,0.001,0.005,0.01,0.02 \
        --seeds 7,11,17 \
        --output docs/evaluations/quality-weight-tuning.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import RetrievalWeights
from tools.tune_weights import stratified_split


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _score(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    weights: RetrievalWeights,
) -> dict:
    agent.retriever.weights = weights
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return {
        "score": float(result["recommended_technical_score"]),
        "hit": float(result["hit_rate_at_10"]),
        "mrr": float(result["mrr"]),
        "mttc": float(result["mttc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search only rating_coef and rating_count_coef"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--rating-coefs",
        default="0,0.0017193930814524492,0.005,0.01,0.02,0.04",
    )
    parser.add_argument(
        "--rating-count-coefs",
        default="0,0.0003346942922951214,0.001,0.005,0.01,0.02",
    )
    parser.add_argument("--seeds", default="7,11,17")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--output", default="docs/evaluations/quality-weight-tuning.json")
    parser.add_argument(
        "--score-full",
        action="store_true",
        help="Also score every grid point on the full public set. Use only for reporting, not selection.",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, openai_enabled=False)
    defaults = RetrievalWeights()
    rating_coefs = _floats(args.rating_coefs)
    rating_count_coefs = _floats(args.rating_count_coefs)
    seeds = _ints(args.seeds)

    rows: list[dict] = []
    try:
        for rating_coef in rating_coefs:
            for rating_count_coef in rating_count_coefs:
                weights = replace(
                    defaults,
                    rating_coef=rating_coef,
                    rating_count_coef=rating_count_coef,
                )
                split_scores: list[dict] = []
                for seed in seeds:
                    train, holdout = stratified_split(
                        samples, args.holdout_fraction, seed
                    )
                    train_metrics = _score(
                        agent, train, catalog_ids, categories, products, weights
                    )
                    holdout_metrics = _score(
                        agent, holdout, catalog_ids, categories, products, weights
                    )
                    split_scores.append(
                        {
                            "seed": seed,
                            "train": train_metrics,
                            "holdout": holdout_metrics,
                        }
                    )
                row = {
                    "rating_coef": rating_coef,
                    "rating_count_coef": rating_count_coef,
                    "holdout_score_mean": mean(
                        item["holdout"]["score"] for item in split_scores
                    ),
                    "holdout_mrr_mean": mean(
                        item["holdout"]["mrr"] for item in split_scores
                    ),
                    "holdout_hit_mean": mean(
                        item["holdout"]["hit"] for item in split_scores
                    ),
                    "holdout_mttc_mean": mean(
                        item["holdout"]["mttc"] for item in split_scores
                    ),
                    "train_score_mean": mean(
                        item["train"]["score"] for item in split_scores
                    ),
                    "splits": split_scores,
                    "weights": asdict(weights),
                }
                if args.score_full:
                    row["full_public"] = _score(
                        agent, samples, catalog_ids, categories, products, weights
                    )
                rows.append(row)
                print(
                    "rating_coef={:.8g} rating_count_coef={:.8g} "
                    "holdout={:.6f} mrr={:.6f} hit={:.3f} mttc={:.3f}".format(
                        rating_coef,
                        rating_count_coef,
                        row["holdout_score_mean"],
                        row["holdout_mrr_mean"],
                        row["holdout_hit_mean"],
                        row["holdout_mttc_mean"],
                    ),
                    flush=True,
                )
    finally:
        agent.close()

    rows.sort(key=lambda item: (-item["holdout_score_mean"], item["rating_count_coef"]))
    default_row = next(
        (
            row
            for row in rows
            if row["rating_coef"] == defaults.rating_coef
            and row["rating_count_coef"] == defaults.rating_count_coef
        ),
        None,
    )
    report = {
        "selection_rule": (
            "Choose by repeated stratified holdout mean. Use full_public only as "
            "a sanity check because it can overfit the visible 200 sessions."
        ),
        "dataset_size": len(samples),
        "seeds": seeds,
        "holdout_fraction": args.holdout_fraction,
        "default": {
            "rating_coef": defaults.rating_coef,
            "rating_count_coef": defaults.rating_count_coef,
            "row": default_row,
        },
        "best_by_holdout": rows[0] if rows else None,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if rows:
        best = rows[0]
        print(
            "best_by_holdout: rating_coef={:.8g} rating_count_coef={:.8g} "
            "score={:.6f}".format(
                best["rating_coef"],
                best["rating_count_coef"],
                best["holdout_score_mean"],
            ),
            flush=True,
        )
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
