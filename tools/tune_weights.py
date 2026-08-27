"""Random-search tuner for RetrievalWeights against the public session set.

Splits the labeled sessions into stratified train/holdout partitions, runs the
deterministic offline evaluator per sampled weight configuration on the train
split, then scores the best train configurations on the holdout split. The
catalog index is built once and shared across trials.

Usage:
    OPENAI_ENABLED=false python -m tools.tune_weights \
        --catalog data/catalog.jsonl --dataset data/public_set.jsonl \
        --trials 60 --seed 7 --output docs/evaluations/weight-tuning.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import RetrievalWeights


MULTIPLIER_LOW = 0.5
MULTIPLIER_HIGH = 2.0
TOP_TRIALS_FOR_HOLDOUT = 5


def stratified_split(
    samples: list[dict], holdout_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministically split samples per scenario_type; order is preserved
    within each partition by original dataset order."""
    rng = random.Random(seed)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_scenario[str(sample["scenario_type"])].append(sample)
    holdout_ids: set[str] = set()
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        count = max(1, round(len(group) * holdout_fraction)) if group else 0
        chosen = rng.sample(range(len(group)), min(count, len(group)))
        holdout_ids.update(str(group[index]["sample_id"]) for index in chosen)
    train = [item for item in samples if str(item["sample_id"]) not in holdout_ids]
    holdout = [item for item in samples if str(item["sample_id"]) in holdout_ids]
    return train, holdout


def sample_weights(rng: random.Random) -> RetrievalWeights:
    """Sample each weight from a log-uniform multiplier band around its default."""
    defaults = RetrievalWeights()
    sampled: dict[str, float] = {}
    for field in fields(RetrievalWeights):
        default_value = float(getattr(defaults, field.name))
        if default_value == 0:
            sampled[field.name] = 0.0
            continue
        multiplier = math.exp(
            rng.uniform(math.log(MULTIPLIER_LOW), math.log(MULTIPLIER_HIGH))
        )
        sampled[field.name] = default_value * multiplier
    return RetrievalWeights(**sampled)


def _score(
    agent: Agent,
    weights: RetrievalWeights,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> float:
    agent.retriever.weights = weights
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return float(result["recommended_technical_score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="RetrievalWeights random search")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--output", default="docs/evaluations/weight-tuning.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    train, holdout = stratified_split(samples, args.holdout_fraction, args.seed)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, openai_enabled=False)
    rng = random.Random(args.seed)

    def score(weights: RetrievalWeights, split: list[dict]) -> float:
        return _score(agent, weights, split, catalog_ids, categories, products)

    defaults = RetrievalWeights()
    default_train = score(defaults, train)
    default_holdout = score(defaults, holdout)
    print(
        f"default weights: train={default_train:.6f} holdout={default_holdout:.6f}",
        flush=True,
    )

    trials: list[dict] = []
    for index in range(args.trials):
        weights = sample_weights(rng)
        train_score = score(weights, train)
        trials.append(
            {"trial": index, "weights": asdict(weights), "train": train_score}
        )
        print(f"trial {index:03d}: train={train_score:.6f}", flush=True)

    trials.sort(key=lambda item: -item["train"])
    for entry in trials[:TOP_TRIALS_FOR_HOLDOUT]:
        entry["holdout"] = score(RetrievalWeights(**entry["weights"]), holdout)
        print(
            f"trial {entry['trial']:03d}: train={entry['train']:.6f} "
            f"holdout={entry['holdout']:.6f}",
            flush=True,
        )

    evaluated = [entry for entry in trials if "holdout" in entry]
    best = max(evaluated, key=lambda item: item["holdout"]) if evaluated else None
    report = {
        "seed": args.seed,
        "trials_requested": args.trials,
        "split_sizes": {"train": len(train), "holdout": len(holdout)},
        "default": {
            "weights": asdict(defaults),
            "train": default_train,
            "holdout": default_holdout,
        },
        "trials": trials,
        "best": best,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if best is not None:
        improvement = best["holdout"] - default_holdout
        print(
            f"best trial {best['trial']}: holdout={best['holdout']:.6f} "
            f"(default {default_holdout:.6f}, delta {improvement:+.6f})"
        )


if __name__ == "__main__":
    main()
