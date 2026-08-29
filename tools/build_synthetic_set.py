"""Generate a synthetic session set from catalog products never used as public targets.

The evaluator derives the intent card and behavior from the target product at
run time (materialize_hidden_fields), so a session only needs the target, a
scenario type, and a user profile. Holding out the *target* is the point: it
makes the set independent of the 200 public samples that every tuned constant
in this repo was fitted on.

User profiles are resampled from the public set rather than invented, since
they are anonymized aggregates drawn from the same population and are not the
variable under test.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluator.local_evaluator import intent_card, load_jsonl

# official mix, matching the public set exactly: 40/40/15/5
SCENARIO_MIX = (("buying", 0.40), ("browsing", 0.40), ("intent_override", 0.15), ("boundary", 0.05))
DIFFICULTY_MIX = (("easy", 0.40), ("medium", 0.45), ("hard", 0.15))


def _allocate(total: int, mix: tuple[tuple[str, float], ...]) -> list[str]:
    counts = {name: int(total * share) for name, share in mix}
    while sum(counts.values()) < total:  # largest-share bucket absorbs rounding
        counts[max(mix, key=lambda item: item[1])[0]] += 1
    out: list[str] = []
    for name, n in counts.items():
        out.extend([name] * n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a held-out synthetic session set.")
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--public", default="data/public_set.jsonl")
    ap.add_argument("--output", default="data/synthetic_set.jsonl")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="synthetic")
    args = ap.parse_args()

    public = load_jsonl(args.public)
    used = {str(s["ground_truth"]["parent_asin"]) for s in public}
    profiles = [s["user_profile"] for s in public]

    products: dict[str, dict] = {}
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin not in used:
                products[parent_asin] = product

    rng = random.Random(args.seed)
    eligible = sorted(products)
    if len(eligible) < args.count:
        raise SystemExit(f"only {len(eligible)} unused products for {args.count} sessions")
    chosen = rng.sample(eligible, args.count)

    scenarios = _allocate(args.count, SCENARIO_MIX)
    difficulties = _allocate(args.count, DIFFICULTY_MIX)
    rng.shuffle(scenarios)
    rng.shuffle(difficulties)

    rows = []
    for index, parent_asin in enumerate(chosen, start=1):
        rows.append({
            "category_bucket": "clothing",
            "difficulty_bucket": difficulties[index - 1],
            "ground_truth": {"parent_asin": parent_asin},
            "sample_id": f"{args.prefix}_{index:04d}",
            "scenario_type": scenarios[index - 1],
            "user_profile": rng.choice(profiles),
        })

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # comparability check: a set whose intent cards are much thinner than the
    # public set's would be harder for reasons unrelated to overfitting.
    def constraint_profile(samples, source):
        counts = [len(set(intent_card(source[str(s["ground_truth"]["parent_asin"])])["hard_constraints"]))
                  for s in samples]
        return sum(counts) / len(counts)

    all_products = dict(products)
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                p = json.loads(line)
                all_products[str(p["parent_asin"])] = p

    print(f"wrote {len(rows)} sessions -> {out}")
    print(f"  scenarios:    {dict((s, scenarios.count(s)) for s in dict.fromkeys(scenarios))}")
    print(f"  difficulty:   {dict((d, difficulties.count(d)) for d in dict.fromkeys(difficulties))}")
    print(f"  pool:         {len(eligible)} catalog products, {len(used)} public targets excluded")
    print(f"  mean hard_constraints  synthetic {constraint_profile(rows, all_products):.2f}"
          f"  vs public {constraint_profile(public, all_products):.2f}")


if __name__ == "__main__":
    main()
