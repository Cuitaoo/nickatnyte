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
    ap.add_argument("--match-popularity", action="store_true",
                    help="sample targets to match the public set's rating-count distribution")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="synthetic")
    args = ap.parse_args()

    public = load_jsonl(args.public)
    used = {str(s["ground_truth"]["parent_asin"]) for s in public}
    profiles = [s["user_profile"] for s in public]

    public_targets: list[dict] = []
    products: dict[str, dict] = {}
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin in used:
                public_targets.append(product)
            else:
                products[parent_asin] = product

    rng = random.Random(args.seed)
    eligible = sorted(products)
    if len(eligible) < args.count:
        raise SystemExit(f"only {len(eligible)} unused products for {args.count} sessions")

    if args.match_popularity:
        # A uniform draw from the catalogue is NOT how the public set was
        # sampled. Public targets sit at a median of ~6,800 ratings against a
        # catalogue median of 12, because the benchmark anchors on real
        # purchases and real purchases skew popular. A holdout that ignores
        # that is mis-specified on exactly the axis any popularity signal
        # depends on, and will report false negatives for it.
        #
        # Match the public quantile-for-quantile: sort both by rating count and
        # pick, for each public target, an unused product of comparable
        # popularity.
        public_counts = sorted(
            float(t.get("rating_number") or 0.0) for t in public_targets
        )
        by_count = sorted(
            eligible, key=lambda a: float(products[a].get("rating_number") or 0.0)
        )
        counts = [float(products[a].get("rating_number") or 0.0) for a in by_count]
        chosen, taken = [], set()
        import bisect

        step = max(1, len(public_counts) // args.count)
        for target_count in public_counts[::step][: args.count]:
            index = bisect.bisect_left(counts, target_count)
            # Walk outward for the nearest still-unused product.
            for offset in range(len(by_count)):
                for candidate_index in (index + offset, index - offset):
                    if 0 <= candidate_index < len(by_count):
                        asin = by_count[candidate_index]
                        if asin not in taken:
                            taken.add(asin)
                            chosen.append(asin)
                            break
                else:
                    continue
                break
        while len(chosen) < args.count:
            asin = rng.choice(eligible)
            if asin not in taken:
                taken.add(asin)
                chosen.append(asin)
        rng.shuffle(chosen)
    else:
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
