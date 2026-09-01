"""Diagnose why a machine does not reproduce the reported score.

Run it before opening a bug: it checks the things that actually differ between
machines and prints what it found, rather than making you guess.

    PYTHONPATH=. python tools/verify_setup.py --catalog /path/to/catalog.jsonl

The tuned configuration is compiled into `starter/config.py`, so a fresh clone
needs no environment set up at all. What this checks is that nothing in your
shell is *overriding* it: an exported variable still wins over the shipped
default, so a stale `.env` sourced out of habit is now the most likely reason
your number differs from ours.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

# Settings that materially move the score. The value is what
# starter/config.py ships; this table exists so the diagnostic fails loudly if
# the two ever drift apart.
EXPECTED = {
    "TECHJAM_RERANK_ENABLED": "true",
    "TECHJAM_VECTOR_ENABLED": "false",
    "TECHJAM_VECTOR_INDEX_MODE": "memory",
    "TECHJAM_DEPTH_MODE": "hybrid",
    "TECHJAM_DEPTH_NORMALIZED_MARGIN": "true",
    "TECHJAM_DEPTH_RATIO_WIDE": "0.30",
    "TECHJAM_DEPTH_RATIO_MID": "0.10",
    "TECHJAM_DEPTH_FLOOR_TURN": "5",
    "TECHJAM_RATING_COUNT_COEF": "0.030",
    "TECHJAM_IDF_WEIGHTING": "true",
    "TECHJAM_AUDIENCE_GUARDRAIL": "true",
    "TECHJAM_AUDIENCE_PENALTY": "0.30",
    "TECHJAM_STAGED_FILTER": "true",
    "TECHJAM_CONFIDENCE_CONTROLLER": "true",
    "TECHJAM_OTHER_AFTER_OVERRIDE": "true",
    "TECHJAM_OTHER_AFTER_NO_PREFERENCE": "2",
    "TECHJAM_EXPLANATIONS": "true",
}

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()
    problems: list[str] = []

    print("=" * 66)
    print("REPRODUCIBILITY CHECK")
    print("=" * 66)

    # 1. commit -------------------------------------------------------------
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        subject = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        print(f"\n[1] commit\n  {OK} {head}  {subject[:52]}")
        if dirty:
            print(f"  {WARN} working tree has uncommitted changes:")
            for line in dirty.splitlines()[:5]:
                print(f"         {line}")
            problems.append("uncommitted local changes may alter the score")
    except Exception:
        print(f"\n[1] commit\n  {WARN} not a git checkout")

    # 2. catalog ------------------------------------------------------------
    print("\n[2] catalog for runtime in-memory embeddings")
    catalog = Path(args.catalog)
    if not catalog.exists():
        print(f"  {BAD} catalog not found at {catalog}")
        print("         data/catalog.jsonl is gitignored - supply it yourself")
        problems.append("catalog missing")
    else:
        with catalog.open(encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())
        if row_count == 50_000:
            print(f"  {OK} {row_count:,} products; embeddings will be built in host RAM")
        else:
            print(f"  {WARN} expected 50,000 products, found {row_count:,}")
            problems.append("catalog row count differs from the measured catalog")
        print(f"  {OK} no uploaded vector-index files are required")

    # 3. effective configuration -------------------------------------------
    print("\n[3] effective configuration (shipped defaults + your shell)")
    from starter.config import DEFAULTS, getenv

    drifted = [
        (key, DEFAULTS.get(key), want)
        for key, want in EXPECTED.items()
        if DEFAULTS.get(key, "").lower() != want.lower()
    ]
    if drifted:
        for key, have, want in drifted:
            print(f"  {BAD} starter/config.py has {key}={have}, this check expects {want}")
            problems.append(f"config.py and verify_setup disagree on {key}")
    else:
        print(f"  {OK} starter/config.py ships all {len(EXPECTED)} score-critical settings")

    overrides = [
        (key, os.environ[key], getenv(key))
        for key in sorted(DEFAULTS)
        if key in os.environ and os.environ[key].strip() != DEFAULTS[key]
    ]
    if not overrides:
        print(f"  {OK} nothing in your shell overrides them")
    else:
        print(f"  {BAD} {len(overrides)} setting(s) overridden by your shell:")
        for key, have, _ in overrides[:8]:
            print(f"         {key}={have}   shipped default is {DEFAULTS[key]}")
        print("         an exported variable wins over the shipped default.")
        print("         unset them (or start a clean shell) to get our number:")
        print("           env -i PATH=\"$PATH\" HOME=\"$HOME\" OPENAI_API_KEY=... \\")
        print("             python -m evaluator.local_evaluator --catalog ...")
        problems.append(f"{len(overrides)} score-critical setting(s) overridden in the shell")

    if not os.getenv("OPENAI_API_KEY"):
        print(f"  {WARN} OPENAI_API_KEY is not set - the agent runs offline (-0.0002)")

    # 4. models -------------------------------------------------------------
    print("\n[4] models (loaded lazily, and failures are silent)")
    try:
        from starter.retrieval import CatalogRetriever

        retriever = CatalogRetriever(catalog)
        if retriever.cross_encoder_reranker is None:
            print(f"  {WARN} cross-encoder not constructed (TECHJAM_RERANK_ENABLED off?)")
        else:
            try:
                retriever.cross_encoder_reranker._cross_encoder()
                print(f"  {OK} cross-encoder loaded")
            except Exception as exc:
                print(f"  {WARN} cross-encoder FAILED to load: {type(exc).__name__}")
                print("         TECHJAM_RERANK_LOCAL_ONLY=true means it will not download.")
                print("         Measured impact is about +0.0004 - not your problem here.")
        if retriever.vector_index is not None:
            row_count = retriever.vector_index.config.get("row_count", "unknown")
            print(
                f"  {OK} vector embeddings ready in host memory "
                f"({row_count} rows, source={retriever.vector_index_status})"
            )
        else:
            print(f"  {WARN} vector embeddings unavailable (worth about 0.0005)")
            if retriever.vector_index_error:
                print(f"         {retriever.vector_index_error}")
        retriever.close()
    except Exception as exc:
        print(f"  {BAD} could not construct the retriever: {type(exc).__name__}: {exc}")
        problems.append("retriever construction failed")

    # verdict ---------------------------------------------------------------
    print("\n" + "=" * 66)
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for item in problems:
            print(f"  - {item}")
    else:
        print("Setup matches. Expected on this configuration:")
        print("  public   0.914274   Hit@10 0.990  MRR 0.900581  MTTC 3.545")
        print("  held out 0.880493   (data/synthetic_pop.jsonl)")
        print("\nStill different? The remaining causes are library versions")
        print("(sentence-transformers / torch change cross-encoder output slightly)")
        print("and the LLM being stochastic - run OPENAI_ENABLED=false to compare")
        print("two deterministic runs.")
    print("=" * 66)


if __name__ == "__main__":
    main()
