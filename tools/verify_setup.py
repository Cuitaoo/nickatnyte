"""Diagnose why a machine does not reproduce the reported score.

Run it before opening a bug: it checks the four things that actually differ
between machines and prints what it found, rather than making you guess.

    PYTHONPATH=. python tools/verify_setup.py --catalog /path/to/catalog.jsonl

The score is produced by `.env.example`, not by `.env`. `.env` is gitignored,
so yours is your own - if you have an old one lying around and source it, you
get your old settings and a different number. That is the single most common
cause of "I can't reproduce this".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Settings that .env.example sets deliberately and that materially move the
# score. Value is what the winning configuration expects.
EXPECTED = {
    "TECHJAM_RERANK_ENABLED": "true",
    "TECHJAM_VECTOR_ENABLED": "true",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    print("\n[2] catalog vs prebuilt vector index")
    catalog = Path(args.catalog)
    index_config = Path("data/vector_index/config.json")
    if not catalog.exists():
        print(f"  {BAD} catalog not found at {catalog}")
        print("         data/catalog.jsonl is gitignored - supply it yourself")
        problems.append("catalog missing")
    elif not index_config.exists():
        print(f"  {BAD} data/vector_index/config.json missing")
        problems.append("vector index missing")
    else:
        expected = json.loads(index_config.read_text())["catalog_sha256"]
        actual = sha256(catalog)
        if actual == expected:
            print(f"  {OK} sha256 {actual[:16]} matches the index")
        else:
            print(f"  {BAD} catalog sha256 does NOT match the vector index")
            print(f"         index expects {expected[:16]}")
            print(f"         yours is      {actual[:16]}")
            print("         a different catalogue changes every number")
            problems.append("catalog differs from the one the index was built on")

    # 3. effective configuration -------------------------------------------
    print("\n[3] effective environment (what your shell actually has)")
    example = parse_env_file(Path(".env.example"))
    unset, wrong = [], []
    for key, want in EXPECTED.items():
        have = os.getenv(key)
        if have is None:
            unset.append(key)
        elif have.strip().lower() != want.lower():
            wrong.append((key, have, want))
    if not unset and not wrong:
        print(f"  {OK} all {len(EXPECTED)} score-critical settings match .env.example")
    for key, have, want in wrong:
        print(f"  {BAD} {key}={have}   expected {want}")
        problems.append(f"{key} is {have}, expected {want}")
    if unset:
        print(f"  {BAD} {len(unset)} setting(s) not exported at all: {', '.join(unset[:6])}")
        print("         you probably sourced .env (yours) instead of .env.example (ours)")
        print("         run:  set -a; source .env.example; export OPENAI_API_KEY=...; set +a")
        problems.append("score-critical settings are unset")

    if Path(".env").exists():
        own = parse_env_file(Path(".env"))
        drift = [
            k for k, v in own.items()
            if k in EXPECTED and v.strip().lower() != EXPECTED[k].lower()
        ]
        if drift:
            print(f"  {WARN} your .env disagrees with .env.example on: {', '.join(drift[:6])}")
            print("         .env is gitignored and is NOT the reference configuration")

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
        print(
            f"  {OK} vector index loaded"
            if retriever.vector_index is not None
            else f"  {WARN} vector index not loaded (worth about 0.0005)"
        )
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
        print("  public   0.914524   Hit@10 0.990  MRR 0.901415  MTTC 3.545")
        print("  held out 0.876643   (data/synthetic_pop.jsonl)")
        print("\nStill different? The remaining causes are library versions")
        print("(sentence-transformers / torch change cross-encoder output slightly)")
        print("and OPENAI_ENABLED - the LLM is stochastic, so run it off to compare.")
    print("=" * 66)


if __name__ == "__main__":
    main()
