from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.vector_index import DEFAULT_EMBEDDING_MODEL, build_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached product embeddings.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--index-dir", default="data/vector_index")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = build_vector_index(
        catalog_path=args.catalog,
        index_dir=args.index_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
