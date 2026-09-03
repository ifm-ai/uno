from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .constants import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the pinned Uno base model once.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        cache_dir=str(args.cache_dir),
    )
    print(
        json.dumps(
            {
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_MODEL_REVISION,
                "snapshot_path": path,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
