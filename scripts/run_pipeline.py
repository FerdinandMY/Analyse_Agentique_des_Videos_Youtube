from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Allow running from repo root without installing the package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.pipeline import Pipeline


def _load_json_maybe_path(value: str) -> Any:
    # If it looks like a file path, load it; otherwise parse as JSON string.
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph video analysis pipeline.")
    parser.add_argument("--video_ids", type=str, default="[]", help="JSON list of video ids.")
    parser.add_argument("--comments", type=str, default="[]", help="JSON list of comments or a path to a JSON file.")
    parser.add_argument("--thread_id", type=str, default="dev_run_1")
    parser.add_argument("--checkpoint_mode", type=str, default="memory", choices=["memory", "sqlite"])
    parser.add_argument("--sqlite_path", type=str, default="checkpoints.db")
    args = parser.parse_args()

    video_ids = json.loads(args.video_ids)
    comments = _load_json_maybe_path(args.comments)

    pipeline = Pipeline(
        checkpoint_mode=args.checkpoint_mode,
        sqlite_path=args.sqlite_path,
    )

    result = pipeline.run(
        {
            "video_ids": video_ids,
            "comments": comments,
            "thread_id": args.thread_id,
        }
    )

    print(json.dumps({"status": result.status, "data": result.data, "errors": result.errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()