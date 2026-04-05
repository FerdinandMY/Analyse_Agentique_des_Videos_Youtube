"""
main.py — CLI Entry Point
==========================
Run the full YouTube Quality Analyzer pipeline from the command line.

Usage examples:

    # Analyse from a CSV file
    python main.py --csv_path data/raw/comments.csv --video_id dQw4w9WgXcQ --topic "machine learning"

    # Analyse from pre-loaded comments (JSON list of strings)
    python main.py --comments '["Great video!", "Very informative"]' --video_id abc123 --topic "python"

    # Save report to file
    python main.py --csv_path data/raw/comments.csv --output report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph import run_pipeline
from utils.logger import get_logger

logger = get_logger("main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube Quality Analyzer — Multi-agent LLM pipeline"
    )
    parser.add_argument("--csv_path", default="", help="Path to pre-collected comments CSV")
    parser.add_argument("--video_id", default="", help="YouTube video ID (metadata)")
    parser.add_argument("--topic", default="", help="User thematic query for A7 Topic Matcher")
    parser.add_argument(
        "--comments",
        default="",
        help="Pre-loaded comments as a JSON list of strings (bypasses CSV loading)",
    )
    parser.add_argument("--thread_id", default="cli-run", help="LangGraph thread ID")
    parser.add_argument("--output", default="", help="Output file path for the JSON report")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    raw_comments = None
    if args.comments:
        try:
            raw_comments = json.loads(args.comments)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --comments JSON: %s", exc)
            sys.exit(1)

    logger.info(
        "Starting pipeline — video_id=%s topic=%s csv=%s",
        args.video_id or "(none)",
        args.topic or "(none)",
        args.csv_path or "(none)",
    )

    report = run_pipeline(
        csv_path=args.csv_path,
        video_id=args.video_id,
        topic=args.topic,
        raw_comments=raw_comments,
        thread_id=args.thread_id,
    )

    output_str = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        logger.info("Report saved to %s", args.output)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
