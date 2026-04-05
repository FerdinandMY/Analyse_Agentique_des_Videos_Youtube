"""
utils/checkpoint.py — JSON Checkpoint Utility
==============================================
Saves the output of each agent to a JSON file after execution.
This allows the pipeline to resume from a checkpoint if the Kaggle session
is interrupted (NFR-03).

Checkpoints are written to: checkpoints/<agent_name>_<pipeline_id>.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_CHECKPOINT_DIR = Path("checkpoints")

# Pipeline run ID set once per run (can be overridden via set_pipeline_id)
_pipeline_id: str = str(int(time.time()))


def set_pipeline_id(pipeline_id: str) -> None:
    """Override the default timestamp-based pipeline ID."""
    global _pipeline_id
    _pipeline_id = pipeline_id


def save_checkpoint(agent_name: str, data: Any) -> None:
    """
    Persist agent output to disk as JSON.

    Args:
        agent_name: Name of the agent (e.g. "a3_sentiment").
        data:       Serialisable dict to save.
    """
    try:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = _CHECKPOINT_DIR / f"{agent_name}_{_pipeline_id}.json"
        path.write_text(
            json.dumps({"agent": agent_name, "pipeline_id": _pipeline_id, "data": data}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Never crash the pipeline due to checkpoint failure.
        pass


def load_checkpoint(agent_name: str, pipeline_id: str | None = None) -> Any:
    """
    Load a previously saved checkpoint.

    Args:
        agent_name:  Name of the agent.
        pipeline_id: Specific run ID. Defaults to current _pipeline_id.

    Returns:
        Parsed JSON dict, or None if not found.
    """
    pid = pipeline_id or _pipeline_id
    path = _CHECKPOINT_DIR / f"{agent_name}_{pid}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
