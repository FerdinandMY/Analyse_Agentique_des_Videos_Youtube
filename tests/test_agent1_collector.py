"""
Tests — A1 Loader / Validator (test_agent1_collector.py)
"""
import os
import tempfile

import pandas as pd
import pytest

from agents.a1_loader import a1_loader, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv(tmp_path, rows: list[dict], filename: str = "comments.csv") -> str:
    path = str(tmp_path / filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestA1LoaderHappyPath:
    def test_loads_minimal_csv(self, tmp_path):
        path = _csv(tmp_path, [{"text": "Great video!"}, {"text": "Love it"}])
        result = a1_loader({"csv_path": path})
        assert "raw_comments" in result
        assert len(result["raw_comments"]) == 2
        assert result["raw_comments"][0]["text"] == "Great video!"

    def test_loads_optional_columns(self, tmp_path):
        path = _csv(tmp_path, [
            {"text": "Nice", "video_id": "abc123", "author_likes": 5, "reply_count": 2}
        ])
        result = a1_loader({"csv_path": path})
        record = result["raw_comments"][0]
        assert record["video_id"] == "abc123"
        assert record["author_likes"] == 5
        assert record["reply_count"] == 2

    def test_strips_empty_text_rows(self, tmp_path):
        path = _csv(tmp_path, [
            {"text": "Valid comment"},
            {"text": ""},
            {"text": "   "},
        ])
        result = a1_loader({"csv_path": path})
        # Empty/whitespace-only rows are dropped
        assert len(result["raw_comments"]) == 1

    def test_required_columns_set(self):
        assert "text" in REQUIRED_COLUMNS

    def test_bypasses_load_when_raw_comments_populated(self, tmp_path):
        """If raw_comments already in state, A1 returns empty dict (no-op)."""
        existing = [{"text": "pre-loaded"}]
        result = a1_loader({"raw_comments": existing, "csv_path": ""})
        assert result == {}

    def test_multiple_rows(self, tmp_path):
        rows = [{"text": f"Comment {i}"} for i in range(20)]
        path = _csv(tmp_path, rows)
        result = a1_loader({"csv_path": path})
        assert len(result["raw_comments"]) == 20


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestA1LoaderErrors:
    def test_missing_csv_path(self):
        result = a1_loader({"csv_path": ""})
        assert "errors" in result
        assert any("csv_path" in e for e in result["errors"])

    def test_file_not_found(self):
        result = a1_loader({"csv_path": "/nonexistent/path/comments.csv"})
        assert "errors" in result
        assert any("not found" in e.lower() for e in result["errors"])

    def test_missing_required_column(self, tmp_path):
        path = _csv(tmp_path, [{"author": "Bob", "likes": 3}])
        result = a1_loader({"csv_path": path})
        assert "errors" in result
        assert any("text" in e for e in result["errors"])

    def test_all_rows_empty_text(self, tmp_path):
        path = _csv(tmp_path, [{"text": ""}, {"text": "   "}])
        result = a1_loader({"csv_path": path})
        # No errors but zero comments loaded
        raw = result.get("raw_comments", [])
        assert len(raw) == 0

    def test_no_csv_path_key(self):
        result = a1_loader({})
        assert "errors" in result
