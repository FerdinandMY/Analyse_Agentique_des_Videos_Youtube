"""
Tests — A2 Preprocessor (test_agent2_preprocessor.py)
"""
import pytest

from agents.a2_preprocessor import a2_preprocessor, MIN_CHARS, MAX_CHARS
from utils.text_cleaner import normalize_text


# ---------------------------------------------------------------------------
# normalize_text unit tests
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO World") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("hello   world\t!") == "hello world !"

    def test_strip_edges(self):
        assert normalize_text("  hello  ") == "hello"

    def test_none_returns_none(self):
        assert normalize_text(None) is None

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_already_clean(self):
        assert normalize_text("clean text") == "clean text"

    def test_newline_collapsed(self):
        assert normalize_text("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# A2 Preprocessor happy path
# ---------------------------------------------------------------------------

_SAMPLE_RAW = [
    {"text": "This is a great video!"},
    {"text": "Excellent content, very informative."},
    {"text": "I learned a lot from this tutorial."},
]


class TestA2PreprocessorHappyPath:
    def test_produces_cleaned_comments(self):
        result = a2_preprocessor({"raw_comments": _SAMPLE_RAW})
        assert "cleaned_comments" in result
        assert len(result["cleaned_comments"]) == 3

    def test_cleaned_text_lowercase(self):
        result = a2_preprocessor({"raw_comments": [{"text": "HELLO WORLD"}]})
        assert result["cleaned_comments"][0]["cleaned_text"] == "hello world"

    def test_original_text_preserved(self):
        result = a2_preprocessor({"raw_comments": [{"text": "Original Text"}]})
        assert result["cleaned_comments"][0]["text"] == "Original Text"

    def test_language_field_added(self):
        result = a2_preprocessor({"raw_comments": [{"text": "Hello world, this is English"}]})
        assert "language" in result["cleaned_comments"][0]

    def test_preserves_optional_metadata(self):
        raw = [{"text": "Nice", "video_id": "abc123", "author_likes": 7}]
        result = a2_preprocessor({"raw_comments": raw})
        record = result["cleaned_comments"][0]
        assert record["video_id"] == "abc123"
        assert record["author_likes"] == 7

    def test_accepts_string_items(self):
        result = a2_preprocessor({"raw_comments": ["Simple string comment"]})
        assert len(result["cleaned_comments"]) == 1


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestA2PreprocessorFiltering:
    def test_filters_too_short(self):
        raw = [{"text": "ab"}, {"text": "Good comment here"}]
        result = a2_preprocessor({"raw_comments": raw})
        assert len(result["cleaned_comments"]) == 1

    def test_truncates_too_long(self):
        long_text = "a " * 1100  # > MAX_CHARS when lowercased
        result = a2_preprocessor({"raw_comments": [{"text": long_text}]})
        cleaned = result["cleaned_comments"]
        if cleaned:
            assert len(cleaned[0]["cleaned_text"]) <= MAX_CHARS

    def test_deduplicates_exact(self):
        raw = [
            {"text": "Duplicate comment"},
            {"text": "Duplicate comment"},
            {"text": "Duplicate comment"},
        ]
        result = a2_preprocessor({"raw_comments": raw})
        assert len(result["cleaned_comments"]) == 1

    def test_deduplicates_case_insensitive(self):
        raw = [{"text": "Hello World"}, {"text": "hello world"}]
        result = a2_preprocessor({"raw_comments": raw})
        assert len(result["cleaned_comments"]) == 1

    def test_empty_raw_comments(self):
        result = a2_preprocessor({"raw_comments": []})
        assert result["cleaned_comments"] == []

    def test_missing_raw_comments_key(self):
        result = a2_preprocessor({})
        assert result["cleaned_comments"] == []

    def test_filters_none_text(self):
        raw = [{"text": None}, {"text": "Valid text here"}]
        result = a2_preprocessor({"raw_comments": raw})
        assert len(result["cleaned_comments"]) == 1

    def test_min_chars_constant(self):
        assert MIN_CHARS == 3

    def test_max_chars_constant(self):
        assert MAX_CHARS == 2000
