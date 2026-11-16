"""Lightweight tests for the benchmark helpers.

The expensive benchmarks live in ``benchmarks/`` and are intentionally not
collected by pytest. These tests only verify that the corpus generator and
quality metrics behave deterministically.
"""

import json
import os

from benchmarks.corpus import CORPUS_VERSION, ensure_corpus
from benchmarks.metrics import text_contains_phrase, word_error_rate


def test_word_error_rate_identical_is_zero():
    words = ["alpha", "beta", "gamma"]
    assert word_error_rate(words, words) == 0.0


def test_word_error_rate_counts_substitutions_insertions_deletions():
    assert word_error_rate(["a", "b"], ["a", "x"]) == 0.5
    assert word_error_rate(["a", "b"], ["a"]) == 0.5
    assert word_error_rate(["a"], ["a", "b"]) == 1.0


def test_text_contains_phrase_normalizes_whitespace():
    assert text_contains_phrase("The\nquarterly\nreport", ["The", "quarterly"])
    assert not text_contains_phrase("The report", ["quarterly", "report"])


def test_small_corpus_is_generated_with_manifest(tmp_path):
    cache = tmp_path / "corpus"
    items = ensure_corpus(cache_dir=str(cache), scale=0.2)

    assert [item.name for item in items] == [
        "a4_clean", "a4_two_column", "receipt", "a4_hard_shadow",
    ]
    for item in items:
        assert os.path.exists(item.path)
        assert item.expected_words

    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == CORPUS_VERSION
    assert manifest["scale"] == 0.2
    assert len(manifest["items"]) == 4
