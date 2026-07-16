"""
Unit tests for the pure logic in this project: date normalization and
field scoring (task/task_definition.py), JSON extraction from raw model
output (baseline/run_baseline.py), and the baseline-vs-optimized diff
(evaluation/compare_final.py).

No LLM calls are made in any of these tests -- optimizer/optimize_prompt.py's
proposal step and every script's actual Ollama calls are a manual/
integration concern, the same split used throughout this portfolio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "task"))
sys.path.insert(0, str(Path(__file__).parent.parent / "baseline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from task_definition import normalize_date, score_field, score_extraction  # noqa: E402
from run_baseline import parse_prediction  # noqa: E402
from compare_final import build_diff  # noqa: E402


# --- normalize_date() tests ---

def test_normalize_date_handles_iso_format():
    assert normalize_date("2024-06-03") == "2024-06-03"


def test_normalize_date_handles_us_slash_format():
    assert normalize_date("07/15/2024") == "2024-07-15"


def test_normalize_date_handles_month_name_with_year():
    assert normalize_date("June 3, 2024") == "2024-06-03"


def test_normalize_date_returns_none_for_unparseable_input():
    assert normalize_date("sometime last week") is None


def test_normalize_date_returns_none_when_year_is_missing():
    # "March 22" alone has no year -- shouldn't silently default to 1900
    assert normalize_date("March 22") is None


# --- score_field() tests ---

def test_score_field_amount_matches_within_tolerance():
    assert score_field("amount", 14.50, 14.50) == 1.0
    assert score_field("amount", "14.50", 14.50) == 1.0  # string numbers should still parse


def test_score_field_amount_mismatch_scores_zero():
    assert score_field("amount", 20.00, 14.50) == 0.0


def test_score_field_amount_handles_non_numeric_gracefully():
    assert score_field("amount", "twenty dollars", 20.00) == 0.0


def test_score_field_date_normalizes_before_comparing():
    assert score_field("date", "06/03/2024", "2024-06-03") == 1.0


def test_score_field_category_is_case_insensitive():
    assert score_field("category", "Food", "food") == 1.0


def test_score_field_vendor_allows_partial_match():
    assert score_field("vendor", "the electric company", "electric company") == 1.0


def test_score_field_vendor_mismatch_scores_zero():
    assert score_field("vendor", "Chipotle", "Zara") == 0.0


def test_score_field_returns_zero_for_none_prediction():
    assert score_field("vendor", None, "Chipotle") == 0.0


# --- score_extraction() tests ---

def test_score_extraction_all_fields_correct():
    predicted = {"vendor": "Chipotle", "amount": 14.50, "date": "2024-06-03", "category": "food"}
    expected = {"vendor": "Chipotle", "amount": 14.50, "date": "2024-06-03", "category": "food"}
    assert score_extraction(predicted, expected) == 1.0


def test_score_extraction_partial_credit():
    predicted = {"vendor": "Chipotle", "amount": 99.00, "date": "2024-06-03", "category": "shopping"}
    expected = {"vendor": "Chipotle", "amount": 14.50, "date": "2024-06-03", "category": "food"}
    # vendor correct, date correct, amount and category wrong -> 2/4
    assert score_extraction(predicted, expected) == 0.5


def test_score_extraction_returns_zero_for_non_dict_prediction():
    assert score_extraction(None, {"vendor": "x", "amount": 1, "date": "2024-01-01", "category": "food"}) == 0.0
    assert score_extraction("not json", {"vendor": "x", "amount": 1, "date": "2024-01-01", "category": "food"}) == 0.0


# --- parse_prediction() tests ---

def test_parse_prediction_extracts_clean_json():
    raw = '{"vendor": "Chipotle", "amount": 14.5, "date": "2024-06-03", "category": "food"}'
    result = parse_prediction(raw)
    assert result == {"vendor": "Chipotle", "amount": 14.5, "date": "2024-06-03", "category": "food"}


def test_parse_prediction_extracts_json_surrounded_by_prose():
    raw = 'Sure, here is the extracted data:\n{"vendor": "Zara", "amount": 64.99, "date": "2024-11-05", "category": "shopping"}\nLet me know if you need anything else!'
    result = parse_prediction(raw)
    assert result["vendor"] == "Zara"


def test_parse_prediction_returns_none_when_no_json_present():
    assert parse_prediction("I'm not sure how to extract that.") is None


def test_parse_prediction_returns_none_for_malformed_json():
    assert parse_prediction('{"vendor": "Zara", "amount":}') is None


# --- build_diff() tests ---

def make_outcome(scores):
    return {
        "results": [
            {"input": f"example {i}", "expected": {}, "predicted": {}, "score": s}
            for i, s in enumerate(scores)
        ]
    }


def test_build_diff_flags_improved_examples():
    baseline = make_outcome([0.0, 0.5, 1.0])
    optimized = make_outcome([1.0, 0.5, 1.0])
    diff = build_diff(baseline, optimized)
    assert diff[0]["status"] == "improved"
    assert diff[1]["status"] == "unchanged"
    assert diff[2]["status"] == "unchanged"


def test_build_diff_flags_regressed_examples():
    baseline = make_outcome([1.0])
    optimized = make_outcome([0.5])
    diff = build_diff(baseline, optimized)
    assert diff[0]["status"] == "regressed"
