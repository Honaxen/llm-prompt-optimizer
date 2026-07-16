"""
The fixed task this whole project optimizes a prompt for: extracting
structured expense data from a short, informally-written sentence.

Input:  "Grabbed lunch with Sarah at Chipotle for $14.50 on June 3rd"
Output: {"vendor": "Chipotle", "amount": 14.50, "date": "2024-06-03", "category": "food"}

This task was chosen because prompt wording actually matters for it:
dates arrive in inconsistent formats, amounts are sometimes written out
("twenty bucks"), and category isn't stated explicitly -- it has to be
inferred from context. A vague prompt ("extract the expense info") and a
precise one (exact output schema, explicit category list, date format
requirement) will score very differently on the same model. That gap is
exactly what optimizer/ is meant to close automatically.

Everything here is pure data and pure scoring logic -- no LLM calls.
baseline/ and optimizer/ both import from this module.
"""

from datetime import datetime

CATEGORIES = ["food", "transport", "shopping", "entertainment", "utilities", "other"]

# Each example: an informally-written sentence, and the ground-truth
# structured fields it should extract to. Dates are stored normalized
# (YYYY-MM-DD) as the target -- part of what a good prompt needs to do
# is get the model to normalize dates itself, not just copy them verbatim.
EVAL_EXAMPLES = [
    {"input": "Grabbed lunch with Sarah at Chipotle for $14.50 on June 3rd",
     "expected": {"vendor": "Chipotle", "amount": 14.50, "date": "2024-06-03", "category": "food"}},
    {"input": "Paid the electric bill, $89.23, due 07/15/2024",
     "expected": {"vendor": "electric company", "amount": 89.23, "date": "2024-07-15", "category": "utilities"}},
    {"input": "Uber ride home last night cost about twenty bucks, March 22",
     "expected": {"vendor": "Uber", "amount": 20.00, "date": "2024-03-22", "category": "transport"}},
    {"input": "Bought a new jacket at Zara for $64.99 yesterday (2024-11-05)",
     "expected": {"vendor": "Zara", "amount": 64.99, "date": "2024-11-05", "category": "shopping"}},
    {"input": "Movie tickets for two, AMC theater, $28, Feb 14th",
     "expected": {"vendor": "AMC", "amount": 28.00, "date": "2024-02-14", "category": "entertainment"}},
    {"input": "Coffee at Blue Bottle this morning, 5 dollars 50 cents, 9/1",
     "expected": {"vendor": "Blue Bottle", "amount": 5.50, "date": "2024-09-01", "category": "food"}},
    {"input": "Gas station fill-up, $52.10, on the 12th of April",
     "expected": {"vendor": "gas station", "amount": 52.10, "date": "2024-04-12", "category": "transport"}},
    {"input": "Water bill payment of $34 on 2024-08-20",
     "expected": {"vendor": "water company", "amount": 34.00, "date": "2024-08-20", "category": "utilities"}},
    {"input": "Got groceries at Trader Joe's, came out to $76.34, October 9",
     "expected": {"vendor": "Trader Joe's", "amount": 76.34, "date": "2024-10-09", "category": "shopping"}},
    {"input": "Concert ticket, Ticketmaster, ninety-five dollars, on 5/30/2024",
     "expected": {"vendor": "Ticketmaster", "amount": 95.00, "date": "2024-05-30", "category": "entertainment"}},
    {"input": "Dinner at that new Thai place, $41.20, last Friday which was Jan 26",
     "expected": {"vendor": "Thai place", "amount": 41.20, "date": "2024-01-26", "category": "food"}},
    {"input": "Parking downtown, $15, December 3rd",
     "expected": {"vendor": "parking", "amount": 15.00, "date": "2024-12-03", "category": "transport"}},
]


def normalize_date(raw: str) -> str | None:
    """
    Attempts to parse a variety of date formats into YYYY-MM-DD.
    Returns None if parsing fails -- a failed parse counts as a wrong
    answer for that field, not a crash.

    Formats without a year (e.g. "March 22") are skipped entirely rather
    than parsed with an assumed default year -- Python's own strptime
    behavior for yearless dates is documented as ambiguous and changing
    in 3.15, so this avoids relying on it at all instead of working
    around the deprecation warning.
    """
    raw = raw.strip()
    formats_with_year = [
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y",
        "%b %d, %Y", "%d %B %Y",
    ]
    for fmt in formats_with_year:
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Yearless formats ("March 22", "3/22") are recognized as *shaped like*
    # a date but deliberately not resolved to a specific year -- treated
    # the same as unparseable, since a date without a year isn't something
    # this task's scoring can compare against a fixed expected date anyway.
    return None


def score_field(field: str, predicted, expected) -> float:
    if predicted is None:
        return 0.0

    if field == "amount":
        try:
            return 1.0 if abs(float(predicted) - float(expected)) < 0.01 else 0.0
        except (TypeError, ValueError):
            return 0.0

    if field == "date":
        normalized_predicted = normalize_date(str(predicted))
        return 1.0 if normalized_predicted == expected else 0.0

    if field == "category":
        return 1.0 if str(predicted).strip().lower() == str(expected).strip().lower() else 0.0

    if field == "vendor":
        # Vendor names are fuzzier by nature ("electric company" vs "the
        # electric company") -- a loose substring check is more realistic
        # than demanding an exact match.
        pred_norm = str(predicted).strip().lower()
        exp_norm = str(expected).strip().lower()
        return 1.0 if exp_norm in pred_norm or pred_norm in exp_norm else 0.0

    return 0.0


def score_extraction(predicted: dict, expected: dict) -> float:
    """
    Field-level accuracy for one example: fraction of the four fields
    (vendor, amount, date, category) extracted correctly. A malformed or
    missing prediction (not a dict) scores 0 across all fields rather
    than raising, since "the model didn't return valid JSON" is itself a
    prompt-quality signal the optimizer needs to see.
    """
    if not isinstance(predicted, dict):
        return 0.0

    fields = ["vendor", "amount", "date", "category"]
    scores = [score_field(field, predicted.get(field), expected[field]) for field in fields]
    return sum(scores) / len(fields)
