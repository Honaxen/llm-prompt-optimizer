"""
Runs the hand-written baseline prompt against the fixed eval set from
task/task_definition.py and reports its average score.

This prompt is deliberately realistic, not deliberately broken -- it's
what a reasonable first attempt looks like, not a strawman. The point of
optimizer/optimize_prompt.py isn't to beat an intentionally bad baseline;
it's to see whether automated iteration finds real headroom over a prompt
someone would actually ship as-is.

run_prompt_on_eval_set() is the shared core both this script and
optimizer/optimize_prompt.py call -- keeping scoring identical between
the baseline and every optimizer candidate is what makes the final
comparison fair.

Usage:
    python run_baseline.py --model gemma3:12b --output ../reports/baseline_results.json
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "task"))
from task_definition import EVAL_EXAMPLES, score_extraction  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

BASELINE_PROMPT = """Extract the expense information from the following text. Return the vendor, amount, date, and category.

Text: {input}

Respond with the extracted information."""


def call_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body.get("response", "").strip()


def parse_prediction(raw_output: str) -> dict | None:
    """
    Extracts the first JSON object from the model's raw output. The
    baseline prompt doesn't explicitly demand JSON -- which is also where
    its first real weakness shows up. Without an explicit output-format
    instruction, the model is free to respond in prose, and this parser
    will often come up empty. That gap is exactly the kind of thing
    optimizer/optimize_prompt.py should learn to fix.
    """
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def run_prompt_on_eval_set(model: str, prompt_template: str, eval_examples: list) -> dict:
    """
    Runs one prompt template against the full eval set and returns
    per-example results plus the average score. Shared between the
    baseline and every optimizer candidate so all prompts are scored
    identically.
    """
    results = []
    for example in eval_examples:
        formatted_prompt = prompt_template.format(input=example["input"])
        raw_output = call_ollama(model, formatted_prompt)
        predicted = parse_prediction(raw_output)
        score = score_extraction(predicted, example["expected"])

        results.append({
            "input": example["input"],
            "expected": example["expected"],
            "predicted": predicted,
            "raw_output": raw_output,
            "score": score,
        })

    average_score = sum(r["score"] for r in results) / len(results)
    return {"average_score": round(average_score, 4), "results": results}


def main(args):
    print(f"Running baseline prompt against {len(EVAL_EXAMPLES)} eval examples...")
    outcome = run_prompt_on_eval_set(args.model, BASELINE_PROMPT, EVAL_EXAMPLES)

    print(f"\nBaseline average score: {outcome['average_score']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"prompt": BASELINE_PROMPT, **outcome}, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hand-written baseline prompt against the eval set")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--output", default="../reports/baseline_results.json")
    args = parser.parse_args()

    main(args)
