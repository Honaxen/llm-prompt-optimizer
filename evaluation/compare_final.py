"""
Independently re-runs the baseline prompt and the best optimized prompt
(from optimizer/optimize_prompt.py's output) against the eval set, and
produces a clean side-by-side comparison -- including which specific
examples flipped from wrong to right (or right to wrong).

This is a fresh run, not a reuse of the scores already cached in
optimizer/optimize_prompt.py's evolution.json -- an independent
re-measurement is what makes the final comparison trustworthy rather
than just re-reporting numbers the optimizer already claimed.

Usage:
    python compare_final.py \
        --model gemma3:12b \
        --evolution_file ../reports/evolution.json \
        --output ../reports/final_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "task"))
sys.path.insert(0, str(Path(__file__).parent.parent / "baseline"))
from task_definition import EVAL_EXAMPLES  # noqa: E402
from run_baseline import run_prompt_on_eval_set, BASELINE_PROMPT  # noqa: E402


def build_diff(baseline_outcome: dict, optimized_outcome: dict) -> list:
    """
    Pairs up per-example results from both prompts (they ran against the
    same eval set, in the same order) and flags whether each example
    improved, regressed, or stayed the same.
    """
    diff = []
    for base_r, opt_r in zip(baseline_outcome["results"], optimized_outcome["results"]):
        if opt_r["score"] > base_r["score"]:
            status = "improved"
        elif opt_r["score"] < base_r["score"]:
            status = "regressed"
        else:
            status = "unchanged"

        diff.append({
            "input": base_r["input"],
            "expected": base_r["expected"],
            "baseline_predicted": base_r["predicted"],
            "baseline_score": base_r["score"],
            "optimized_predicted": opt_r["predicted"],
            "optimized_score": opt_r["score"],
            "status": status,
        })
    return diff


def main(args):
    with open(args.evolution_file, "r") as f:
        evolution = json.load(f)
    final_prompt = evolution["final_prompt"]

    print(f"Re-running baseline prompt against {len(EVAL_EXAMPLES)} eval examples...")
    baseline_outcome = run_prompt_on_eval_set(args.model, BASELINE_PROMPT, EVAL_EXAMPLES)

    print(f"Re-running final optimized prompt against {len(EVAL_EXAMPLES)} eval examples...")
    optimized_outcome = run_prompt_on_eval_set(args.model, final_prompt, EVAL_EXAMPLES)

    diff = build_diff(baseline_outcome, optimized_outcome)
    improved = sum(1 for d in diff if d["status"] == "improved")
    regressed = sum(1 for d in diff if d["status"] == "regressed")
    unchanged = sum(1 for d in diff if d["status"] == "unchanged")

    report = {
        "baseline_prompt": BASELINE_PROMPT,
        "baseline_score": baseline_outcome["average_score"],
        "optimized_prompt": final_prompt,
        "optimized_score": optimized_outcome["average_score"],
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_count": unchanged,
        "diff": diff,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Final Comparison ===")
    print(f"Baseline score:   {report['baseline_score']}")
    print(f"Optimized score:  {report['optimized_score']}")
    print(f"Improved: {improved}  Regressed: {regressed}  Unchanged: {unchanged}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent final comparison: baseline vs optimized prompt")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--evolution_file", default="../reports/evolution.json")
    parser.add_argument("--output", default="../reports/final_comparison.json")
    args = parser.parse_args()

    main(args)
