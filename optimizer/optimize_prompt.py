"""
The core optimization loop: starting from an initial prompt, repeatedly
proposes variations, scores each against the eval set, and keeps the
best-performing prompt seen so far -- for a fixed number of generations.

The key mechanism is what gets fed back into the proposal step: not just
"the current prompt scored X", but the actual failing examples -- the
input, what was expected, and what the model produced instead. That's
what lets the proposal step target specific, observed failure patterns
(e.g. "the model isn't returning JSON", "dates aren't being normalized")
instead of just guessing at generic improvements.

Selection is elitist and greedy: at the end of each generation, the best
prompt across {current best, all new candidates} moves forward. A
candidate that scores worse than the current best is discarded --
this can get stuck in a local optimum, but it guarantees the score never
regresses across generations, which matters more for a small, fixed eval
set than exploring more aggressively would.

Usage:
    python optimize_prompt.py \
        --model gemma3:12b \
        --optimizer_model gemma3:12b \
        --generations 4 \
        --candidates_per_generation 3 \
        --output ../reports/evolution.json
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "task"))
sys.path.insert(0, str(Path(__file__).parent.parent / "baseline"))
from task_definition import EVAL_EXAMPLES, CATEGORIES  # noqa: E402
from run_baseline import run_prompt_on_eval_set, BASELINE_PROMPT  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"

PROPOSAL_PROMPT_TEMPLATE = """You are improving a prompt used to extract structured expense data (vendor, amount, date, category) from short, informally-written sentences.

Current prompt:
---
{current_prompt}
---

This prompt scored {current_score} (0 to 1, higher is better) on a test set. Here are examples where it failed -- the input, what the correct extraction should have been, and what the model actually produced:

{failure_examples}

Valid categories are: {categories}

Propose {n} improved versions of the prompt that address these specific failures. Each version MUST:
- Keep the exact placeholder "{{input}}" somewhere in it, so the input text can be substituted in
- Instruct the model to respond with ONLY a JSON object with keys: vendor, amount, date, category
- Instruct the model to normalize dates to YYYY-MM-DD format
- Be a complete, standalone prompt (not a diff or explanation)

Reply with ONLY a JSON array of {n} prompt strings, no other text.
"""


def call_ollama(model: str, prompt: str, temperature: float = 0.0, timeout: int = 90) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        return body.get("response", "").strip()


def format_failure_examples(results: list, max_examples: int = 3) -> str:
    """Picks the worst-scoring examples to show the proposer -- these are
    the most informative signal for what needs to change."""
    worst = sorted(results, key=lambda r: r["score"])[:max_examples]

    blocks = []
    for r in worst:
        blocks.append(
            f"Input: {r['input']}\n"
            f"Expected: {json.dumps(r['expected'])}\n"
            f"Model produced: {r['raw_output'][:200]}\n"
            f"Score: {r['score']}"
        )
    return "\n\n".join(blocks)


def propose_candidates(optimizer_model: str, current_prompt: str, current_score: float,
                        results: list, n: int) -> list:
    failure_examples = format_failure_examples(results)
    meta_prompt = PROPOSAL_PROMPT_TEMPLATE.format(
        current_prompt=current_prompt,
        current_score=current_score,
        failure_examples=failure_examples,
        categories=", ".join(CATEGORIES),
        n=n,
    )

    raw_output = call_ollama(optimizer_model, meta_prompt, temperature=0.9)

    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not match:
        return []

    try:
        candidates = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    # Discard anything that dropped the {input} placeholder -- it can't
    # actually be used to run against the eval set.
    return [c for c in candidates if isinstance(c, str) and "{input}" in c]


def main(args):
    print(f"Scoring initial prompt against {len(EVAL_EXAMPLES)} eval examples...")
    current_prompt = BASELINE_PROMPT
    current_outcome = run_prompt_on_eval_set(args.model, current_prompt, EVAL_EXAMPLES)
    current_score = current_outcome["average_score"]

    history = [{"generation": 0, "prompt": current_prompt, "score": current_score, "source": "baseline"}]
    print(f"Generation 0 (baseline): score = {current_score}")

    for generation in range(1, args.generations + 1):
        print(f"\n--- Generation {generation} ---")
        candidates = propose_candidates(
            args.optimizer_model, current_prompt, current_score,
            current_outcome["results"], args.candidates_per_generation,
        )

        if not candidates:
            print("No valid candidates proposed this generation, stopping early.")
            break

        print(f"Proposed {len(candidates)} candidate(s), scoring each...")
        best_candidate_this_gen = None
        best_score_this_gen = current_score
        best_outcome_this_gen = current_outcome

        for i, candidate in enumerate(candidates, start=1):
            outcome = run_prompt_on_eval_set(args.model, candidate, EVAL_EXAMPLES)
            print(f"  candidate {i}: score = {outcome['average_score']}")

            if outcome["average_score"] > best_score_this_gen:
                best_candidate_this_gen = candidate
                best_score_this_gen = outcome["average_score"]
                best_outcome_this_gen = outcome

        if best_candidate_this_gen is not None:
            current_prompt = best_candidate_this_gen
            current_score = best_score_this_gen
            current_outcome = best_outcome_this_gen
            print(f"  -> new best: {current_score} (improved)")
            history.append({
                "generation": generation, "prompt": current_prompt,
                "score": current_score, "source": "optimized",
            })
        else:
            print(f"  -> no candidate beat {current_score}, keeping current prompt")
            history.append({
                "generation": generation, "prompt": current_prompt,
                "score": current_score, "source": "unchanged (no improvement found)",
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "final_prompt": current_prompt,
            "final_score": current_score,
            "history": history,
        }, f, indent=2)

    print(f"\n=== Optimization Complete ===")
    print(f"Baseline score:  {history[0]['score']}")
    print(f"Final score:     {current_score}")
    print(f"Saved evolution history to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated prompt optimization loop")
    parser.add_argument("--model", default="gemma3:12b", help="Model the prompt is being optimized for")
    parser.add_argument("--optimizer_model", default="gemma3:12b", help="Model proposing prompt variations")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--candidates_per_generation", type=int, default=3)
    parser.add_argument("--output", default="../reports/evolution.json")
    args = parser.parse_args()

    main(args)
