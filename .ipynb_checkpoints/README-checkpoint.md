# LLM Prompt Optimizer

An automated prompt optimization loop — instead of hand-tuning a prompt, the system proposes variations, scores them against an eval set, and keeps what actually works, iteration by iteration.

---

## What This Project Demonstrates

Every other project in this portfolio uses hand-written prompts, refined by intuition and trial and error.
This one turns prompt refinement itself into a measured, automated loop.

| Concern | Solution |
|---|---|
| How do you improve a prompt without guessing? | Score every candidate against a fixed eval set, not a feeling |
| How do you generate better candidates automatically? | An LLM proposes variations based on the actual failing examples, not generic advice |
| Did automated optimization actually beat hand-tuning? | Independent re-run: hand-written baseline vs. the best optimized prompt |
| Can I see how the prompt evolved? | A generation-by-generation report showing every version and its score |

---

## Architecture

```
Fixed Task + Eval Set
  ↓
Baseline Prompt  →  hand-written starting point, scored first
  ↓
Optimizer Loop  →  propose variations → score each → keep the best → repeat
  ↓
Final Comparison  →  independent re-run: baseline vs. best optimized prompt
```

---

## Project Structure

```
llm-prompt-optimizer/
├── task/
│   └── task_definition.py         — extraction task, 12 eval examples, field-level scoring
├── baseline/
│   └── run_baseline.py            — hand-written baseline prompt + shared eval-set runner
├── optimizer/
│   └── optimize_prompt.py         — failure-driven propose/score/select loop
├── evaluation/
│   └── compare_final.py           — independent baseline vs. optimized comparison + diff
├── reports/                       — evolution history, final comparison (generated)
├── tests/
│   └── test_prompt_optimizer.py   — 22/22 passing
├── docs/
│   └── architecture.md
└── requirements.txt
```

---

## Getting Started

```bash
pip install -r requirements.txt
ollama serve
```

### 1. Score the baseline prompt

```bash
python baseline/run_baseline.py --model gemma3:12b --output reports/baseline_results.json
```

### 2. Run the optimization loop

```bash
python optimizer/optimize_prompt.py \
  --model gemma3:12b \
  --optimizer_model gemma3:12b \
  --generations 4 \
  --candidates_per_generation 3 \
  --output reports/evolution.json
```

Example evolution *(illustrative — replace with your own run)*:
```
Generation 0 (baseline):        score = 0.520
Generation 1 (optimized):       score = 0.708  -- added explicit JSON schema instruction
Generation 2 (optimized):       score = 0.833  -- added date normalization instruction
Generation 3 (unchanged):       score = 0.833  -- no candidate beat current best
Generation 4 (optimized):       score = 0.896  -- added explicit category list + few-shot example
```

### 3. Compare baseline vs. optimized, independently

```bash
python evaluation/compare_final.py \
  --model gemma3:12b \
  --evolution_file reports/evolution.json \
  --output reports/final_comparison.json
```

Example output *(illustrative — replace with your own run)*:
```
=== Final Comparison ===
Baseline score:   0.520
Optimized score:  0.896
Improved: 6  Regressed: 0  Unchanged: 6
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Stack

Python · Ollama · pytest

---

## What I Learned

**Failure-driven feedback beats "try to do better."**
Early versions of the proposal step just told the model its current score and asked for improvements. Feeding it the actual worst-scoring examples — the input, the expected output, and what the model produced instead — gave it something concrete to fix instead of guessing at generic changes.

**The biggest score jumps came from format, not phrasing.**
The largest single improvement in most runs wasn't clever wording — it was simply telling the model to respond with only a JSON object matching an exact schema. The baseline prompt never said that, so the parser often found nothing to extract at all. A format instruction fixed more than any amount of rewording the task description.

**Greedy, non-regressing selection was the right tradeoff for a small eval set.**
Only ever keeping a candidate that beats the current best means the score can't drop across generations — at the cost of possibly missing a candidate that looked worse initially but might have led somewhere better with more iterations. For twelve eval examples, guaranteeing monotonic improvement mattered more than exploring more aggressively.

**Optimizing against the same set you score on is a real limitation, not just a footnote.**
With only twelve examples, the optimizer selects candidates against the same set the final comparison measures. Some of the measured gain likely reflects fitting to this specific small set rather than a guaranteed improvement on new inputs — the same concern as validating against your own tuning set. A proper version of this would hold out a separate scoring set the optimizer never touches.

**Re-scoring independently at the end caught the temptation to just trust the loop's own numbers.**
It would have been easy to just report whatever score the optimizer's internal bookkeeping claimed for its final prompt. Running `compare_final.py` as a completely separate re-measurement, using the same shared scoring function but a fresh call to both prompts, is what makes the final comparison a real check rather than the optimizer grading its own homework.

---

## Related Projects

- [llm-evaluation-playground](https://github.com/Honaxen/llm-evaluation-playground) — the eval-set-driven scoring approach this project automates on top of
- [llm-preference-alignment](https://github.com/Honaxen/llm-preference-alignment) — same "score candidates, keep what wins" pattern, applied to prompts instead of model weights

---

## Author

[Honaxen](https://github.com/Honaxen)