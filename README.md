# LLM Prompt Optimizer

Work in progress -- this README is a placeholder and will be replaced once the project is complete.

An automated prompt optimization loop -- instead of hand-tuning a prompt, the system proposes variations, scores them against an eval set, and keeps what actually works, iteration by iteration.

---

## What This Project Will Demonstrate

Every other project in this portfolio uses hand-written prompts, refined by intuition and trial and error.
This one turns prompt refinement itself into a measured, automated loop.

Concern -> Solution (planned)
- How do you improve a prompt without guessing?        -> Score every candidate against a fixed eval set, not a feeling
- How do you generate better candidates automatically?  -> An LLM proposes prompt variations based on what scored well and what didn't
- Did automated optimization actually beat hand-tuning?  -> Direct comparison: hand-written baseline vs. the best optimized prompt
- Can I see how the prompt evolved?                      -> A generation-by-generation report showing every version and its score

---

## Planned Architecture

Fixed Task + Eval Set (task/)
  -> Baseline Prompt (baseline/)              hand-written starting point, scored first
  -> Optimizer Loop (optimizer/)              propose variations -> score each -> keep the best -> repeat
  -> Final Comparison (evaluation/)           baseline vs. best optimized prompt, same eval set
  -> Evolution Report (reports/)               every generation's prompt + score, side by side

---

## Project Structure

llm-prompt-optimizer/
  task/           - fixed task definition + eval dataset
  baseline/        - hand-written baseline prompt
  optimizer/       - propose/score/select loop
  evaluation/      - baseline vs. optimized, final comparison
  reports/         - generation-by-generation evolution report
  tests/
  docs/

---

## Stack

Python - Ollama - pytest

---

## Status

- [ ] Fixed task + eval dataset
- [ ] Hand-written baseline prompt, scored
- [ ] Optimizer loop (propose, score, select, repeat)
- [ ] Baseline vs. optimized final comparison
- [ ] Evolution report

---

## Related Projects

- [llm-evaluation-playground](https://github.com/Honaxen/llm-evaluation-playground) -- the eval-set-driven scoring approach this project automates on top of
- [llm-preference-alignment](https://github.com/Honaxen/llm-preference-alignment) -- same "score candidates, keep what wins" pattern, applied to prompts instead of model weights

---

## Author

[Honaxen](https://github.com/Honaxen)
