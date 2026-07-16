# Architecture

## Overview

This project turns prompt refinement -- normally a manual, intuition-driven
process -- into a measured, automated loop with four stages:

```
Fixed Task + Eval Set (task/)
    |
    v
Baseline Prompt (baseline/)          hand-written starting point, scored first
    |
    v
Optimizer Loop (optimizer/)          propose variations -> score each -> keep the best -> repeat
    |
    v
Final Comparison (evaluation/)       independent re-run: baseline vs. best optimized prompt
```

The same scoring function and the same eval set are used at every stage --
the baseline, every optimizer candidate, and the final comparison are all
measured identically. That consistency is what makes "the optimized
prompt scored higher" a real claim rather than an artifact of comparing
apples to oranges.

---

## Stage 1: Fixed Task + Eval Set

`task/task_definition.py` defines a structured-extraction task (pull
vendor, amount, date, and category out of an informally-written expense
sentence) along with twelve eval examples and field-level scoring logic.

This task was chosen specifically because prompt wording has real,
measurable leverage over it: dates arrive in inconsistent formats,
amounts are sometimes spelled out, and category is never stated
explicitly -- it has to be inferred. A vague prompt and a precise one
(exact output schema, explicit category list, an instruction to
normalize dates) produce genuinely different scores on the same model,
which is what gives the optimizer loop something real to find.

Scoring is field-level, not exact-match-the-whole-object: an extraction
that gets three of four fields right scores 0.75, not 0. Vendor names
use a loose substring match (recognizing "the electric company" and
"electric company" as the same answer) since demanding exact string
equality would penalize phrasing differences that have nothing to do
with extraction quality.

---

## Stage 2: Baseline Prompt

`baseline/run_baseline.py` holds a deliberately realistic hand-written
prompt -- not a strawman built to lose. It asks for the four fields but
doesn't specify an output format, which is where its first real weakness
shows up: without an explicit JSON instruction, the model is free to
respond in prose, and `parse_prediction()` will often find nothing to
extract.

This file also defines `run_prompt_on_eval_set()`, the function every
other stage calls to score a prompt. Centralizing it here means the
baseline and every later optimizer candidate are evaluated by identical
code -- there's no separate, slightly different scoring path for
"the prompt we're proud of" versus "the prompt we're comparing against."

---

## Stage 3: Optimizer Loop

`optimizer/optimize_prompt.py` runs a fixed number of generations. Each
generation:

1. Takes the current best prompt and its score.
2. Picks the worst-scoring examples from the last run -- the actual
   input, the expected output, and what the model produced instead.
3. Feeds those specific failures to an LLM and asks it to propose several
   new prompt variations that address them.
4. Scores every candidate against the same eval set.
5. Keeps the best of {current best, all new candidates}; discards the rest.

The feedback loop is deliberately failure-driven rather than just
"this prompt scored X, do better" -- showing the proposer concrete
examples of what went wrong (a missing JSON field, an unnormalized date)
gives it something specific to fix, instead of asking it to guess at
generic improvements.

Selection is elitist and greedy: a generation that produces no candidate
beating the current best doesn't regress it. This trades away some
exploration (a temporarily worse candidate might have led somewhere
better) for a guarantee that matters more given a small, fixed eval set:
the score can only go up or stay flat across generations, never down.

---

## Stage 4: Final Comparison

`evaluation/compare_final.py` re-runs both the original baseline prompt
and the final optimized prompt fresh -- it does not reuse the scores
already cached inside the optimizer's own evolution history. An
independent re-measurement is what makes the final number trustworthy
rather than just re-reporting a number the optimizer already claimed
about itself.

The output includes a per-example diff: which specific inputs flipped
from wrong to right (or, if it happened, right to wrong), not just an
aggregate score. That diff is often more informative than the headline
number -- it shows, concretely, what kind of extraction failure the
optimization loop actually fixed.

---

## A Real Limitation, Stated Plainly

The eval set doubles as both the optimizer's selection signal and the
final scoring set. With only twelve examples, repeatedly selecting
candidates against the same set the final score is measured on means
some of the improvement likely reflects fitting to this specific small
set rather than a guaranteed gain on new, unseen inputs -- the same
concern as validating a model against the set used to tune it. A more
rigorous version of this project would hold out a separate set purely
for final scoring, untouched by the optimizer; twelve examples was too
small to split further and still leave both sides meaningful. Worth
treating the measured improvement as a real signal from this experiment,
not a guaranteed generalization claim.

---

## Why This Order

- The optimizer (Stage 3) needs a baseline (Stage 2) to start from and a
  scoring function (Stage 1) to know whether a candidate is actually better.
- The final comparison (Stage 4) needs a completed optimization run --
  there's nothing to compare the baseline against until the loop has
  produced a final prompt.
- Re-scoring both prompts independently at the end, rather than trusting
  the optimizer's own internal bookkeeping, is what turns "the loop
  reported an improvement" into "the improvement was measured twice,
  by different code paths, and agreed."