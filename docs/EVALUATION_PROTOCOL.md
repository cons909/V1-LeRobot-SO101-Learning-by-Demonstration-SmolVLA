# Evaluation Protocol — SO-101 Screwdriver-to-Bin Task

Pre-registered before running trials. Defines success criteria, controls, and metrics for
comparing the three camera configurations (1 camera / 2 cameras / claw+side). Written to be
reproducible and defensible in a portfolio/competition context (ISEF-style: no fabricated or
post-hoc-adjusted data). Trials are run with `eval_harness.py`; see `EVAL_HARNESS_DESIGN.md`
for the automation architecture.

## 1. Controls (hold constant across all trials, all models)
- Same physical screwdriver and gray bin.
- Same lighting condition (same time of day / same lamp state).
- Same table/background, no added or removed clutter between trials.
- Same person operating the reset process and judging when each trial is over.

## 2. Confound to disclose, not hide
The three checkpoints were not all trained for the same number of steps (side+claw and the
1-camera model stop at 20k; the 2-camera model's evaluated checkpoint is 50k, its final major
training run). This is disclosed directly in the results table (checkpoint step is logged per
trial) rather than normalized away — the comparison is "best available checkpoint per config,"
not an artificially matched step count.

## 3. Object position sampling
Positions are generalized: the screwdriver is not placed at one fixed, memorized spot for every
trial — it varies naturally within the reachable workspace across the trial set, so results
reflect generalization rather than memorization of a single placement.

## 4. Success rubric (graded, decided in advance)
1. **Full success** — object grasped and placed inside the bin.
2. **Grasp only** — object grasped but dropped/misplaced before reaching the bin.
3. **Reach only** — arm moved toward the object, no stable grasp achieved.
4. **No meaningful engagement** — arm stayed near average/home pose or moved unrelated to the
   object ("collapsed to average action" failure mode).

A trial ends when the operator presses `n` — the operator watches the arm and judges in real
time when the attempt is over (succeeded, failed, stuck, whatever happened), and immediately
after that is prompted with the fixed rubric above. Not adjusted after the fact based on how the
run "felt" — the grading happens in the same moment the trial is judged over, not retroactively.

**Overtime trials get a different, reduced rubric.** If a trial runs the full `TRIAL_TIMEOUT_S`
(currently 180s / 3 min — check the harness file, this has been tuned more than once) with no
`n` press, the full 1-4 rubric above doesn't apply — the harness asks a reduced 4-choice question
instead: did it grasp the object and place it in the bin, just too slowly
(`timeout_success_slow`); did it grasp/touch the object at some point but not finish
(`timeout_grasped`); did it never touch the object at all (`timeout_no_touch`); or did the object
fall out of bounds so it never got a chance to grab it (`timeout_object_oob`)? The arm is **not**
auto-returned anywhere on a timeout — torque releases and the operator repositions it by hand,
same as any other trial end. Time-to-completion for these trials is logged at the max time —
they're explicitly marked as overtime (`stop_reason = timeout`), not folded into the
normal 1-4 categories, since "ran out of time" is a materially different failure mode from the
model confidently reaching a wrong conclusion within a normal timeframe.

Safety-clamp events are **not** a rubric outcome — they're logged as a continuous per-trial
count instead (see §6), since a trial can have clamp events and still succeed, or have none and
still fail.

## 5. Trial count
N = 50 trials per model (150 total across the three configs), run via `eval_harness.py`. This is
a modest sample for strong statistical claims — report results as directional evidence with the
raw counts shown, not as statistically significant differences. Do not stop early just because
results look good or bad partway through (avoids optional-stopping bias).

## 6. Per-trial metrics logged
Produced automatically by `eval_harness.py` into `results/<model>/trial_log.csv`: trial ID,
timestamp, model/config, checkpoint step, outcome, time-to-completion (sec), safety-clamp event
count, average loop latency (ms, observation received → action executed), stop reason
(`manual_end` / `timeout`), notes, video filename(s).

**How clamp events are counted, with no Pi-side modification:** the Mac already knows the action
it requested; every Pi server variant returns `action_sent` (what was actually executed, after
the robot's own safety clamp) in its ack. If any joint's requested vs. executed value differs by
more than `CLAMP_TOLERANCE`, that step counts as a clamp event. Simpler and more precise than
parsing the Pi terminal's printed warnings.

**How a trial ends:** the operator judges it live and presses `n` (deliberately not a fixed
proximity-to-home check — the arm doesn't need to return to any exact position for a trial to
count as over, since a completed pick-and-place can end almost anywhere in the workspace). The
120s timeout is the only automatic backstop, for a trial the operator didn't end in time. `q`
quits the whole session instead of ending just the current trial.

**Reset pacing is fixed, not operator-gated.** Right after a trial is graded and logged, the
next `RESET_COUNTDOWN_S` (8s minimum) reset countdown begins automatically — there's no separate
"press a key to continue" step between grading and the countdown; the `n` press that ends the
trial is the only manual gate in the loop.

## 7. Dropped: offline held-out action-prediction error
Considered and dropped for V1. It would have compared predicted vs. recorded teleop actions on
dataset episodes, but without a genuine train/validation split (training used the full dataset),
it would only measure training-set fit, not real generalization — not worth the added complexity
for what it would prove. Left as a possible V2 addition if a real held-out split is introduced.

## 8. Reporting format
One results table per model (generated by `summarize_eval.py`): N, outcome counts/percentages
across all 8 categories (`full_success`, `grasp_only`, `reach_only`, `no_engagement`,
`timeout_success_slow`, `timeout_grasped`, `timeout_no_touch`, `timeout_object_oob`), mean
time-to-completion (all
trials, and successes-only), mean clamp events/trial, mean loop latency — plus a combined
cross-model comparison table, and 1–2 representative videos per model (one success, one representative
failure — failures get documented too, not hidden).
