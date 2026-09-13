<div align="center">

# V1 — LeRobot SO-101: Learning by Demonstration with SmolVLA

**Teaching a low-cost robotic arm to pick and place, purely from human demonstration —
and honestly reporting what worked and what didn't across three camera configurations.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with LeRobot](https://img.shields.io/badge/Built%20with-LeRobot-blue)](https://github.com/huggingface/lerobot)
[![Policy: SmolVLA](https://img.shields.io/badge/Policy-SmolVLA-6f42c1)](https://huggingface.co/blog/smolvla)
[![Eval videos on HF Hub](https://img.shields.io/badge/Dataset-150%20eval%20videos%20on%20HF%20Hub-orange)](https://huggingface.co/datasets/cons909/V1-LeRobot-SO101-Eval-Videos)

</div>

---

A pick-and-place task (screwdriver → gray bin) taught to a low-cost SO-101 robotic arm through
imitation learning, using HuggingFace's [LeRobot](https://github.com/huggingface/lerobot)
framework and the [SmolVLA](https://huggingface.co/blog/smolvla) vision-language-action policy.
This is **V1**: three camera configurations were built, trained, and evaluated under one
pre-registered protocol. This repo documents what was tried and what actually happened — it does
not try to improve on those results.

## Contents

- [Goal](#goal)
- [Hardware setup](#hardware-setup)
- [The three configurations](#the-three-configurations)
- [Evaluation methodology](#evaluation-methodology)
- [Results](#results)
- [Sample videos](#sample-videos)
- [Challenges along the way](#challenges-along-the-way)
- [What's next (V2)](#whats-next-v2)
- [Repo structure](#repo-structure)
- [License](#license)

## Goal

Teach a robotic arm to pick up a screwdriver from a table and place it in a gray bin, purely from
human teleoperation demonstrations (no scripted motion, no classical planning) — then compare
three different camera setups honestly, with a real evaluation protocol and real data, not just
a demo reel of the best runs.

## Hardware setup

- **Arm:** SO-101 leader/follower pair, Feetech STS3215 servos.
- **Compute:** a Raspberry Pi handles the arm and cameras; a Mac runs SmolVLA inference and sends
  actions to the Pi over a ZMQ socket. The Pi is physically mounted on the back of the robot
  (cable-length constraint), and the side-camera mounts were hand-built to keep the cameras
  rigid between trials.
- **Cameras:** Logitech C920s, in three configurations (see below).

## The three configurations

| Config | Cameras | Checkpoint step | Notes |
|---|---|---|---|
| **1 camera** | Single left-side camera | 20,000 | Simplest setup; least visual context. |
| **2 cameras** ("side left + side right") | Two side-mounted cameras | 50,000 | Internally the checkpoint's input feature keys are literally named `front`/`side` (that's what the trained model expects — a code-level detail, not a placement description). Started as a claw-mounted camera + one side camera; the claw camera was permanently swapped out mid-project for a second side camera, so this became genuinely two side views. |
| **Side + claw** | One side camera + one claw-mounted camera | 20,000 | Subjectively the most capable of the three during earlier hand-testing, despite added latency from the claw camera's mount. |

**Checkpoint steps differ across configs** (20k / 50k / 20k) — this is disclosed, not
normalized away. Each model is evaluated at its best available checkpoint from actual training,
not at an artificially matched step count. See
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) §2.

## Evaluation methodology

Full pre-registered protocol: [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).
Summary:

- **N = 50 trials per model** (150 total), object position varied naturally across trials
  (not fixed/memorized), same physical setup and operator throughout.
- **Graded live**, in the moment a trial ends, using a fixed rubric decided in advance —
  never adjusted after the fact.
- Trials that hit the 180s timeout are graded on a separate, reduced timeout rubric rather than
  forced into the normal outcome categories (a stalled attempt and a wrong-but-decisive one are
  different failure modes).
- Safety-clamp events (the arm's built-in per-joint safety limiter overriding a requested action)
  are logged as a continuous per-trial count, not a pass/fail outcome.
- This is a modest sample size — results are reported as directional evidence, not a
  statistically significant claim.

The harness that ran all 150 trials: [`harness/`](harness/). The Pi-side servers it talked to:
[`pi_servers/`](pi_servers/). Full build/debugging history (every bug, a real motor-overheating
safety incident, and the camera hardware troubleshooting) is in
[`docs/EVAL_HARNESS_DESIGN.md`](docs/EVAL_HARNESS_DESIGN.md).

## Results

<div align="center">
<img src="media/charts/outcome_comparison.png" alt="Trial outcomes by camera configuration, all 8 graded categories, N=50 per model" width="720">
</div>

Each bar is one camera configuration, stacked to 100% of its 50 trials, reading bottom to top:

- **green — `full_success`**: object grasped and placed in the bin (exact % labeled on the bar).
- **yellow — `timeout_success_slow`**: got the job done, but not before the 180s clock ran out.
- **orange — `timeout_grasped`**: touched or grasped the object within 180s, but didn't finish.
- **red — `timeout_no_touch`**: never engaged the object at all before timing out.
- **purple — `timeout_object_oob`**: the object ended up out of the arm's reach.

Taller green = more full successes. A bar dominated by red means that config mostly never engaged
the object in the first place — exactly what happened to the 2-camera config (78% `timeout_no_touch`),
versus side + claw, which reaches nearly to full green before hitting mostly orange/red. Full
definitions for every category are in [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) §4.

Full per-trial data: [`results/`](results/README.md) (`trial_log.csv` per model,
[`results/SUMMARY.txt`](results/SUMMARY.txt) for the generated tables). All 150 trial videos
(not just the samples below) are archived on Hugging Face Hub:
**[cons909/V1-LeRobot-SO101-Eval-Videos](https://huggingface.co/datasets/cons909/V1-LeRobot-SO101-Eval-Videos)**.

| Model | N | Full success | Mean time (all trials) | Mean time (successes) | Clamps/trial | Loop latency |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 camera | 50 | 10/50 (20%) | 158.4s | 73.5s | 620.92 | 97ms |
| 2 cameras (side left + side right) | 50 | 1/50 (2%) | 178.2s | 84.7s | 388.80 | 160ms |
| **Side + claw** | 50 | **15/50 (30%)** | 149.0s | 83.6s | **1.64** | 253ms |

Side + claw had the highest full-success rate and by far the fewest safety-clamp events, despite
the highest average loop latency — consistent with earlier hand-testing impressions that it was
the most capable of the three, even with the extra lag from the claw camera's mount. The 2-camera
config's near-total shift toward `timeout_no_touch` (78% of its trials) stands out as its
dominant failure mode — it rarely engaged the object at all, rather than engaging and failing
partway through.

Full outcome breakdown (all 8 graded categories) for each model is in
[`results/SUMMARY.txt`](results/SUMMARY.txt).

## Sample videos

One `full_success` and one representative-failure clip per model (the rest of the 150 videos are
on Hugging Face Hub, linked above). Click a link to open the file's page, where GitHub plays it
inline. Only one camera's footage ships per multi-camera config, to avoid a family member
appearing in the dropped camera's frame — this does not affect the logged results, only which raw
video clips are published.

| Config | Full success | Representative failure |
|---|---|---|
| 1 camera | [watch](media/1cam_trial007_full_success.mp4) | [watch](media/1cam_trial004_timeout_grasped.mp4) |
| 2 cameras | [watch](media/2cam_trial017_full_success.mp4) | [watch](media/2cam_trial002_timeout_no_touch.mp4) |
| Side + claw | [watch](media/side_claw_trial018_full_success.mp4) | [watch](media/side_claw_trial005_timeout_no_touch.mp4) |

## Challenges along the way

- **Motor safety incident:** partway through evaluation, a wrist motor overheated during a run —
  the root cause was that only the 1-camera server had load-based safety monitoring
  (HOLD/E-STOP on motor load) built in originally. Fixed by porting that protection into the
  other servers.
- **Camera hardware churn:** the 2-camera config's setup changed mid-project (claw camera
  physically swapped for a second side camera), which required re-diagnosing camera indices,
  USB bandwidth contention, and eventually a UVC driver quirk where `cv2.VideoCapture.set()`
  reports failure even when the requested resolution/fps is actually applied correctly.
- **Action-chunking policy statefulness:** SmolVLA queues multiple future actions per inference
  call; the harness has to reset policy state between trials or it carries stale context forward.
- **Latency vs. accuracy tradeoff:** the best-performing config (side + claw) also had the
  highest loop latency, suggesting more visual context helped even at a speed cost — worth
  investigating further in V2.

Full details, including every bug found and fixed, are in
[`docs/EVAL_HARNESS_DESIGN.md`](docs/EVAL_HARNESS_DESIGN.md).

## What's next (V2)

- Reproduce the best-performing configuration (side + claw) on new/upgraded hardware and
  re-measure, rather than assuming the result transfers.
- Then revisit the 2-camera config, changing one variable at a time instead of multiple
  simultaneous hardware changes.
- Consider an offline held-out action-prediction-error metric if a genuine train/validation
  split is introduced (dropped for V1 — see
  [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) §7).

## Repo structure

```
docs/          Evaluation protocol + harness design/build log
harness/       Mac-side evaluation runner scripts
pi_servers/    Raspberry Pi-side robot/camera control servers
results/       Per-model trial_log.csv + generated summary tables
media/         Sample videos + results chart (full dataset of 150 videos on Hugging Face Hub)
```

## License

MIT — see [`LICENSE`](LICENSE).

---

<div align="center">

*This is a college-admissions portfolio project. The goal of V1 was to document three
already-built configurations honestly, including their failures, not to optimize them further.*

</div>
