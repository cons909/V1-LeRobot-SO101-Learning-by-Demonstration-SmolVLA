# Results

Per-trial evaluation data for all three camera configurations, N = 50 trials each (150 total).
See the main [repo README](../README.md#results) for the headline comparison table and chart, and
[`../docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md) for exactly how these numbers
were produced.

## Files here

- **`1cam/trial_log.csv`, `2cam/trial_log.csv`, `side_claw/trial_log.csv`** — one row per trial:
  `trial_id, date, model_config, checkpoint_step, outcome, time_to_completion_sec,
  safety_clamp_events, avg_loop_latency_ms, stop_reason, notes, video_filename`.
- **`SUMMARY.txt`** — the generated cross-model comparison table (output of
  [`../harness/summarize_eval.py --all`](../harness/summarize_eval.py)).

`outcome` is one of 8 pre-registered categories: `full_success`, `grasp_only`, `reach_only`,
`no_engagement`, `timeout_success_slow`, `timeout_grasped`, `timeout_no_touch`,
`timeout_object_oob` — full definitions in `EVALUATION_PROTOCOL.md` §4.

## Videos

This repo only ships a handful of sample clips (in [`../media/`](../media/)) to stay lightweight.
**All 150 trial videos** are on Hugging Face Hub:
[cons909/V1-LeRobot-SO101-Eval-Videos](https://huggingface.co/datasets/cons909/V1-LeRobot-SO101-Eval-Videos)
— that page has direct playable links and a full file browser.
