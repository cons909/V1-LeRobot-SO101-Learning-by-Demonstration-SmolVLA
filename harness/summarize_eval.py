"""
Builds the results table(s) from trial_log.csv files produced by eval_harness.py.

Usage:
    python summarize_eval.py --model 1cam        # summary for one model
    python summarize_eval.py --all                # combined comparison table, all models found
"""

import argparse
import csv
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent.parent / "results"

OUTCOME_ORDER = [
    "full_success", "grasp_only", "reach_only", "no_engagement",
    "timeout_success_slow", "timeout_grasped", "timeout_no_touch", "timeout_object_oob",
]


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows):
    n = len(rows)
    counts = {o: 0 for o in OUTCOME_ORDER}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    durations = [float(r["time_to_completion_sec"]) for r in rows]
    clamp_events = [int(r["safety_clamp_events"]) for r in rows]
    latencies = [float(r["avg_loop_latency_ms"]) for r in rows]

    success_durations = [
        float(r["time_to_completion_sec"]) for r in rows if r["outcome"] == "full_success"
    ]

    return {
        "n": n,
        "counts": counts,
        "mean_duration_all": sum(durations) / n if n else 0,
        "mean_duration_success": (
            sum(success_durations) / len(success_durations) if success_durations else None
        ),
        "mean_clamp_events": sum(clamp_events) / n if n else 0,
        "mean_latency_ms": sum(latencies) / n if n else 0,
    }


def print_model_table(model_key, label, stats):
    print(f"\n## {label} ({model_key})  —  N = {stats['n']}")
    print(f"{'Outcome':<20}{'Count':>8}{'%':>8}")
    for outcome in OUTCOME_ORDER:
        c = stats["counts"].get(outcome, 0)
        pct = (c / stats["n"] * 100) if stats["n"] else 0
        print(f"{outcome:<20}{c:>8}{pct:>7.1f}%")
    print(f"\nMean time-to-completion (all trials): {stats['mean_duration_all']:.1f}s")
    if stats["mean_duration_success"] is not None:
        print(f"Mean time-to-completion (successes only): {stats['mean_duration_success']:.1f}s")
    else:
        print("Mean time-to-completion (successes only): n/a (no successes)")
    print(f"Mean safety-clamp events per trial: {stats['mean_clamp_events']:.2f}")
    print(f"Mean loop latency: {stats['mean_latency_ms']:.0f} ms")


def write_model_markdown(model_key, label, stats, out_path):
    lines = [f"# Results — {label} ({model_key})", "", f"N = {stats['n']} trials", ""]
    lines.append("| Outcome | Count | % |")
    lines.append("|---|---|---|")
    for outcome in OUTCOME_ORDER:
        c = stats["counts"].get(outcome, 0)
        pct = (c / stats["n"] * 100) if stats["n"] else 0
        lines.append(f"| {outcome} | {c} | {pct:.1f}% |")
    lines.append("")
    lines.append(f"- Mean time-to-completion (all trials): {stats['mean_duration_all']:.1f}s")
    if stats["mean_duration_success"] is not None:
        lines.append(
            f"- Mean time-to-completion (successes only): {stats['mean_duration_success']:.1f}s"
        )
    lines.append(f"- Mean safety-clamp events per trial: {stats['mean_clamp_events']:.2f}")
    lines.append(f"- Mean loop latency: {stats['mean_latency_ms']:.0f} ms")
    lines.append("")
    out_path.write_text("\n".join(lines))


def print_comparison_table(all_stats):
    print("\n## Cross-model comparison\n")
    header = f"{'Model':<15}{'N':>5}{'Full success':>15}{'Mean time(s)':>14}{'Clamps/trial':>14}{'Latency(ms)':>13}"
    print(header)
    for model_key, (label, stats) in all_stats.items():
        n = stats["n"]
        full = stats["counts"].get("full_success", 0)
        pct = (full / n * 100) if n else 0
        success_col = f"{full}/{n} ({pct:.0f}%)"
        print(
            f"{model_key:<15}{n:>5}{success_col:>15}"
            f"{stats['mean_duration_all']:>14.1f}{stats['mean_clamp_events']:>14.2f}"
            f"{stats['mean_latency_ms']:>13.0f}"
        )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["1cam", "2cam", "side_claw"])
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    labels = {
        "1cam": "1 Camera (left side)",
        "2cam": "2 Cameras (side left + side right)",
        "side_claw": "Side + Claw",
    }

    if args.model:
        csv_path = RESULTS_ROOT / args.model / "trial_log.csv"
        if not csv_path.exists():
            print(f"No trial log found at {csv_path}")
            return
        rows = load_rows(csv_path)
        stats = summarize_rows(rows)
        print_model_table(args.model, labels[args.model], stats)
        write_model_markdown(
            args.model, labels[args.model], stats,
            RESULTS_ROOT / args.model / "summary.md",
        )
        print(f"\nWritten: {RESULTS_ROOT / args.model / 'summary.md'}")

    if args.all:
        all_stats = {}
        for model_key, label in labels.items():
            csv_path = RESULTS_ROOT / model_key / "trial_log.csv"
            if not csv_path.exists():
                print(f"(skipping {model_key}: no trial log found)")
                continue
            rows = load_rows(csv_path)
            stats = summarize_rows(rows)
            all_stats[model_key] = (label, stats)
            print_model_table(model_key, label, stats)
        if all_stats:
            print_comparison_table(all_stats)


if __name__ == "__main__":
    main()
