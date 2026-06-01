"""
eval_postprocess.py
-------------------
Compute clip_rate_095, clip_rate_099, detach_rate, action_max_mean, and
palm_handle_dist_max_mean from one or more per-step metrics CSVs that
``evaluate_ppo_baseline.py`` writes via ``--log_csv``.

Usage:
    python scripts/eval_postprocess.py CSV [CSV ...]

Prints a short JSON-per-CSV line and a Markdown summary table on stdout.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_rows(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "episode": int(r["episode"]),
                "step": int(r["step"]),
                "reward": float(r["reward"]),
                "done": int(r["done"]),
                "action_abs_max": float(r["action_abs_max"]),
                "palm_to_handle_dist": float(r["palm_to_handle_dist"]),
                "success": int(r["success"]),
                "normalized_progress": float(r["normalized_progress"]),
            }
            for r in reader
        ]


def episode_stats(rows):
    """Return per-episode dicts with terminal reward, success flag, max palm dist."""
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode"]].append(r)
    eps = []
    for ep, ep_rows in sorted(by_ep.items()):
        ep_rows.sort(key=lambda r: r["step"])
        last = ep_rows[-1]
        # detach gives r_detach=-50; r_dist+r_act+r_time at slip step are tiny,
        # so a step reward < -40 reliably flags detach. Min step reward is robust.
        min_reward = min(r["reward"] for r in ep_rows)
        any_success = any(r["success"] for r in ep_rows)
        max_palm = max(r["palm_to_handle_dist"] for r in ep_rows)
        eps.append({
            "episode": ep,
            "steps": len(ep_rows),
            "terminal_reward": last["reward"],
            "min_reward": min_reward,
            "any_success": any_success,
            "max_palm_dist": max_palm,
        })
    return eps


def summarize(path: Path):
    rows = load_rows(path)
    eps = episode_stats(rows)
    n_steps = len(rows)
    if n_steps == 0:
        return {"path": str(path), "error": "empty"}
    clip095 = sum(1 for r in rows if r["action_abs_max"] >= 0.95) / n_steps
    clip099 = sum(1 for r in rows if r["action_abs_max"] >= 0.99) / n_steps
    action_max_mean_step = mean(r["action_abs_max"] for r in rows)
    n_eps = len(eps)
    detach_rate = (
        sum(1 for e in eps if (not e["any_success"]) and e["min_reward"] <= -40.0)
        / n_eps
    )
    success_rate = sum(1 for e in eps if e["any_success"]) / n_eps
    palm_max_mean = mean(e["max_palm_dist"] for e in eps)
    length_mean = mean(e["steps"] for e in eps)
    return {
        "path": str(path),
        "episodes": n_eps,
        "n_steps": n_steps,
        "success_rate": success_rate,
        "length_mean": length_mean,
        "action_max_mean_step": action_max_mean_step,
        "clip_rate_095": clip095,
        "clip_rate_099": clip099,
        "detach_rate": detach_rate,
        "palm_handle_dist_max_mean": palm_max_mean,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--summary_json", default=None, type=Path,
                   help="Optional: dump a list of summaries to this path.")
    args = p.parse_args()

    results = []
    for path in args.csvs:
        if not path.exists():
            print(f"[skip] missing: {path}", file=sys.stderr)
            continue
        results.append(summarize(path))

    print("\n| csv | eps | success | length | act_max | clip095 | clip099 | detach | palm_max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if "error" in r:
            print(f"| {Path(r['path']).name} | ERROR ({r['error']}) | | | | | | | |")
            continue
        print(
            f"| {Path(r['path']).name} | "
            f"{r['episodes']} | "
            f"{r['success_rate']:.2f} | "
            f"{r['length_mean']:.1f} | "
            f"{r['action_max_mean_step']:.3f} | "
            f"{r['clip_rate_095']:.3f} | "
            f"{r['clip_rate_099']:.3f} | "
            f"{r['detach_rate']:.2f} | "
            f"{r['palm_handle_dist_max_mean']:.3f} |"
        )

    print("\nJSON:")
    for r in results:
        print(json.dumps(r))

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
