"""Aggregate evaluate_ppo_baseline.py summary JSONs into a fair comparison.

Reads N summary JSONs produced by ``scripts/evaluate_ppo_baseline.py`` and
writes:

* ``comparison.json`` -- per-method aggregates with the same metric set
* ``comparison.csv``  -- one row per (method, mode) combination
* ``comparison.md``   -- a markdown table for paper notes / chat replies

All inputs are expected to be evaluated under identical evaluation conditions
(same object, same number of episodes, same seed, same max_episode_length,
``is_eval_mode=True``). Either deterministic or stochastic actions are fine,
but the mode is recorded per row so deterministic vs stochastic gaps stay
visible in the table.

Example::

    python scripts/compare_history_vs_ppo.py \\
        --inputs old_ppo_summary.json:old_ppo_baseline:det \\
                 history_ppo_summary.json:history_ppo_short:det \\
                 history_long_summary.json:history_ppo_long:det \\
                 old_ppo_stoch_summary.json:old_ppo_baseline:stoch \\
                 history_short_stoch_summary.json:history_ppo_short:stoch \\
                 history_long_stoch_summary.json:history_ppo_long:stoch \\
        --out_dir output/baselines/history_vs_ppo_45936

The ``--old`` / ``--new`` / ``--out_dir`` 2-row form is still supported for
backward compatibility with the earlier 2-checkpoint comparison.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Summary JSON files to aggregate. Each entry is "
             "PATH[:LABEL[:MODE]]. MODE is 'det' or 'stoch' and is "
             "stored in the table. If omitted, MODE is read from the JSON.",
    )
    parser.add_argument("--old", default=None,
                        help="(legacy) summary JSON for old PPO baseline")
    parser.add_argument("--new", default=None,
                        help="(legacy) summary JSON for history PPO")
    parser.add_argument("--old_label", default="old_ppo_baseline")
    parser.add_argument("--new_label", default="history_ppo")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write comparison CSV/JSON/MD")
    return parser.parse_args()


def parse_input_spec(spec, fallback_idx):
    parts = spec.split(":")
    path = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else f"method_{fallback_idx}"
    mode = parts[2] if len(parts) > 2 and parts[2] else None
    return path, label, mode


def stats(values):
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0,
                "min": float(values[0]), "max": float(values[0])}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate(summary):
    eps = summary["episode_metrics"]
    returns = [float(e["return"]) for e in eps]
    steps = [float(e["steps"]) for e in eps]
    final_joint = [float(e["final_target_joint"]) for e in eps]
    progress = [float(e["normalized_progress"]) for e in eps]
    palm_dist = [float(e["mean_palm_handle_dist"]) for e in eps]
    action_l2 = [float(e["mean_action_l2"]) for e in eps]
    action_abs_max = [float(e["mean_action_abs_max"]) for e in eps]
    success = [float(e["success"]) for e in eps]

    return {
        "object_id": summary.get("object_id"),
        "checkpoint": summary.get("checkpoint"),
        "episodes": int(summary.get("episodes", len(eps))),
        "stochastic": bool(summary.get("stochastic", False)),
        "success_rate": float(mean(success)) if success else float("nan"),
        "n_success": int(sum(s > 0 for s in success)),
        "return": stats(returns),
        "steps": stats(steps),
        "final_target_joint": stats(final_joint),
        "normalized_progress": stats(progress),
        "palm_handle_dist": stats(palm_dist),
        "action_l2": stats(action_l2),
        "action_abs_max": stats(action_abs_max),
    }


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{digits}f}"


def write_csv(out_path, rows, fieldnames):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_for_csv(label, mode, agg):
    return {
        "method": label,
        "mode": mode,
        "object_id": agg["object_id"],
        "episodes": agg["episodes"],
        "success_rate": agg["success_rate"],
        "n_success": agg["n_success"],
        "return_mean": agg["return"]["mean"],
        "return_std": agg["return"]["std"],
        "return_min": agg["return"]["min"],
        "return_max": agg["return"]["max"],
        "steps_mean": agg["steps"]["mean"],
        "steps_std": agg["steps"]["std"],
        "steps_min": agg["steps"]["min"],
        "steps_max": agg["steps"]["max"],
        "final_target_joint_mean": agg["final_target_joint"]["mean"],
        "final_target_joint_std": agg["final_target_joint"]["std"],
        "normalized_progress_mean": agg["normalized_progress"]["mean"],
        "normalized_progress_std": agg["normalized_progress"]["std"],
        "normalized_progress_max": agg["normalized_progress"]["max"],
        "palm_handle_dist_mean": agg["palm_handle_dist"]["mean"],
        "action_l2_mean": agg["action_l2"]["mean"],
        "action_abs_max_mean": agg["action_abs_max"]["mean"],
        "checkpoint": agg["checkpoint"],
    }


def make_markdown(rows, all_agg):
    metric_rows = [
        ("success_rate", "success rate", 4),
        ("return_mean", "return mean", 4),
        ("return_std", "return std", 4),
        ("return_min", "return min", 4),
        ("return_max", "return max", 4),
        ("steps_mean", "episode length mean", 2),
        ("steps_std", "episode length std", 2),
        ("final_target_joint_mean", "final target joint (rad)", 6),
        ("normalized_progress_mean", "normalized progress mean", 4),
        ("normalized_progress_std", "normalized progress std", 4),
        ("normalized_progress_max", "normalized progress max", 4),
        ("palm_handle_dist_mean", "mean palm-handle dist (m)", 4),
        ("action_l2_mean", "mean action L2", 4),
        ("action_abs_max_mean", "mean action |max|", 4),
    ]

    lines = []
    object_id = rows[0]["object_id"] if rows else "?"
    episodes = int(rows[0]["episodes"]) if rows else 0
    lines.append(f"# Fair multi-episode evaluation on object `{object_id}`")
    lines.append("")
    lines.append(f"- episodes per (method, mode): {episodes}")
    lines.append("- `seed=42`, `max_episode_length=300`, `num_envs=1`,")
    lines.append("  `is_eval_mode=True` (RSI off; every reset starts at the")
    lines.append("  expert drag-start grasp pose).")
    lines.append("- mode `det` uses deterministic mu output;")
    lines.append("  mode `stoch` samples from `N(mu, exp(logstd))`")
    lines.append("  with the trained sigma parameter.")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    seen = set()
    for r in rows:
        if r["method"] in seen:
            continue
        seen.add(r["method"])
        lines.append(f"- **{r['method']}**: `{r['checkpoint']}`")
    lines.append("")

    header_cells = ["metric"] + [f"{r['method']} ({r['mode']})" for r in rows]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "---|" * len(header_cells))
    for key, name, digits in metric_rows:
        cells = [name]
        for r in rows:
            cells.append(fmt(r[key], digits))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append("- `success_rate` is binary success per episode "
                 "(joint passes `task_done_frac=0.5` of expert range).")
    lines.append("- `episode length mean` close to 13 means the policy "
                 "starts pulling and triggers the `-50` detach penalty "
                 "almost immediately. Length close to 300 means the policy "
                 "stays attached but does not progress.")
    lines.append("- The det/stoch gap is informative: a large gap means "
                 "the trained policy relies on action noise to stay in "
                 "distribution, which is a flat-MLP pathology that a "
                 "temporal encoder (GLA) is meant to address.")
    lines.append("- Counterexample: with `bounds_loss_coef=0.01` on object "
                 "45936 (`history_ppo_long_bounds01`), det and stoch both "
                 "succeed (~0.95 vs ~0.90 over 20 episodes) -- the gap is "
                 "small; the pathology was insufficient action-bound "
                 "regularisation, not the lack of a sequence encoder.")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    if args.inputs:
        for i, spec in enumerate(args.inputs):
            path, label, mode = parse_input_spec(spec, i)
            inputs.append((path, label, mode))
    elif args.old and args.new:
        inputs = [
            (args.old, args.old_label, None),
            (args.new, args.new_label, None),
        ]
    else:
        raise SystemExit(
            "Provide --inputs PATH[:LABEL[:MODE]] ... or --old / --new"
        )

    rows = []
    aggs = []
    for path, label, mode in inputs:
        with open(path) as f:
            summary = json.load(f)
        agg = aggregate(summary)
        if mode is None:
            mode = "stoch" if agg["stochastic"] else "det"
        rows.append(flatten_for_csv(label, mode, agg))
        aggs.append({"label": label, "mode": mode, "agg": agg})

    fieldnames = list(rows[0].keys())
    csv_path = out_dir / "comparison.csv"
    write_csv(csv_path, rows, fieldnames)

    json_path = out_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "object_id": rows[0]["object_id"] if rows else None,
                "episodes": rows[0]["episodes"] if rows else 0,
                "rows": aggs,
            },
            f,
            indent=2,
        )

    md_path = out_dir / "comparison.md"
    md = make_markdown(rows, aggs)
    md_path.write_text(md)

    print(md)
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
