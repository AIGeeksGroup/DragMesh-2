"""
Batch runner for the HandDrag trajectory-tracking baseline.

Run this from an activated isaacgym environment:

    python scripts/run_trajectory_baselines.py --phase drag

Each object gets:
  output/baselines/trajectory_tracking/<object_id>.csv
  output/baselines/trajectory_tracking/<object_id>.json
  output/baselines/trajectory_tracking/<object_id>.log

The aggregate table is written to:
  output/baselines/trajectory_tracking/summary.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "baselines" / "trajectory_tracking"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trajectory-tracking baselines over saved HandDrag objects"
    )
    parser.add_argument("--config", default="configs/env/hand_config.yaml")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset_root", default=None,
                        help="Legacy layout root containing <object_id>/trajectory.json. If omitted, use --manifest.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--object_ids", nargs="*", default=None)
    parser.add_argument("--phase", default="drag")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--sim_steps_per_frame", type=int, default=2)
    parser.add_argument("--control_mode", choices=("pd_target", "delta_action"),
                        default="pd_target")
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--hold_final_steps", type=int, default=0)
    parser.add_argument("--break_on_done", action="store_true")
    parser.add_argument("--no_skip_missing_assets", action="store_true",
                        help="Fail on missing GAPartNet assets instead of recording missing_asset")
    return parser.parse_args()



def load_manifest_samples(manifest_path: Path, object_ids=None):
    selected = set(str(x) for x in object_ids) if object_ids else None
    samples = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("enabled", "1").strip() != "1":
                continue
            object_id = row["object_id"]
            if selected and object_id not in selected and row.get("sample_id") not in selected:
                continue
            samples.append({
                "sample_id": row.get("sample_id") or object_id,
                "object_id": object_id,
                "trajectory": Path(row["trajectory"]),
            })
    return samples

def discover_object_ids(dataset_root: Path):
    return sorted(
        p.parent.name for p in dataset_root.glob("*/trajectory.json")
        if p.parent.name.isdigit()
    )


def load_config(config_path: Path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def asset_path(cfg, object_id: str):
    asset_cfg = cfg["asset"]
    return (
        Path(asset_cfg["asset_root"])
        / asset_cfg["arti_obj_root"]
        / object_id
        / "mobility_annotation_gapartnet.urdf"
    )


def baseline_env():
    env = os.environ.copy()
    isaacgym_root = os.environ.get("ISAACGYM_ROOT")
    isaacgym_python = Path(isaacgym_root) / "python" if isaacgym_root else None
    if isaacgym_python and isaacgym_python.exists():
        old_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(isaacgym_python)
            if not old_pythonpath
            else f"{isaacgym_python}:{old_pythonpath}"
        )

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dir = str(Path(conda_prefix) / "lib")
        old_ld = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = lib_dir if not old_ld else f"{lib_dir}:{old_ld}"

    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    env.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions/{py_tag}_cu121")
    return env


def make_command(args, object_id: str, trajectory: Path, metrics_csv: Path,
                 summary_json: Path):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "track_trajectory_baseline.py"),
        "--config", args.config,
        "--object_id", object_id,
        "--trajectory", str(trajectory),
        "--phase", args.phase,
        "--sim_steps_per_frame", str(args.sim_steps_per_frame),
        "--control_mode", args.control_mode,
        "--settle_steps", str(args.settle_steps),
        "--hold_final_steps", str(args.hold_final_steps),
        "--log_csv", str(metrics_csv),
        "--summary_json", str(summary_json),
    ]
    if args.max_frames is not None:
        cmd.extend(["--max_frames", str(args.max_frames)])
    if args.break_on_done:
        cmd.append("--break_on_done")
    return cmd


def tail(text: str, n_lines: int = 20):
    return "\n".join(text.splitlines()[-n_lines:])


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(PROJECT_ROOT / args.config)
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        samples = [
            {"sample_id": object_id, "object_id": object_id,
             "trajectory": dataset_root / object_id / "trajectory.json"}
            for object_id in (args.object_ids or discover_object_ids(dataset_root))
        ]
    else:
        samples = load_manifest_samples(Path(args.manifest), args.object_ids)
    env = baseline_env()
    rows = []

    for sample in samples:
        object_id = sample["object_id"]
        sample_id = sample["sample_id"]
        trajectory = sample["trajectory"]
        metrics_csv = out_dir / f"{sample_id}.csv"
        summary_json = out_dir / f"{sample_id}.json"
        log_path = out_dir / f"{sample_id}.log"

        base_row = {"sample_id": sample_id, "object_id": object_id, "status": "pending", "error": ""}
        if not trajectory.exists():
            base_row.update(status="missing_trajectory", error=str(trajectory))
            rows.append(base_row)
            continue

        asset = asset_path(cfg, object_id)
        if not asset.exists() and not args.no_skip_missing_assets:
            base_row.update(status="missing_asset", error=str(asset))
            rows.append(base_row)
            continue

        cmd = make_command(args, object_id, trajectory, metrics_csv, summary_json)
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path.write_text(proc.stdout)

        if proc.returncode != 0:
            base_row.update(status="failed", error=tail(proc.stdout))
            rows.append(base_row)
            continue

        with open(summary_json) as f:
            summary = json.load(f)
        summary["status"] = "ok"
        summary["error"] = ""
        rows.append(summary)

    fieldnames = [
        "sample_id", "object_id", "status", "phase", "target_joint_idx", "handle_link_name",
        "start_frame", "end_frame", "sim_steps", "final_target_joint",
        "expert_target_joint", "selected_expert_delta", "normalized_progress",
        "mean_palm_handle_dist", "success_step_fraction", "error",
    ]
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(row.get("status") == "ok" for row in rows)
    print(f"wrote {summary_csv}")
    print(f"completed {ok}/{len(rows)} objects")


if __name__ == "__main__":
    main()
