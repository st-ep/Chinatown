"""Generate an action-varied viscosity video dataset from the Genesis pour scene.

This dataset varies viscosity and a small fixed set of robot-pour actions while
keeping geometry, camera, and liquid color fixed. It is intended as the next
step after the single-action viscosity dataset, so the model must use dynamics
instead of appearance shortcuts.

Typical usage:
    python scripts/generate_viscosity_action_dataset.py --num-shards 2 --shard-index 0 --cuda-device 0
    python scripts/generate_viscosity_action_dataset.py --num-shards 2 --shard-index 1 --cuda-device 1
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chinatown" / "robotic_arm_pour_genesis.py"
RUNNER_PATH = ROOT / "scripts" / "run_robotic_arm_pour_viscosity_genesis.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "viscosity_action_dataset_256"
DEFAULT_CACHE_PATH = ROOT / "outputs" / "viscosity_dataset_128" / "_cache" / "settled_particles_water.npy"
DEFAULT_SHARED_LIQUID_COLOR = (0.25, 0.55, 0.95, 1.0)
DEFAULT_MIN_VISCOSITY = 1.0e-3
DEFAULT_MAX_VISCOSITY = 3.0e-2
DEFAULT_VISCOSITIES_PER_ACTION = 64
DEFAULT_HIGH_VISCOSITY_THRESHOLD = 1.2e-2
DEFAULT_HIGH_MICROSTEPS_PER_FRAME = 360
DEFAULT_HIGH_SOURCE_BOUNDARY_CORRECTION_INTERVAL = 4


@dataclass(frozen=True)
class ActionVariant:
    action_index: int
    label: str
    tilt_seconds: float
    return_seconds: float
    pour_pose_fraction: float

    def num_frames(self, frame_rate: int) -> int:
        return int(round((self.tilt_seconds + self.return_seconds) * frame_rate))


DEFAULT_ACTIONS = [
    ActionVariant(0, "baseline_pose080_tilt3p0_return1p6", 3.0, 1.6, 0.80),
    ActionVariant(1, "fast_pose080_tilt2p5_return1p3", 2.5, 1.3, 0.80),
    ActionVariant(2, "slow_pose080_tilt3p5_return1p9", 3.5, 1.9, 0.80),
    ActionVariant(3, "deep_pose084_tilt3p0_return1p6", 3.0, 1.6, 0.84),
]


@dataclass(frozen=True)
class DatasetRun:
    index: int
    viscosity_index: int
    action: ActionVariant
    run_id: str
    viscosity: float
    log10_viscosity: float
    video_path: Path
    metadata_path: Path
    microsteps_per_frame: int
    source_boundary_correction_interval: int


def _load_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_action_dataset_constants", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _python_cmd(args: argparse.Namespace) -> list[str]:
    if args.use_current_python:
        return [sys.executable]
    return ["conda", "run", "-n", args.conda_env, "python"]


def _parse_color(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("color must be R,G,B or R,G,B,A")
    try:
        color = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color channels must be numeric") from exc
    if any(channel < 0.0 or channel > 1.0 for channel in color):
        raise argparse.ArgumentTypeError("color channels must be in [0, 1]")
    if len(color) == 3:
        return (*color, 1.0)
    return color


def _color_arg(color: tuple[float, float, float, float]) -> str:
    return ",".join(f"{channel:.6g}" for channel in color)


def _mu_slug(value: float) -> str:
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def _viscosity_values(args: argparse.Namespace) -> list[float]:
    log_min = math.log10(args.min_viscosity)
    log_max = math.log10(args.max_viscosity)
    if args.viscosities_per_action == 1:
        return [10.0 ** (0.5 * (log_min + log_max))]
    if args.sample_mode == "logspace":
        return [
            10.0 ** (log_min + (log_max - log_min) * index / (args.viscosities_per_action - 1))
            for index in range(args.viscosities_per_action)
        ]

    import random

    rng = random.Random(args.seed)
    return sorted(
        10.0 ** rng.uniform(log_min, log_max) for _ in range(args.viscosities_per_action)
    )


def _solver_settings(args: argparse.Namespace, viscosity: float) -> tuple[int, int]:
    if viscosity >= args.high_viscosity_threshold:
        return args.high_microsteps_per_frame, args.high_source_boundary_correction_interval
    return args.base_microsteps_per_frame, args.base_source_boundary_correction_interval


def _build_runs(args: argparse.Namespace, mod) -> list[DatasetRun]:
    viscosities = _viscosity_values(args)
    runs: list[DatasetRun] = []
    runs_dir = args.output_dir / "runs"
    for action in DEFAULT_ACTIONS:
        for viscosity_index, viscosity in enumerate(viscosities):
            index = action.action_index * len(viscosities) + viscosity_index
            run_id = (
                f"run_{index:04d}_action_{action.action_index:02d}_"
                f"mu_{_mu_slug(viscosity)}"
            )
            run_dir = runs_dir / action.label / f"mu_{viscosity_index:03d}_{_mu_slug(viscosity)}"
            microsteps, correction_interval = _solver_settings(args, viscosity)
            runs.append(
                DatasetRun(
                    index=index,
                    viscosity_index=viscosity_index,
                    action=action,
                    run_id=run_id,
                    viscosity=viscosity,
                    log10_viscosity=math.log10(viscosity),
                    video_path=run_dir / "video.mp4",
                    metadata_path=run_dir / "metadata.json",
                    microsteps_per_frame=microsteps,
                    source_boundary_correction_interval=correction_interval,
                )
            )
    del mod
    return [run for run in runs if run.index % args.num_shards == args.shard_index]


def _metadata(args: argparse.Namespace, mod, run: DatasetRun) -> dict:
    return {
        "schema": "chinatown-viscosity-action-video-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "index": run.index,
        "viscosity_index": run.viscosity_index,
        "mu": run.viscosity,
        "log10_mu": run.log10_viscosity,
        "density": mod.WATER_DENSITY,
        "surface_tension": mod.LIQUID_SURFACE_TENSION,
        "liquid_color": list(args.liquid_color),
        "action_id": run.action.label,
        "action_index": run.action.action_index,
        "tilt_seconds": run.action.tilt_seconds,
        "return_seconds": run.action.return_seconds,
        "pour_pose_fraction": run.action.pour_pose_fraction,
        "geometry_id": "default_two_glass",
        "camera_id": "fixed_0",
        "camera_pos": list(mod.CAMERA_POS),
        "camera_lookat": list(mod.CAMERA_LOOKAT),
        "camera_fov": mod.CAMERA_FOV,
        "fill_fraction": mod.WATER_FILL_FRACTION,
        "num_frames": run.action.num_frames(mod.FRAME_RATE),
        "fps": mod.VIDEO_FPS,
        "microsteps_per_frame": run.microsteps_per_frame,
        "source_boundary_correction_interval": run.source_boundary_correction_interval,
        "cache_path": str(args.cache_path),
        "seed": args.seed + run.index,
        "seed_note": "Recorded for provenance; this fixed-cache dataset is expected to be mostly deterministic.",
        "video_path": str(run.video_path),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_row(run: DatasetRun, mod, status: str, error: str = "") -> dict[str, str]:
    return {
        "index": str(run.index),
        "viscosity_index": str(run.viscosity_index),
        "run_id": run.run_id,
        "mu": f"{run.viscosity:.15g}",
        "log10_mu": f"{run.log10_viscosity:.15g}",
        "action_index": str(run.action.action_index),
        "action_id": run.action.label,
        "tilt_seconds": f"{run.action.tilt_seconds:.15g}",
        "return_seconds": f"{run.action.return_seconds:.15g}",
        "pour_pose_fraction": f"{run.action.pour_pose_fraction:.15g}",
        "num_frames": str(run.action.num_frames(mod.FRAME_RATE)),
        "microsteps_per_frame": str(run.microsteps_per_frame),
        "source_boundary_correction_interval": str(run.source_boundary_correction_interval),
        "video_path": str(run.video_path),
        "metadata_path": str(run.metadata_path),
        "status": status,
        "error": error,
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "viscosity_index",
        "run_id",
        "mu",
        "log10_mu",
        "action_index",
        "action_id",
        "tilt_seconds",
        "return_seconds",
        "pour_pose_fraction",
        "num_frames",
        "microsteps_per_frame",
        "source_boundary_correction_interval",
        "video_path",
        "metadata_path",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    mod = _load_module()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--viscosities-per-action", type=int, default=DEFAULT_VISCOSITIES_PER_ACTION)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the dataset into this many global-index shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run only the shard whose index satisfies run_index %% num_shards == shard_index.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest path for this worker. Defaults to manifest.csv, or manifest_shard_N_of_M.csv for sharded runs.",
    )
    parser.add_argument("--sample-mode", choices=["logspace", "log-uniform"], default="logspace")
    parser.add_argument("--min-viscosity", type=float, default=DEFAULT_MIN_VISCOSITY)
    parser.add_argument("--max-viscosity", type=float, default=DEFAULT_MAX_VISCOSITY)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--conda-env", default="genesis-sim")
    parser.add_argument(
        "--use-current-python",
        action="store_true",
        help="Run child renders with the current interpreter instead of conda run.",
    )
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Shared settled cache. Defaults to the cache from the 128-video single-action dataset.",
    )
    parser.add_argument("--liquid-color", type=_parse_color, default=DEFAULT_SHARED_LIQUID_COLOR)
    parser.add_argument("--liquid-vis-mode", choices=["particle", "recon"], default=None)
    parser.add_argument("--base-microsteps-per-frame", type=int, default=mod.MICROSTEPS_PER_FRAME)
    parser.add_argument(
        "--base-source-boundary-correction-interval",
        type=int,
        default=mod.SOURCE_BOUNDARY_CORRECTION_INTERVAL,
    )
    parser.add_argument("--high-viscosity-threshold", type=float, default=DEFAULT_HIGH_VISCOSITY_THRESHOLD)
    parser.add_argument("--high-microsteps-per-frame", type=int, default=DEFAULT_HIGH_MICROSTEPS_PER_FRAME)
    parser.add_argument(
        "--high-source-boundary-correction-interval",
        type=int,
        default=DEFAULT_HIGH_SOURCE_BOUNDARY_CORRECTION_INTERVAL,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebake-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.viscosities_per_action <= 0:
        parser.error("--viscosities-per-action must be positive")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    if args.min_viscosity <= 0.0 or args.max_viscosity <= 0.0:
        parser.error("viscosity bounds must be positive")
    if args.min_viscosity >= args.max_viscosity:
        parser.error("--min-viscosity must be less than --max-viscosity")
    for name, value in (
        ("--base-microsteps-per-frame", args.base_microsteps_per_frame),
        ("--base-source-boundary-correction-interval", args.base_source_boundary_correction_interval),
        ("--high-microsteps-per-frame", args.high_microsteps_per_frame),
        ("--high-source-boundary-correction-interval", args.high_source_boundary_correction_interval),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    runs = _build_runs(args, mod)
    total_planned = len(DEFAULT_ACTIONS) * args.viscosities_per_action
    if args.manifest_path is None:
        if args.num_shards == 1:
            manifest_path = args.output_dir / "manifest.csv"
        else:
            manifest_path = args.output_dir / f"manifest_shard_{args.shard_index}_of_{args.num_shards}.csv"
    else:
        manifest_path = args.manifest_path
    manifest_rows: list[dict[str, str]] = []

    dataset_metadata = {
        "schema": "chinatown-viscosity-action-video-dataset-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_actions": len(DEFAULT_ACTIONS),
        "viscosities_per_action": args.viscosities_per_action,
        "num_videos": total_planned,
        "num_shards": args.num_shards,
        "sample_mode": args.sample_mode,
        "min_viscosity": args.min_viscosity,
        "max_viscosity": args.max_viscosity,
        "liquid_color": list(args.liquid_color),
        "fps": mod.VIDEO_FPS,
        "actions": [
            {
                "action_index": action.action_index,
                "action_id": action.label,
                "tilt_seconds": action.tilt_seconds,
                "return_seconds": action.return_seconds,
                "pour_pose_fraction": action.pour_pose_fraction,
                "num_frames": action.num_frames(mod.FRAME_RATE),
            }
            for action in DEFAULT_ACTIONS
        ],
        "geometry_id": "default_two_glass",
        "camera_id": "fixed_0",
        "cache_path": str(args.cache_path),
        "high_viscosity_threshold": args.high_viscosity_threshold,
    }
    _write_json(args.output_dir / "dataset_metadata.json", dataset_metadata)

    print(
        f"viscosity/action dataset shard {args.shard_index}/{args.num_shards}: "
        f"{len(runs)} videos, {total_planned} total planned, "
        f"{len(DEFAULT_ACTIONS)} actions x {args.viscosities_per_action} viscosities, "
        f"mu=[{args.min_viscosity:.6g}, {args.max_viscosity:.6g}], "
        f"color={_color_arg(args.liquid_color)}",
        flush=True,
    )
    print(f"output dir: {args.output_dir}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"shared settled cache: {args.cache_path}", flush=True)

    for run in runs:
        command = [
            *_python_cmd(args),
            str(RUNNER_PATH),
            "--viscosity",
            f"{run.viscosity:.15g}",
            "--cache-path",
            str(args.cache_path),
            "--output-path",
            str(run.video_path),
            "--liquid-color",
            _color_arg(args.liquid_color),
            "--microsteps-per-frame",
            str(run.microsteps_per_frame),
            "--source-boundary-correction-interval",
            str(run.source_boundary_correction_interval),
            "--cuda-device",
            args.cuda_device,
            "--tilt-seconds",
            f"{run.action.tilt_seconds:.15g}",
            "--return-seconds",
            f"{run.action.return_seconds:.15g}",
            "--pour-pose-fraction",
            f"{run.action.pour_pose_fraction:.15g}",
        ]
        if args.liquid_vis_mode is not None:
            command.extend(["--liquid-vis-mode", args.liquid_vis_mode])
        if run.index == 0 and (args.rebake_cache or not args.cache_path.exists()):
            command.append("--rebake")

        if args.skip_existing and run.video_path.exists() and run.metadata_path.exists():
            print(f"skip existing {run.run_id}", flush=True)
            manifest_rows.append(_manifest_row(run, mod, "skipped"))
            _write_manifest(manifest_path, manifest_rows)
            continue

        metadata = _metadata(args, mod, run)
        _write_json(run.metadata_path, {**metadata, "command": command})

        try:
            _run_command(command, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            manifest_rows.append(_manifest_row(run, mod, "failed", str(exc)))
            _write_manifest(manifest_path, manifest_rows)
            raise

        status = "dry-run" if args.dry_run else "complete"
        manifest_rows.append(_manifest_row(run, mod, status))
        _write_manifest(manifest_path, manifest_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
