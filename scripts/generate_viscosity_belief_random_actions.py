"""Generate randomized-action viscosity rollouts for online belief training.

The dataset is factorial by design: every action program is run with every
viscosity. This lets the estimator learn liquid effects separately from action
effects.

Typical two-GPU launch:
    python scripts/generate_viscosity_belief_random_actions.py --num-shards 2 --shard-index 0 --cuda-device 0
    python scripts/generate_viscosity_belief_random_actions.py --num-shards 2 --shard-index 1 --cuda-device 1
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chinatown" / "robotic_arm_pour_genesis.py"
RUNNER_PATH = ROOT / "scripts" / "run_robotic_arm_pour_viscosity_genesis.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "viscosity_belief_random_actions_2048_v3"
DEFAULT_CACHE_PATH = ROOT / "outputs" / "viscosity_dataset_128" / "_cache" / "settled_particles_water.npy"
DEFAULT_SHARED_LIQUID_COLOR = (0.25, 0.55, 0.95, 1.0)
DEFAULT_MIN_VISCOSITY = 1.0e-3
DEFAULT_MAX_VISCOSITY = 3.0e-2
DEFAULT_NUM_VISCOSITIES = 32
DEFAULT_NUM_ACTIONS = 64
DEFAULT_HIGH_VISCOSITY_THRESHOLD = 1.2e-2
DEFAULT_HIGH_MICROSTEPS_PER_FRAME = 360
DEFAULT_HIGH_SOURCE_BOUNDARY_CORRECTION_INTERVAL = 4
DEFAULT_MIN_VIDEO_BYTES = 100_000
DEFAULT_BAD_RUN_MIN_RECEIVER_FRACTION = 0.005
DEFAULT_BAD_RUN_MAX_SPILL_WITHOUT_RECEIVER_FRACTION = 0.20


@dataclass(frozen=True)
class ActionProgram:
    action_index: int
    action_id: str
    family: str
    keyframes: tuple[tuple[float, float], ...]
    params: dict[str, float]

    @property
    def duration_seconds(self) -> float:
        return self.keyframes[-1][0]

    def num_frames(self, frame_rate: int) -> int:
        return max(1, int(round(self.duration_seconds * frame_rate)))


@dataclass(frozen=True)
class DatasetRun:
    index: int
    viscosity_index: int
    action: ActionProgram
    run_id: str
    viscosity: float
    log10_viscosity: float
    video_path: Path
    metadata_path: Path
    action_program_path: Path
    action_trace_path: Path
    metrics_path: Path
    microsteps_per_frame: int
    source_boundary_correction_interval: int


def _load_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_belief_dataset_constants", MODULE_PATH)
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        "action_family",
        "duration_seconds",
        "num_frames",
        "microsteps_per_frame",
        "source_boundary_correction_interval",
        "video_path",
        "metadata_path",
        "action_program_path",
        "action_trace_path",
        "metrics_path",
        "status",
        "error",
    ]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _lhs_values(rng: random.Random, count: int, low: float, high: float) -> list[float]:
    values = [low + (high - low) * ((index + rng.random()) / count) for index in range(count)]
    rng.shuffle(values)
    return values


def _round_keyframes(keyframes: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple((round(time, 6), round(fraction, 6)) for time, fraction in keyframes)


def _make_family_actions(
    *,
    start_index: int,
    family: str,
    count: int,
    seed: int,
) -> list[ActionProgram]:
    rng = random.Random(seed)
    actions: list[ActionProgram] = []

    if family == "probe":
        pose = _lhs_values(rng, count, 0.68, 0.80)
        tilt = _lhs_values(rng, count, 1.2, 3.0)
        hold = _lhs_values(rng, count, 0.6, 2.0)
        ret = _lhs_values(rng, count, 0.9, 2.4)
        for i in range(count):
            t0 = 0.0
            t1 = tilt[i]
            t2 = t1 + hold[i]
            t3 = t2 + ret[i]
            params = {"max_pose_fraction": pose[i], "tilt_seconds": tilt[i], "hold_seconds": hold[i], "return_seconds": ret[i]}
            keyframes = _round_keyframes([(t0, 0.0), (t1, pose[i]), (t2, pose[i]), (t3, 0.0)])
            index = start_index + i
            actions.append(ActionProgram(index, f"action_{index:03d}_probe", family, keyframes, params))
        return actions

    if family == "normal":
        pose = _lhs_values(rng, count, 0.72, 0.86)
        tilt = _lhs_values(rng, count, 1.6, 4.0)
        hold = _lhs_values(rng, count, 0.2, 1.8)
        ret = _lhs_values(rng, count, 1.0, 2.8)
        for i in range(count):
            t1 = tilt[i]
            t2 = t1 + hold[i]
            t3 = t2 + ret[i]
            params = {"max_pose_fraction": pose[i], "tilt_seconds": tilt[i], "hold_seconds": hold[i], "return_seconds": ret[i]}
            keyframes = _round_keyframes([(0.0, 0.0), (t1, pose[i]), (t2, pose[i]), (t3, 0.0)])
            index = start_index + i
            actions.append(ActionProgram(index, f"action_{index:03d}_normal", family, keyframes, params))
        return actions

    if family == "aggressive":
        pose = _lhs_values(rng, count, 0.82, 0.94)
        tilt = _lhs_values(rng, count, 0.55, 1.8)
        hold = _lhs_values(rng, count, 0.0, 0.9)
        ret = _lhs_values(rng, count, 0.55, 1.6)
        for i in range(count):
            t1 = tilt[i]
            t2 = t1 + hold[i]
            t3 = t2 + ret[i]
            params = {"max_pose_fraction": pose[i], "tilt_seconds": tilt[i], "hold_seconds": hold[i], "return_seconds": ret[i]}
            keyframes = _round_keyframes([(0.0, 0.0), (t1, pose[i]), (t2, pose[i]), (t3, 0.0)])
            index = start_index + i
            actions.append(ActionProgram(index, f"action_{index:03d}_aggressive", family, keyframes, params))
        return actions

    if family == "stop_start":
        first_pose = _lhs_values(rng, count, 0.56, 0.75)
        max_pose = _lhs_values(rng, count, 0.76, 0.94)
        partial_pose = _lhs_values(rng, count, 0.25, 0.55)
        probe_tilt = _lhs_values(rng, count, 0.7, 2.0)
        pause = _lhs_values(rng, count, 0.15, 1.0)
        partial_return = _lhs_values(rng, count, 0.35, 1.2)
        second_tilt = _lhs_values(rng, count, 0.5, 1.8)
        hold = _lhs_values(rng, count, 0.0, 1.2)
        ret = _lhs_values(rng, count, 0.6, 2.2)
        for i in range(count):
            max_fraction = max(max_pose[i], first_pose[i] + 0.10)
            partial_fraction = min(partial_pose[i], first_pose[i] - 0.08)
            t1 = probe_tilt[i]
            t2 = t1 + pause[i]
            t3 = t2 + partial_return[i]
            t4 = t3 + second_tilt[i]
            t5 = t4 + hold[i]
            t6 = t5 + ret[i]
            params = {
                "first_pose_fraction": first_pose[i],
                "partial_pose_fraction": partial_fraction,
                "max_pose_fraction": max_fraction,
                "probe_tilt_seconds": probe_tilt[i],
                "pause_seconds": pause[i],
                "partial_return_seconds": partial_return[i],
                "second_tilt_seconds": second_tilt[i],
                "hold_seconds": hold[i],
                "return_seconds": ret[i],
            }
            keyframes = _round_keyframes(
                [
                    (0.0, 0.0),
                    (t1, first_pose[i]),
                    (t2, first_pose[i]),
                    (t3, partial_fraction),
                    (t4, max_fraction),
                    (t5, max_fraction),
                    (t6, 0.0),
                ]
            )
            index = start_index + i
            actions.append(ActionProgram(index, f"action_{index:03d}_stop_start", family, keyframes, params))
        return actions

    raise ValueError(f"unknown action family {family}")


def _action_programs(num_actions: int, seed: int) -> list[ActionProgram]:
    if num_actions % 4 != 0:
        raise ValueError("--num-actions must be divisible by 4")
    per_family = num_actions // 4
    actions: list[ActionProgram] = []
    for family_index, family in enumerate(("probe", "normal", "aggressive", "stop_start")):
        actions.extend(
            _make_family_actions(
                start_index=family_index * per_family,
                family=family,
                count=per_family,
                seed=seed + family_index * 997,
            )
        )
    return actions


def _viscosity_values(args: argparse.Namespace) -> list[float]:
    log_min = math.log10(args.min_viscosity)
    log_max = math.log10(args.max_viscosity)
    if args.num_viscosities == 1:
        return [args.min_viscosity]
    return [
        10.0 ** (log_min + (log_max - log_min) * index / (args.num_viscosities - 1))
        for index in range(args.num_viscosities)
    ]


def _solver_settings(args: argparse.Namespace, viscosity: float) -> tuple[int, int]:
    if viscosity >= args.high_viscosity_threshold:
        return args.high_microsteps_per_frame, args.high_source_boundary_correction_interval
    return args.base_microsteps_per_frame, args.base_source_boundary_correction_interval


def _action_payload(action: ActionProgram, mod) -> dict:
    return {
        "schema": "chinatown-piecewise-pour-action-v1",
        "action_index": action.action_index,
        "action_id": action.action_id,
        "family": action.family,
        "duration_seconds": action.duration_seconds,
        "num_frames": action.num_frames(mod.FRAME_RATE),
        "params": action.params,
        "keyframes": [
            {"time_seconds": time, "pose_fraction": fraction}
            for time, fraction in action.keyframes
        ],
    }


def _build_runs(args: argparse.Namespace, mod) -> list[DatasetRun]:
    actions = _action_programs(args.num_actions, args.seed)
    viscosities = _viscosity_values(args)
    runs: list[DatasetRun] = []
    runs_dir = args.output_dir / "runs"
    action_dir = args.output_dir / "actions"
    for action in actions:
        action_program_path = action_dir / f"{action.action_id}.json"
        _write_json(action_program_path, _action_payload(action, mod))
        for viscosity_index, viscosity in enumerate(viscosities):
            index = action.action_index * len(viscosities) + viscosity_index
            run_id = (
                f"run_{index:04d}_action_{action.action_index:03d}_"
                f"mu_{_mu_slug(viscosity)}"
            )
            run_dir = runs_dir / action.action_id / f"mu_{viscosity_index:03d}_{_mu_slug(viscosity)}"
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
                    action_program_path=action_program_path,
                    action_trace_path=run_dir / "action_trace.csv",
                    metrics_path=run_dir / "per_frame_metrics.csv",
                    microsteps_per_frame=microsteps,
                    source_boundary_correction_interval=correction_interval,
                )
            )
    return [run for run in runs if run.index % args.num_shards == args.shard_index]


def _metadata(args: argparse.Namespace, mod, run: DatasetRun, command: list[str]) -> dict:
    return {
        "schema": "chinatown-viscosity-belief-random-action-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "index": run.index,
        "viscosity_index": run.viscosity_index,
        "mu": run.viscosity,
        "log10_mu": run.log10_viscosity,
        "density": mod.WATER_DENSITY,
        "surface_tension": mod.LIQUID_SURFACE_TENSION,
        "liquid_color": list(args.liquid_color),
        "action_index": run.action.action_index,
        "action_id": run.action.action_id,
        "action_family": run.action.family,
        "action_duration_seconds": run.action.duration_seconds,
        "action_params": run.action.params,
        "action_keyframes": [
            {"time_seconds": time, "pose_fraction": fraction}
            for time, fraction in run.action.keyframes
        ],
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
        "glass_mesh_path": str(args.glass_mesh_path),
        "seed": args.seed,
        "video_path": str(run.video_path),
        "action_program_path": str(run.action_program_path),
        "action_trace_path": str(run.action_trace_path),
        "metrics_path": str(run.metrics_path),
        "command": command,
    }


def _manifest_row(run: DatasetRun, mod, status: str, error: str = "") -> dict[str, str]:
    return {
        "index": str(run.index),
        "viscosity_index": str(run.viscosity_index),
        "run_id": run.run_id,
        "mu": f"{run.viscosity:.15g}",
        "log10_mu": f"{run.log10_viscosity:.15g}",
        "action_index": str(run.action.action_index),
        "action_id": run.action.action_id,
        "action_family": run.action.family,
        "duration_seconds": f"{run.action.duration_seconds:.15g}",
        "num_frames": str(run.action.num_frames(mod.FRAME_RATE)),
        "microsteps_per_frame": str(run.microsteps_per_frame),
        "source_boundary_correction_interval": str(run.source_boundary_correction_interval),
        "video_path": str(run.video_path),
        "metadata_path": str(run.metadata_path),
        "action_program_path": str(run.action_program_path),
        "action_trace_path": str(run.action_trace_path),
        "metrics_path": str(run.metrics_path),
        "status": status,
        "error": error,
    }


def _video_frame_count(path: Path) -> int:
    command = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "ffprobe failed")
    return int(result.stdout.strip())


def _normalize_video_frame_count(path: Path, *, expected_frames: int, frame_rate: int) -> str:
    frame_count = _video_frame_count(path)
    if frame_count == expected_frames:
        return "video frame count ok"
    if frame_count != expected_frames - 1:
        raise ValueError(f"video frame count {frame_count} != expected {expected_frames}")

    tmp_path = path.with_name(f".{path.name}.pad_tmp.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vf",
        f"tpad=stop_mode=clone:stop_duration={1.0 / frame_rate:.12g},setpts=N/({frame_rate}*TB)",
        "-r",
        str(frame_rate),
        "-frames:v",
        str(expected_frames),
        "-an",
        "-pix_fmt",
        "yuv420p",
        str(tmp_path),
    ]
    subprocess.run(command, check=True)
    repaired_count = _video_frame_count(tmp_path)
    if repaired_count != expected_frames:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"repaired video frame count {repaired_count} != expected {expected_frames}")
    tmp_path.replace(path)
    return f"padded final video frame: {frame_count} -> {expected_frames}"


def _existing_run_health(
    args: argparse.Namespace,
    run: DatasetRun,
    mod,
) -> tuple[bool, str]:
    for path in (
        run.video_path,
        run.metadata_path,
        run.action_trace_path,
        run.metrics_path,
    ):
        if not path.exists():
            return False, f"missing {path.name}"
    if run.video_path.stat().st_size < args.min_video_bytes:
        return False, f"video too small: {run.video_path.stat().st_size} bytes"

    try:
        with run.metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
    except (OSError, csv.Error) as exc:
        return False, f"metrics unreadable: {exc}"

    expected_frames = run.action.num_frames(mod.FRAME_RATE)
    try:
        video_health = _normalize_video_frame_count(
            run.video_path,
            expected_frames=expected_frames,
            frame_rate=mod.FRAME_RATE,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return False, f"video frame count invalid: {exc}"

    if len(rows) != expected_frames:
        return False, f"metrics frame count {len(rows)} != expected {expected_frames}"
    if not rows:
        return False, "metrics empty"

    final = rows[-1]
    try:
        receiver_fraction = float(final["receiver_fraction"])
        spilled_fraction = float(final["spilled_fraction"])
        live_particles = int(final["live_particles"])
        initial_particles = int(final["initial_particle_count"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"metrics malformed: {exc}"

    for name, value in (
        ("receiver_fraction", receiver_fraction),
        ("spilled_fraction", spilled_fraction),
    ):
        if not math.isfinite(value) or value < -1.0e-6 or value > 1.05:
            return False, f"{name} out of range: {value}"
    if initial_particles <= 0 or live_particles <= 0:
        return False, "non-positive particle count"
    if (
        receiver_fraction < args.bad_run_min_receiver_fraction
        and spilled_fraction > args.bad_run_max_spill_without_receiver_fraction
    ):
        return (
            False,
            "near-zero receiver capture with high spill: "
            f"receiver={receiver_fraction:.6g}, spill={spilled_fraction:.6g}",
        )

    return True, video_health


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
    parser.add_argument("--num-viscosities", type=int, default=DEFAULT_NUM_VISCOSITIES)
    parser.add_argument("--num-actions", type=int, default=DEFAULT_NUM_ACTIONS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--min-viscosity", type=float, default=DEFAULT_MIN_VISCOSITY)
    parser.add_argument("--max-viscosity", type=float, default=DEFAULT_MAX_VISCOSITY)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--conda-env", default="genesis-sim")
    parser.add_argument("--use-current-python", action="store_true")
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--glass-mesh-path",
        type=Path,
        default=None,
        help="Shard-local read-only OBJ mesh path. Defaults under OUTPUT_DIR/_mesh/.",
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
    parser.add_argument(
        "--no-validate-existing",
        dest="validate_existing",
        action="store_false",
        help="When --skip-existing is set, skip by file existence only.",
    )
    parser.set_defaults(validate_existing=True)
    parser.add_argument("--min-video-bytes", type=int, default=DEFAULT_MIN_VIDEO_BYTES)
    parser.add_argument(
        "--bad-run-min-receiver-fraction",
        type=float,
        default=DEFAULT_BAD_RUN_MIN_RECEIVER_FRACTION,
        help="Existing runs below this receiver fraction are suspicious only if spill is also high.",
    )
    parser.add_argument(
        "--bad-run-max-spill-without-receiver-fraction",
        type=float,
        default=DEFAULT_BAD_RUN_MAX_SPILL_WITHOUT_RECEIVER_FRACTION,
        help="Existing runs above this spill fraction with near-zero receiver capture are rerendered.",
    )
    parser.add_argument("--rebake-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.num_viscosities <= 0:
        parser.error("--num-viscosities must be positive")
    if args.num_actions <= 0 or args.num_actions % 4 != 0:
        parser.error("--num-actions must be positive and divisible by 4")
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
    if args.min_video_bytes <= 0:
        parser.error("--min-video-bytes must be positive")
    if args.bad_run_min_receiver_fraction < 0.0:
        parser.error("--bad-run-min-receiver-fraction must be non-negative")
    if args.bad_run_max_spill_without_receiver_fraction < 0.0:
        parser.error("--bad-run-max-spill-without-receiver-fraction must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.glass_mesh_path is None:
        args.glass_mesh_path = (
            args.output_dir / "_mesh" / f"shard_{args.shard_index}_pouring_glass.obj"
        )
    args.glass_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mod.GLASS_MESH_PATH = args.glass_mesh_path
    mod.build_glass_mesh(args.glass_mesh_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    runs = _build_runs(args, mod)
    total_planned = args.num_actions * args.num_viscosities
    manifest_path = args.manifest_path
    if manifest_path is None:
        manifest_path = (
            args.output_dir / "manifest.csv"
            if args.num_shards == 1
            else args.output_dir / f"manifest_shard_{args.shard_index}_of_{args.num_shards}.csv"
        )
    manifest_rows: list[dict[str, str]] = []
    actions = _action_programs(args.num_actions, args.seed)

    shard_glass_mesh_paths = [
        str(args.output_dir / "_mesh" / f"shard_{index}_pouring_glass.obj")
        for index in range(args.num_shards)
    ]
    dataset_metadata = {
        "schema": "chinatown-viscosity-belief-random-action-dataset-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_actions": args.num_actions,
        "num_viscosities": args.num_viscosities,
        "num_rollouts": total_planned,
        "num_shards": args.num_shards,
        "min_viscosity": args.min_viscosity,
        "max_viscosity": args.max_viscosity,
        "liquid_color": list(args.liquid_color),
        "fps": mod.VIDEO_FPS,
        "geometry_id": "default_two_glass",
        "camera_id": "fixed_0",
        "cache_path": str(args.cache_path),
        "shard_glass_mesh_paths": shard_glass_mesh_paths,
        "high_viscosity_threshold": args.high_viscosity_threshold,
        "validate_existing": args.validate_existing,
        "bad_run_min_receiver_fraction": args.bad_run_min_receiver_fraction,
        "bad_run_max_spill_without_receiver_fraction": args.bad_run_max_spill_without_receiver_fraction,
        "actions": [_action_payload(action, mod) for action in actions],
    }
    _write_json(args.output_dir / "dataset_metadata.json", dataset_metadata)

    print(
        f"belief random-action dataset shard {args.shard_index}/{args.num_shards}: "
        f"{len(runs)} rollouts, {total_planned} total planned, "
        f"{args.num_actions} actions x {args.num_viscosities} viscosities, "
        f"mu=[{args.min_viscosity:.6g}, {args.max_viscosity:.6g}], "
        f"color={_color_arg(args.liquid_color)}",
        flush=True,
    )
    print(f"output dir: {args.output_dir}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"shared settled cache: {args.cache_path}", flush=True)
    print(f"glass mesh: {args.glass_mesh_path}", flush=True)

    for count, run in enumerate(runs, start=1):
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
            "--glass-mesh-path",
            str(args.glass_mesh_path),
            "--action-program-path",
            str(run.action_program_path),
            "--num-frames",
            str(run.action.num_frames(mod.FRAME_RATE)),
            "--action-trace-path",
            str(run.action_trace_path),
            "--metrics-path",
            str(run.metrics_path),
        ]
        if args.liquid_vis_mode is not None:
            command.extend(["--liquid-vis-mode", args.liquid_vis_mode])
        if run.index == 0 and (args.rebake_cache or not args.cache_path.exists()):
            command.append("--rebake")

        outputs_exist = (
            run.video_path.exists()
            and run.metadata_path.exists()
            and run.action_trace_path.exists()
            and run.metrics_path.exists()
        )
        if args.skip_existing and outputs_exist:
            is_healthy, health_reason = (
                _existing_run_health(args, run, mod)
                if args.validate_existing
                else (True, "not validated")
            )
            if is_healthy:
                print(f"[{count}/{len(runs)}] skip existing {run.run_id}", flush=True)
                manifest_rows.append(_manifest_row(run, mod, "skipped", health_reason))
                _write_manifest(manifest_path, manifest_rows)
                continue
            print(
                f"[{count}/{len(runs)}] rerender invalid existing {run.run_id}: {health_reason}",
                flush=True,
            )

        _write_json(run.metadata_path, _metadata(args, mod, run, command))
        try:
            print(f"[{count}/{len(runs)}] render {run.run_id}", flush=True)
            _run_command(command, dry_run=args.dry_run)
            if not args.dry_run:
                video_health = _normalize_video_frame_count(
                    run.video_path,
                    expected_frames=run.action.num_frames(mod.FRAME_RATE),
                    frame_rate=mod.FRAME_RATE,
                )
                if video_health != "video frame count ok":
                    print(f"[{count}/{len(runs)}] {video_health}", flush=True)
        except subprocess.CalledProcessError as exc:
            manifest_rows.append(_manifest_row(run, mod, "failed", str(exc)))
            _write_manifest(manifest_path, manifest_rows)
            raise
        except (OSError, ValueError) as exc:
            manifest_rows.append(_manifest_row(run, mod, "failed", str(exc)))
            _write_manifest(manifest_path, manifest_rows)
            raise

        status = "dry-run" if args.dry_run else "complete"
        manifest_rows.append(_manifest_row(run, mod, status))
        _write_manifest(manifest_path, manifest_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
