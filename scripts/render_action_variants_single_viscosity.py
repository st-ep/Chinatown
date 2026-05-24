"""Render a small action-variation preview set at one fixed viscosity."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chinatown" / "robotic_arm_pour_genesis.py"
RUNNER_PATH = ROOT / "scripts" / "run_robotic_arm_pour_viscosity_genesis.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "action_variants_single_mu"
DEFAULT_CACHE_PATH = ROOT / "outputs" / "viscosity_dataset_128" / "_cache" / "settled_particles_water.npy"
DEFAULT_LIQUID_COLOR = (0.25, 0.55, 0.95, 1.0)
DEFAULT_VISCOSITY = math.sqrt(1.0e-3 * 3.0e-2)


@dataclass(frozen=True)
class ActionVariant:
    label: str
    tilt_seconds: float
    return_seconds: float
    pour_pose_fraction: float


DEFAULT_VARIANTS = [
    ActionVariant("baseline_pose080_tilt3p0_return1p6", 3.0, 1.6, 0.80),
    ActionVariant("fast_pose080_tilt2p5_return1p3", 2.5, 1.3, 0.80),
    ActionVariant("slow_pose080_tilt3p5_return1p9", 3.5, 1.9, 0.80),
    ActionVariant("shallow_pose074_tilt3p0_return1p6", 3.0, 1.6, 0.74),
    ActionVariant("deep_pose084_tilt3p0_return1p6", 3.0, 1.6, 0.84),
]


def _load_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_action_preview_constants", MODULE_PATH)
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "label",
        "mu",
        "log10_mu",
        "tilt_seconds",
        "return_seconds",
        "pour_pose_fraction",
        "num_frames",
        "video_path",
        "metadata_path",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str], *, dry_run: bool) -> None:
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
    parser.add_argument("--viscosity", type=float, default=DEFAULT_VISCOSITY)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--conda-env", default="genesis-sim")
    parser.add_argument("--use-current-python", action="store_true")
    parser.add_argument("--cuda-device", default="1")
    parser.add_argument("--liquid-color", type=_parse_color, default=DEFAULT_LIQUID_COLOR)
    parser.add_argument("--liquid-vis-mode", choices=["particle", "recon"], default=None)
    parser.add_argument("--microsteps-per-frame", type=int, default=mod.MICROSTEPS_PER_FRAME)
    parser.add_argument(
        "--source-boundary-correction-interval",
        type=int,
        default=mod.SOURCE_BOUNDARY_CORRECTION_INTERVAL,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.viscosity <= 0.0:
        parser.error("--viscosity must be positive")
    if args.microsteps_per_frame <= 0:
        parser.error("--microsteps-per-frame must be positive")
    if args.source_boundary_correction_interval <= 0:
        parser.error("--source-boundary-correction-interval must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    rows: list[dict[str, str]] = []

    dataset_metadata = {
        "schema": "chinatown-action-preview-dataset-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mu": args.viscosity,
        "log10_mu": math.log10(args.viscosity),
        "liquid_color": list(args.liquid_color),
        "cache_path": str(args.cache_path),
        "microsteps_per_frame": args.microsteps_per_frame,
        "source_boundary_correction_interval": args.source_boundary_correction_interval,
        "variants": [variant.__dict__ for variant in DEFAULT_VARIANTS],
    }
    _write_json(args.output_dir / "dataset_metadata.json", dataset_metadata)

    print(
        f"rendering {len(DEFAULT_VARIANTS)} action variants at mu={args.viscosity:.6g} "
        f"to {args.output_dir}",
        flush=True,
    )

    for variant in DEFAULT_VARIANTS:
        num_frames = int(round((variant.tilt_seconds + variant.return_seconds) * mod.FRAME_RATE))
        run_dir = args.output_dir / variant.label
        video_path = run_dir / f"robotic_arm_pour_{variant.label}_mu_{_mu_slug(args.viscosity)}.mp4"
        metadata_path = run_dir / "metadata.json"
        if args.skip_existing and video_path.exists() and metadata_path.exists():
            status = "skipped"
        else:
            command = [
                *_python_cmd(args),
                str(RUNNER_PATH),
                "--viscosity",
                f"{args.viscosity:.15g}",
                "--cache-path",
                str(args.cache_path),
                "--output-path",
                str(video_path),
                "--liquid-color",
                _color_arg(args.liquid_color),
                "--microsteps-per-frame",
                str(args.microsteps_per_frame),
                "--source-boundary-correction-interval",
                str(args.source_boundary_correction_interval),
                "--cuda-device",
                args.cuda_device,
                "--tilt-seconds",
                f"{variant.tilt_seconds:.15g}",
                "--return-seconds",
                f"{variant.return_seconds:.15g}",
                "--pour-pose-fraction",
                f"{variant.pour_pose_fraction:.15g}",
            ]
            if args.liquid_vis_mode is not None:
                command.extend(["--liquid-vis-mode", args.liquid_vis_mode])
            metadata = {
                "schema": "chinatown-action-preview-run-v1",
                "label": variant.label,
                "mu": args.viscosity,
                "log10_mu": math.log10(args.viscosity),
                "tilt_seconds": variant.tilt_seconds,
                "return_seconds": variant.return_seconds,
                "pour_pose_fraction": variant.pour_pose_fraction,
                "num_frames": num_frames,
                "fps": mod.VIDEO_FPS,
                "video_path": str(video_path),
                "command": command,
            }
            _write_json(metadata_path, metadata)
            _run(command, dry_run=args.dry_run)
            status = "dry-run" if args.dry_run else "complete"

        rows.append(
            {
                "label": variant.label,
                "mu": f"{args.viscosity:.15g}",
                "log10_mu": f"{math.log10(args.viscosity):.15g}",
                "tilt_seconds": f"{variant.tilt_seconds:.15g}",
                "return_seconds": f"{variant.return_seconds:.15g}",
                "pour_pose_fraction": f"{variant.pour_pose_fraction:.15g}",
                "num_frames": str(num_frames),
                "video_path": str(video_path),
                "metadata_path": str(metadata_path),
                "status": status,
            }
        )
        _write_manifest(manifest_path, rows)

    print(f"wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
