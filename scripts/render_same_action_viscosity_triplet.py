"""Render three viscosity variants with identical robot action and duration.

This script uses ``run_robotic_arm_pour_viscosity_genesis.py`` for each variant
so the default raised-start trajectory and frame count stay fixed. The
honey-like case uses a smaller honey-stable SPH timestep because mu=0.03 is
numerically unstable at the base water timestep.

Typical usage:
    conda run -n genesis-sim python scripts/render_same_action_viscosity_triplet.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chinatown" / "robotic_arm_pour_genesis.py"
DEFAULT_OUT_DIR = ROOT / "outputs" / "same_action_viscosity"
HONEY_LIKE_VISCOSITY = 0.030
HONEY_LIKE_MICROSTEPS_PER_FRAME = 360
HONEY_LIKE_BOUNDARY_CORRECTION_INTERVAL = 4
DEFAULT_SHARED_LIQUID_COLOR = (0.25, 0.55, 0.95, 1.0)


@dataclass(frozen=True)
class ViscosityVariant:
    label: str
    viscosity: float
    color: tuple[float, float, float, float]
    microsteps_per_frame: int
    source_boundary_correction_interval: int


def _load_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_constants", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _python_cmd(args: argparse.Namespace) -> list[str]:
    if args.use_current_python:
        return [sys.executable]
    return ["conda", "run", "-n", args.conda_env, "python"]


def _mu_slug(value: float) -> str:
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def _color_arg(color: tuple[float, float, float, float]) -> str:
    return ",".join(f"{channel:.6g}" for channel in color)


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


def _default_variants(
    *,
    water_viscosity: float,
    middle_viscosity: float | None,
    honey_like_viscosity: float,
    liquid_color: tuple[float, float, float, float],
    base_microsteps_per_frame: int,
    base_source_boundary_correction_interval: int,
    honey_microsteps_per_frame: int,
    honey_source_boundary_correction_interval: int,
) -> list[ViscosityVariant]:
    if middle_viscosity is None:
        middle_viscosity = math.sqrt(water_viscosity * honey_like_viscosity)
    return [
        ViscosityVariant(
            "water",
            water_viscosity,
            liquid_color,
            base_microsteps_per_frame,
            base_source_boundary_correction_interval,
        ),
        ViscosityVariant(
            "middle_log",
            middle_viscosity,
            liquid_color,
            base_microsteps_per_frame,
            base_source_boundary_correction_interval,
        ),
        ViscosityVariant(
            "honey_like",
            honey_like_viscosity,
            liquid_color,
            honey_microsteps_per_frame,
            honey_source_boundary_correction_interval,
        ),
    ]


def _run(command: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    mod = _load_module()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Shared settled-particle cache. Defaults to the base scene cache so all variants start identically.",
    )
    parser.add_argument("--num-frames", type=int, default=mod.VIDEO_NUM_FRAMES)
    parser.add_argument("--conda-env", default="genesis-sim")
    parser.add_argument(
        "--use-current-python",
        action="store_true",
        help="Run the Genesis scripts with the current interpreter instead of conda run.",
    )
    parser.add_argument(
        "--cuda-device",
        default="1",
        help="CUDA device id to expose through CUDA_VISIBLE_DEVICES for all child renders.",
    )
    parser.add_argument("--water-viscosity", type=float, default=mod.WATER_VISCOSITY)
    parser.add_argument(
        "--middle-viscosity",
        type=float,
        default=None,
        help="Middle viscosity. Defaults to the log midpoint between water and honey-like.",
    )
    parser.add_argument("--honey-like-viscosity", type=float, default=HONEY_LIKE_VISCOSITY)
    parser.add_argument(
        "--base-microsteps-per-frame",
        type=int,
        default=mod.MICROSTEPS_PER_FRAME,
        help="Physics microsteps per 60 Hz video frame for water and log-midpoint variants.",
    )
    parser.add_argument(
        "--honey-microsteps-per-frame",
        type=int,
        default=HONEY_LIKE_MICROSTEPS_PER_FRAME,
        help="Physics microsteps per 60 Hz video frame for the honey-like variant.",
    )
    parser.add_argument(
        "--base-source-boundary-correction-interval",
        type=int,
        default=mod.SOURCE_BOUNDARY_CORRECTION_INTERVAL,
        help="Source-cup particle boundary correction interval for water and log-midpoint variants.",
    )
    parser.add_argument(
        "--honey-source-boundary-correction-interval",
        type=int,
        default=HONEY_LIKE_BOUNDARY_CORRECTION_INTERVAL,
        help="Source-cup particle boundary correction interval for the honey-like variant.",
    )
    parser.add_argument(
        "--liquid-vis-mode",
        choices=["particle", "recon"],
        default=None,
        help="Genesis liquid visualization mode. Defaults to the custom-viscosity runner default.",
    )
    parser.add_argument(
        "--liquid-color",
        type=_parse_color,
        default=DEFAULT_SHARED_LIQUID_COLOR,
        help="Shared liquid render color as R,G,B or R,G,B,A. Applied to every viscosity variant.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerender an existing output, unless the shared cache must be baked first.",
    )
    parser.add_argument(
        "--rebake-cache",
        action="store_true",
        help="Rebuild the shared settled-particle cache once before the water render.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    for name, value in (
        ("--base-microsteps-per-frame", args.base_microsteps_per_frame),
        ("--honey-microsteps-per-frame", args.honey_microsteps_per_frame),
        (
            "--base-source-boundary-correction-interval",
            args.base_source_boundary_correction_interval,
        ),
        (
            "--honey-source-boundary-correction-interval",
            args.honey_source_boundary_correction_interval,
        ),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    for name, value in (
        ("--water-viscosity", args.water_viscosity),
        ("--honey-like-viscosity", args.honey_like_viscosity),
    ):
        if value <= 0.0:
            parser.error(f"{name} must be positive")
    if args.middle_viscosity is not None and args.middle_viscosity <= 0.0:
        parser.error("--middle-viscosity must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_path is None:
        args.cache_path = mod.SETTLED_PARTICLES_CACHE
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)

    variants = _default_variants(
        water_viscosity=args.water_viscosity,
        middle_viscosity=args.middle_viscosity,
        honey_like_viscosity=args.honey_like_viscosity,
        liquid_color=args.liquid_color,
        base_microsteps_per_frame=args.base_microsteps_per_frame,
        base_source_boundary_correction_interval=args.base_source_boundary_correction_interval,
        honey_microsteps_per_frame=args.honey_microsteps_per_frame,
        honey_source_boundary_correction_interval=args.honey_source_boundary_correction_interval,
    )
    output_paths = [
        args.output_dir / f"robotic_arm_pour_same_action_{variant.label}_mu_{_mu_slug(variant.viscosity)}.mp4"
        for variant in variants
    ]

    print(
        "same-action viscosity sweep: "
        f"{args.num_frames} frames at {mod.FRAME_RATE} fps "
        f"({args.num_frames / mod.FRAME_RATE:.2f} s)",
        flush=True,
    )
    print(f"shared settled cache: {args.cache_path}", flush=True)
    print(f"shared liquid color: {_color_arg(args.liquid_color)}", flush=True)
    print("variant solver settings:", flush=True)
    for variant in variants:
        print(
            f"  {variant.label}: mu={variant.viscosity:.6g}, "
            f"{variant.microsteps_per_frame} microsteps/frame, "
            f"dt={(1.0 / mod.FRAME_RATE) / variant.microsteps_per_frame:.8f} s, "
            f"correction interval={variant.source_boundary_correction_interval}",
            flush=True,
        )

    for index, (variant, output_path) in enumerate(zip(variants, output_paths)):
        needs_cache_bake = index == 0 and (args.rebake_cache or not args.cache_path.exists())
        if args.skip_existing and output_path.exists() and not needs_cache_bake:
            print(f"skip existing {output_path}", flush=True)
            continue

        command = [
            *_python_cmd(args),
            str(ROOT / "scripts" / "run_robotic_arm_pour_viscosity_genesis.py"),
            "--viscosity",
            f"{variant.viscosity:.15g}",
            "--num-frames",
            str(args.num_frames),
            "--cache-path",
            str(args.cache_path),
            "--output-path",
            str(output_path),
            "--liquid-color",
            _color_arg(variant.color),
            "--microsteps-per-frame",
            str(variant.microsteps_per_frame),
            "--source-boundary-correction-interval",
            str(variant.source_boundary_correction_interval),
            "--cuda-device",
            args.cuda_device,
        ]
        if args.liquid_vis_mode is not None:
            command.extend(["--liquid-vis-mode", args.liquid_vis_mode])
        if needs_cache_bake:
            command.append("--rebake")
        _run(command, dry_run=args.dry_run)

    if not args.dry_run:
        missing = [path for path in output_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("missing rendered videos: " + ", ".join(str(path) for path in missing))

    print("rendered videos:", flush=True)
    for output_path in output_paths:
        print(f"  {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
