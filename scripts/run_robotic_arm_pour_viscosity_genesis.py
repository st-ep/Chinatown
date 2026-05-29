"""Run the Genesis robot-arm pour with a chosen liquid viscosity.

This is the reusable entry point used by the three-liquid same-action sweep.
Run from the ``genesis-sim`` conda env.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "src" / "chinatown" / "robotic_arm_pour_genesis.py"
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_custom_viscosity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _configure_viscosity(
    mod,
    viscosity: float,
    cache_path: Path,
    liquid_color: tuple[float, float, float, float] | None,
    microsteps_per_frame: int | None,
    source_boundary_correction_interval: int | None,
) -> None:
    if microsteps_per_frame is not None:
        mod.MICROSTEPS_PER_FRAME = int(microsteps_per_frame)
        mod.PHYSICS_DT = mod.VIDEO_DT / mod.MICROSTEPS_PER_FRAME
    if source_boundary_correction_interval is not None:
        mod.SOURCE_BOUNDARY_CORRECTION_INTERVAL = int(source_boundary_correction_interval)
    mod.WATER_VISCOSITY = float(viscosity)
    mod.LIQUID_COLOR = liquid_color or (0.35, 0.62, 0.90, 1.0)
    mod.SETTLED_PARTICLES_CACHE = cache_path


def _configure_glass_mesh(mod, glass_mesh_path: Path | None) -> None:
    if glass_mesh_path is None:
        return
    mod.GLASS_MESH_PATH = glass_mesh_path
    mod.build_glass_mesh(glass_mesh_path)


def _configure_action(
    mod,
    *,
    tilt_seconds: float | None,
    return_seconds: float | None,
    pour_pose_fraction: float | None,
) -> None:
    if tilt_seconds is not None:
        mod.TILT_SECONDS = float(tilt_seconds)
    if return_seconds is not None:
        mod.RETURN_SECONDS = float(return_seconds)
    if pour_pose_fraction is not None:
        mod.POUR_POSE_FRACTION = float(pour_pose_fraction)
        mod.PANDA_Q_POUR = (
            mod.PANDA_Q_UPRIGHT
            + (mod.PANDA_Q_FULL_POUR - mod.PANDA_Q_UPRIGHT) * mod.POUR_POSE_FRACTION
        )
    if tilt_seconds is not None or return_seconds is not None:
        mod.VIDEO_NUM_FRAMES = int(round((mod.TILT_SECONDS + mod.RETURN_SECONDS) * mod.FRAME_RATE))


def _configure_action_program(mod, path: Path | None) -> None:
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    keyframes = [
        (float(frame["time_seconds"]), float(frame["pose_fraction"]))
        for frame in payload["keyframes"]
    ]
    mod.configure_pour_action_program(keyframes, action_id=payload.get("action_id"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--viscosity", type=float, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Settled-particle cache. Defaults to the loaded Genesis module's current cache path.",
    )
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video rendering; run the scene and print particle transfer metrics.",
    )
    parser.add_argument(
        "--rebake",
        action="store_true",
        help="Rebuild the chosen settled starting cache before running.",
    )
    parser.add_argument(
        "--liquid-vis-mode",
        choices=["particle", "recon"],
        default=None,
        help="Genesis liquid visualization mode.",
    )
    parser.add_argument(
        "--liquid-color",
        type=_parse_color,
        default=None,
        help="Liquid render color as R,G,B or R,G,B,A with channels in [0, 1].",
    )
    parser.add_argument(
        "--microsteps-per-frame",
        type=int,
        default=None,
        help="Override physics microsteps per 60 Hz video frame before scene construction.",
    )
    parser.add_argument(
        "--source-boundary-correction-interval",
        type=int,
        default=None,
        help="Override how often source-cup particle boundary correction runs within microsteps.",
    )
    parser.add_argument(
        "--cuda-device",
        default="1",
        help="CUDA device id to expose through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--tilt-seconds",
        type=float,
        default=None,
        help="Override the time spent moving from upright to pouring pose.",
    )
    parser.add_argument(
        "--return-seconds",
        type=float,
        default=None,
        help="Override the time spent returning from pouring pose to upright.",
    )
    parser.add_argument(
        "--pour-pose-fraction",
        type=float,
        default=None,
        help="Override interpolation fraction from upright to the calibrated full-pour robot pose.",
    )
    parser.add_argument(
        "--action-program-path",
        type=Path,
        default=None,
        help="JSON file containing piecewise action keyframes.",
    )
    parser.add_argument(
        "--glass-mesh-path",
        type=Path,
        default=None,
        help="OBJ mesh path used for both moving and receiving glasses.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional CSV path for per-frame privileged particle metrics.",
    )
    parser.add_argument(
        "--action-trace-path",
        type=Path,
        default=None,
        help="Optional CSV path for commanded action trace.",
    )
    args = parser.parse_args(argv)

    if args.viscosity <= 0.0:
        parser.error("--viscosity must be positive")
    if args.tilt_seconds is not None and args.tilt_seconds <= 0.0:
        parser.error("--tilt-seconds must be positive")
    if args.return_seconds is not None and args.return_seconds <= 0.0:
        parser.error("--return-seconds must be positive")
    if args.pour_pose_fraction is not None and not (0.0 < args.pour_pose_fraction <= 1.0):
        parser.error("--pour-pose-fraction must be in (0, 1]")
    if args.action_program_path is not None and not args.action_program_path.exists():
        parser.error("--action-program-path does not exist")
    if args.microsteps_per_frame is not None and args.microsteps_per_frame <= 0:
        parser.error("--microsteps-per-frame must be positive")
    if (
        args.source_boundary_correction_interval is not None
        and args.source_boundary_correction_interval <= 0
    ):
        parser.error("--source-boundary-correction-interval must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    mod = _load_module()
    cache_path = args.cache_path or mod.SETTLED_PARTICLES_CACHE
    _configure_glass_mesh(mod, args.glass_mesh_path)
    _configure_viscosity(
        mod,
        args.viscosity,
        cache_path,
        args.liquid_color,
        args.microsteps_per_frame,
        args.source_boundary_correction_interval,
    )
    _configure_action(
        mod,
        tilt_seconds=args.tilt_seconds,
        return_seconds=args.return_seconds,
        pour_pose_fraction=args.pour_pose_fraction,
    )
    _configure_action_program(mod, args.action_program_path)
    if args.liquid_vis_mode is not None:
        mod.LIQUID_VIS_MODE = args.liquid_vis_mode
    num_frames = mod.VIDEO_NUM_FRAMES if args.num_frames is None else args.num_frames
    if num_frames <= 0:
        parser.error("--num-frames must be positive")

    if args.no_video:
        result = mod.run_simulation(
            num_frames=num_frames,
            settled_cache=cache_path,
            rebake=args.rebake,
        )
        print("viscosity:        ", f"{args.viscosity:.6g}")
        print("particles initial:", result.initial_particle_count)
        print("particles live:   ", result.final_live_particles)
        print("in pourer:        ", result.final_particles_in_pourer, f"({result.pourer_fraction:.2%})")
        print("in receiver:      ", result.final_particles_in_receiver, f"({result.receiver_fraction:.2%})")
        print("max tilt deg:     ", f"{result.max_tilt_degrees:.1f}")
        print("final tilt deg:   ", f"{result.final_tilt_degrees:.1f}")
        print("solid violations: ", result.max_glass_solid_particles)
        print("  upper glass:    ", result.max_pourer_solid_particles)
        print("  upper base:     ", result.max_pourer_base_particles)
        print("  receiver glass: ", result.max_receiver_solid_particles)
        return 0

    output_path = args.output_path or f"outputs/robotic_arm_pour_mu_{args.viscosity:.4g}.mp4"
    output = mod.render_video(
        output_path=output_path,
        num_frames=num_frames,
        settled_cache=cache_path,
        rebake=args.rebake,
        metrics_path=args.metrics_path,
        action_trace_path=args.action_trace_path,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
