"""Merge generated viscosity shards, validate videos, and write train splits."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "outputs" / "viscosity_dataset_128"
MANIFEST_FIELDS = [
    "index",
    "run_id",
    "mu",
    "log10_mu",
    "seed",
    "microsteps_per_frame",
    "source_boundary_correction_interval",
    "video_path",
    "metadata_path",
    "split",
    "status",
    "error",
]


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _close(a: float, b: float, *, atol: float = 1.0e-9, rtol: float = 1.0e-6) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b), 1.0)


def _source_manifests(dataset_dir: Path) -> list[Path]:
    shard_paths = sorted(dataset_dir.glob("manifest_shard_*_of_*.csv"))
    if shard_paths:
        return shard_paths
    return [dataset_dir / "manifest.csv"]


def _merge_source_rows(paths: list[Path]) -> list[dict[str, str]]:
    by_index: dict[int, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in _read_csv(path):
            index = int(row["index"])
            previous = by_index.get(index)
            if previous is not None and previous["run_id"] != row["run_id"]:
                raise ValueError(f"conflicting run_id for index {index}: {previous['run_id']} vs {row['run_id']}")
            by_index[index] = row
    return [by_index[index] for index in sorted(by_index)]


def _assign_split(index: int, *, modulus: int, val_bucket: int, test_bucket: int) -> str:
    bucket = index % modulus
    if bucket == test_bucket:
        return "test"
    if bucket == val_bucket:
        return "val"
    return "train"


def _ffprobe(path: Path) -> VideoProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError("ffprobe found no video stream")
    stream = streams[0]
    frame_count_text = stream.get("nb_frames")
    frame_count = None if frame_count_text in (None, "N/A") else int(frame_count_text)
    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(Fraction(stream["r_frame_rate"])),
        frame_count=frame_count,
        duration=float(stream["duration"]),
    )


def _sample_frame_stats(path: Path, *, width: int, height: int, timestamp: float) -> dict[str, float]:
    expected_bytes = width * height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{max(timestamp, 0.0):.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    data = subprocess.check_output(command)
    if len(data) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} frame bytes, got {len(data)}")
    frame = np.frombuffer(data, dtype=np.uint8)
    return {
        "timestamp": float(timestamp),
        "mean": float(frame.mean()),
        "std": float(frame.std()),
    }


def _validate_row(
    row: dict[str, str],
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    expected_frames: int,
    expected_duration: float,
    duration_tolerance: float,
    blank_check_samples: int,
    blank_std_threshold: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    video_path = _resolve_path(row["video_path"])
    metadata_path = _resolve_path(row["metadata_path"])
    record: dict[str, Any] = {
        "index": int(row["index"]),
        "run_id": row["run_id"],
        "video_path": _display_path(video_path),
        "metadata_path": _display_path(metadata_path),
    }

    if not video_path.exists():
        errors.append("missing video")
        return record, errors, warnings
    if video_path.stat().st_size <= 0:
        errors.append("empty video file")
    if not metadata_path.exists():
        errors.append("missing metadata")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("run_id") != row["run_id"]:
                errors.append("metadata run_id does not match manifest")
            if int(metadata.get("index")) != int(row["index"]):
                errors.append("metadata index does not match manifest")
            if not _close(float(metadata.get("mu")), _parse_float(row, "mu")):
                errors.append("metadata mu does not match manifest")
            if not _close(float(metadata.get("log10_mu")), _parse_float(row, "log10_mu")):
                errors.append("metadata log10_mu does not match manifest")
        except Exception as exc:  # noqa: BLE001 - report all metadata parse issues.
            errors.append(f"metadata parse failed: {exc}")

    try:
        probe = _ffprobe(video_path)
        record["probe"] = {
            "width": probe.width,
            "height": probe.height,
            "fps": probe.fps,
            "frame_count": probe.frame_count,
            "duration": probe.duration,
        }
        if probe.width != expected_width:
            errors.append(f"width {probe.width} != {expected_width}")
        if probe.height != expected_height:
            errors.append(f"height {probe.height} != {expected_height}")
        if not _close(probe.fps, expected_fps, atol=1.0e-6, rtol=1.0e-6):
            errors.append(f"fps {probe.fps} != {expected_fps}")
        if probe.frame_count is not None and probe.frame_count != expected_frames:
            errors.append(f"frame_count {probe.frame_count} != {expected_frames}")
        if not _close(probe.duration, expected_duration, atol=duration_tolerance, rtol=0.0):
            errors.append(f"duration {probe.duration:.6f} != {expected_duration:.6f}")

        if blank_check_samples > 0 and shutil.which("ffmpeg") is not None:
            timestamps = np.linspace(
                0.0,
                max(probe.duration - 1.0 / max(probe.fps, 1.0), 0.0),
                blank_check_samples,
            )
            stats = [
                _sample_frame_stats(video_path, width=probe.width, height=probe.height, timestamp=float(t))
                for t in timestamps
            ]
            record["frame_stats"] = stats
            if max(stat["std"] for stat in stats) < blank_std_threshold:
                errors.append("sampled frames appear blank")
    except Exception as exc:  # noqa: BLE001 - keep validating other rows.
        errors.append(f"video probe failed: {exc}")

    return record, errors, warnings


def _canonicalize_row(row: dict[str, str], split: str) -> dict[str, str]:
    video_path = _resolve_path(row["video_path"])
    metadata_path = _resolve_path(row["metadata_path"])
    status = "complete" if video_path.exists() and metadata_path.exists() else row.get("status", "")
    return {
        "index": str(int(row["index"])),
        "run_id": row["run_id"],
        "mu": f"{float(row['mu']):.15g}",
        "log10_mu": f"{float(row['log10_mu']):.15g}",
        "seed": str(int(row["seed"])),
        "microsteps_per_frame": str(int(row["microsteps_per_frame"])),
        "source_boundary_correction_interval": str(int(row["source_boundary_correction_interval"])),
        "video_path": _display_path(video_path),
        "metadata_path": _display_path(metadata_path),
        "split": split,
        "status": status,
        "error": row.get("error", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--qa-report-path", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--expected-width", type=int, default=1280)
    parser.add_argument("--expected-height", type=int, default=720)
    parser.add_argument("--expected-fps", type=float, default=60.0)
    parser.add_argument("--expected-frames", type=int, default=276)
    parser.add_argument("--expected-duration", type=float, default=4.6)
    parser.add_argument("--duration-tolerance", type=float, default=0.02)
    parser.add_argument("--blank-check-samples", type=int, default=3)
    parser.add_argument("--blank-std-threshold", type=float, default=1.0)
    parser.add_argument("--split-modulus", type=int, default=10)
    parser.add_argument("--val-bucket", type=int, default=5)
    parser.add_argument("--test-bucket", type=int, default=0)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir
    manifest_path = args.manifest_path or dataset_dir / "manifest.csv"
    qa_report_path = args.qa_report_path or dataset_dir / "qa_report.json"
    splits_dir = args.splits_dir or dataset_dir / "splits"

    if args.split_modulus <= 1:
        parser.error("--split-modulus must be greater than 1")
    if args.val_bucket == args.test_bucket:
        parser.error("--val-bucket and --test-bucket must differ")
    if not (0 <= args.val_bucket < args.split_modulus):
        parser.error("--val-bucket must be in [0, split-modulus)")
    if not (0 <= args.test_bucket < args.split_modulus):
        parser.error("--test-bucket must be in [0, split-modulus)")

    source_paths = _source_manifests(dataset_dir)
    source_rows = _merge_source_rows(source_paths)
    canonical_rows: list[dict[str, str]] = []
    qa_records: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []

    for row in source_rows:
        index = int(row["index"])
        split = _assign_split(
            index,
            modulus=args.split_modulus,
            val_bucket=args.val_bucket,
            test_bucket=args.test_bucket,
        )
        canonical = _canonicalize_row(row, split)
        record, errors, warnings = _validate_row(
            canonical,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            expected_fps=args.expected_fps,
            expected_frames=args.expected_frames,
            expected_duration=args.expected_duration,
            duration_tolerance=args.duration_tolerance,
            blank_check_samples=args.blank_check_samples,
            blank_std_threshold=args.blank_std_threshold,
        )
        if errors:
            canonical["status"] = "invalid"
            canonical["error"] = "; ".join(errors)
            all_errors.append({"index": index, "run_id": canonical["run_id"], "errors": errors})
        if warnings:
            all_warnings.append({"index": index, "run_id": canonical["run_id"], "warnings": warnings})
        qa_records.append(record)
        canonical_rows.append(canonical)

    split_counts = {
        split: sum(1 for row in canonical_rows if row["split"] == split)
        for split in ("train", "val", "test")
    }
    _write_csv(manifest_path, canonical_rows)
    for split in ("train", "val", "test"):
        _write_csv(splits_dir / f"{split}.csv", [row for row in canonical_rows if row["split"] == split])

    report = {
        "schema": "chinatown-viscosity-dataset-qa-v1",
        "dataset_dir": _display_path(dataset_dir),
        "source_manifests": [_display_path(path) for path in source_paths],
        "manifest_path": _display_path(manifest_path),
        "splits_dir": _display_path(splits_dir),
        "num_rows": len(canonical_rows),
        "split_counts": split_counts,
        "num_errors": len(all_errors),
        "num_warnings": len(all_warnings),
        "errors": all_errors,
        "warnings": all_warnings,
        "videos": qa_records,
    }
    _write_json(qa_report_path, report)

    print(f"wrote manifest: {manifest_path}")
    print(f"wrote QA report: {qa_report_path}")
    print(f"split counts: {split_counts}")
    if all_errors:
        print(f"QA errors: {len(all_errors)}", file=sys.stderr)
        return 0 if args.allow_errors else 1
    print("QA passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
