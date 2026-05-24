"""Create a contiguous viscosity-band split from an existing manifest."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "outputs" / "viscosity_dataset_128"
DEFAULT_SOURCE_MANIFEST = DEFAULT_DATASET_DIR / "manifest.csv"
DEFAULT_OUTPUT_MANIFEST = DEFAULT_DATASET_DIR / "manifest_blocked_lowband.csv"
DEFAULT_SPLITS_DIR = DEFAULT_DATASET_DIR / "splits_blocked_lowband"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    return lo <= value <= hi


def _parse_bounds(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("bounds must be low,high")
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError("low must be less than high")
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument(
        "--test-log10-range",
        type=_parse_bounds,
        default=(-2.75, -2.50),
        help="Contiguous log10(mu) range assigned to test.",
    )
    parser.add_argument(
        "--val-log10-range",
        type=_parse_bounds,
        default=(-2.15, -1.95),
        help="Contiguous log10(mu) range assigned to validation.",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_csv(args.source_manifest)
    if not rows:
        raise ValueError(f"empty manifest: {args.source_manifest}")
    fieldnames = list(rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")

    output_rows: list[dict[str, str]] = []
    for row in rows:
        output = dict(row)
        log10_mu = float(row["log10_mu"])
        if _in_range(log10_mu, args.test_log10_range):
            split = "test"
        elif _in_range(log10_mu, args.val_log10_range):
            split = "val"
        else:
            split = "train"
        output["split"] = split
        output_rows.append(output)

    split_rows = {
        split: [row for row in output_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, split_row in split_rows.items():
        if not split_row:
            raise ValueError(f"empty split: {split}")

    _write_csv(args.output_manifest, output_rows, fieldnames=fieldnames)
    for split, split_row in split_rows.items():
        _write_csv(args.splits_dir / f"{split}.csv", split_row, fieldnames=fieldnames)

    summary = {
        "schema": "chinatown-viscosity-blocked-split-v1",
        "source_manifest": str(args.source_manifest),
        "output_manifest": str(args.output_manifest),
        "splits_dir": str(args.splits_dir),
        "test_log10_range": list(args.test_log10_range),
        "val_log10_range": list(args.val_log10_range),
        "split_counts": {split: len(split_row) for split, split_row in split_rows.items()},
        "split_ranges": {
            split: [
                min(float(row["log10_mu"]) for row in split_row),
                max(float(row["log10_mu"]) for row in split_row),
            ]
            for split, split_row in split_rows.items()
        },
    }
    summary_path = args.summary_path or args.output_manifest.with_suffix(".summary.json")
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
