"""PyTorch dataset for Chinatown viscosity video manifests."""
from __future__ import annotations

import csv
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VideoClipConfig:
    num_frames: int = 16
    frame_stride: int = 8
    image_size: int = 224
    temporal_start: int = 40
    temporal_end: int = 240
    crop_x: int = 280
    crop_y: int = 0
    crop_size: int = 720
    random_temporal_crop: bool = True
    fixed_start_frame: int | None = None

    @property
    def span(self) -> int:
        return (self.num_frames - 1) * self.frame_stride


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _sample_start(config: VideoClipConfig, *, rng: random.Random | None) -> int:
    latest_start = max(config.temporal_start, config.temporal_end - config.span)
    if config.random_temporal_crop and latest_start > config.temporal_start:
        if rng is None:
            return random.randint(config.temporal_start, latest_start)
        return rng.randint(config.temporal_start, latest_start)
    return (config.temporal_start + latest_start) // 2


def read_clip_ffmpeg(video_path: Path, *, start_frame: int, config: VideoClipConfig) -> torch.Tensor:
    """Read a clip as a CxTxHxW float tensor in [0, 1] using ffmpeg."""
    end_frame = start_frame + config.span
    select_expr = (
        f"select='between(n\\,{start_frame}\\,{end_frame})"
        f"*not(mod(n-{start_frame}\\,{config.frame_stride}))'"
    )
    filters = (
        f"{select_expr},"
        f"crop={config.crop_size}:{config.crop_size}:{config.crop_x}:{config.crop_y},"
        f"scale={config.image_size}:{config.image_size}"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filters,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    data = subprocess.check_output(command)
    frame_bytes = config.image_size * config.image_size * 3
    if len(data) % frame_bytes != 0:
        raise ValueError(f"decoded raw byte count is not frame-aligned for {video_path}")
    decoded_frames = len(data) // frame_bytes
    if decoded_frames == 0:
        raise ValueError(f"ffmpeg decoded no frames from {video_path}")
    array = np.frombuffer(data, dtype=np.uint8).reshape(
        decoded_frames,
        config.image_size,
        config.image_size,
        3,
    )
    if decoded_frames < config.num_frames:
        pad = np.repeat(array[-1:, :, :, :], config.num_frames - decoded_frames, axis=0)
        array = np.concatenate([array, pad], axis=0)
    elif decoded_frames > config.num_frames:
        array = array[: config.num_frames]
    tensor = torch.from_numpy(array.copy()).permute(3, 0, 1, 2).float() / 255.0
    return tensor


class ViscosityVideoDataset(Dataset):
    """Dataset that returns cropped temporal clips and log-viscosity labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        repo_root: str | Path = REPO_ROOT,
        clip_config: VideoClipConfig | None = None,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repo_root = Path(repo_root)
        self.clip_config = clip_config or VideoClipConfig()
        self.deterministic = deterministic
        self.seed = seed
        rows = _read_manifest(self.manifest_path)
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        self.rows = rows
        if not self.rows:
            raise ValueError(f"no rows found for split={split!r} in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        config = self.clip_config
        rng = random.Random(self.seed + index) if self.deterministic else None
        if config.fixed_start_frame is None:
            start_frame = _sample_start(config, rng=rng)
        else:
            start_frame = config.fixed_start_frame
        video_path = _resolve_path(row["video_path"], self.repo_root)
        video = read_clip_ffmpeg(video_path, start_frame=start_frame, config=config)
        target = torch.tensor(float(row["log10_mu"]), dtype=torch.float32)
        mu = torch.tensor(float(row["mu"]), dtype=torch.float32)
        return {
            "video": video,
            "target": target,
            "mu": mu,
            "index": int(row["index"]),
            "run_id": row["run_id"],
            "video_path": str(video_path),
            "start_frame": start_frame,
        }
