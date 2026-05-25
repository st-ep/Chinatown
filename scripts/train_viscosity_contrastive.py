"""Train a compact viscosity video sanity model.

This first-pass model is intentionally small: a 3D CNN video encoder, a
property encoder for log10(mu), a soft continuous contrastive loss, and a
Huber regression head. It is meant to verify that the current dataset contains
a usable viscosity signal before moving to heavier pretrained video encoders.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chinatown_learning.dataset import REPO_ROOT, VideoClipConfig, ViscosityVideoDataset
from chinatown_learning.model import (
    ViscosityContrastiveModel,
    regression_loss,
    soft_contrastive_loss,
)


DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "viscosity_dataset_128" / "manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "viscosity_training" / "sanity_cnn"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_video(video: torch.Tensor) -> torch.Tensor:
    return (video - 0.5) / 0.5


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(np.asarray(x, dtype=np.float64))
    ry = _rankdata(np.asarray(y, dtype=np.float64))
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    target = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    error = pred - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(np.mean(error * error)))
    return {
        "mae_log10_mu": mae,
        "rmse_log10_mu": rmse,
        "typical_multiplicative_error": float(10.0**mae),
        "spearman": _spearman(targets, predictions),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "index",
        "viscosity_index",
        "run_id",
        "action_index",
        "action_id",
        "target_log10_mu",
        "pred_log10_mu",
        "retrieval_log10_mu",
        "abs_error_log10_mu",
        "retrieval_abs_error_log10_mu",
        "video_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_eval_clip_starts(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    starts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not starts:
        return None
    return starts


def _make_dataset(
    args: argparse.Namespace,
    *,
    split: str,
    train: bool,
    fixed_start_frame: int | None = None,
) -> ViscosityVideoDataset:
    config = VideoClipConfig(
        num_frames=args.clip_frames,
        frame_stride=args.frame_stride,
        image_size=args.image_size,
        temporal_start=args.temporal_start,
        temporal_end=args.temporal_end,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_size=args.crop_size,
        random_temporal_crop=train,
        fixed_start_frame=fixed_start_frame,
    )
    return ViscosityVideoDataset(
        args.manifest,
        split=split,
        repo_root=args.repo_root,
        clip_config=config,
        deterministic=not train,
        seed=args.seed,
    )


def _make_loader(
    dataset: ViscosityVideoDataset,
    *,
    args: argparse.Namespace,
    train: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=train and len(dataset) >= args.batch_size,
    )


def _run_epoch(
    model: ViscosityContrastiveModel,
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_contrast = 0.0
    total_reg = 0.0
    total_count = 0
    started = time.time()

    for batch_idx, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        video = _normalize_video(batch["video"].to(device, non_blocking=True))
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            output = model(video, target)
            reg = regression_loss(output["pred_log10_mu"], target, beta=args.huber_beta)
            if target.shape[0] > 1:
                contrast = soft_contrastive_loss(
                    output["video_embedding"],
                    output["property_embedding"],
                    target,
                    temperature=args.temperature,
                    sigma=args.sigma,
                )
            else:
                contrast = reg.new_tensor(0.0)
            loss = args.regression_weight * reg + args.contrastive_weight * contrast
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count = int(target.shape[0])
        total_count += count
        total_loss += float(loss.detach().cpu()) * count
        total_reg += float(reg.detach().cpu()) * count
        total_contrast += float(contrast.detach().cpu()) * count

    denom = max(total_count, 1)
    return {
        "epoch": float(epoch),
        "train_loss": total_loss / denom,
        "train_regression_loss": total_reg / denom,
        "train_contrastive_loss": total_contrast / denom,
        "train_examples": float(total_count),
        "train_seconds": time.time() - started,
    }


@torch.no_grad()
def _evaluate(
    model: ViscosityContrastiveModel,
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    device: torch.device,
    split: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    targets: list[float] = []
    predictions: list[float] = []
    retrieval_predictions: list[float] = []
    rows: list[dict[str, Any]] = []

    grid = torch.linspace(
        args.min_log10_mu,
        args.max_log10_mu,
        args.retrieval_grid_size,
        device=device,
        dtype=torch.float32,
    )
    grid_embedding = model.encode_property(grid)

    for batch_idx, batch in enumerate(loader):
        if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
            break
        video = _normalize_video(batch["video"].to(device, non_blocking=True))
        target = batch["target"].to(device, non_blocking=True)
        output = model(video, target)
        pred = output["pred_log10_mu"]
        similarity = output["video_embedding"] @ grid_embedding.T
        retrieval = grid[torch.argmax(similarity, dim=1)]

        target_cpu = target.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        retrieval_cpu = retrieval.detach().cpu().numpy()
        for i in range(target_cpu.shape[0]):
            target_value = float(target_cpu[i])
            pred_value = float(pred_cpu[i])
            retrieval_value = float(retrieval_cpu[i])
            targets.append(target_value)
            predictions.append(pred_value)
            retrieval_predictions.append(retrieval_value)
            rows.append(
                {
                    "split": split,
                    "index": int(batch["index"][i]),
                    "viscosity_index": int(batch["viscosity_index"][i]),
                    "run_id": batch["run_id"][i],
                    "action_index": int(batch["action_index"][i]),
                    "action_id": batch["action_id"][i],
                    "target_log10_mu": f"{target_value:.8f}",
                    "pred_log10_mu": f"{pred_value:.8f}",
                    "retrieval_log10_mu": f"{retrieval_value:.8f}",
                    "abs_error_log10_mu": f"{abs(pred_value - target_value):.8f}",
                    "retrieval_abs_error_log10_mu": f"{abs(retrieval_value - target_value):.8f}",
                    "video_path": batch["video_path"][i],
                }
            )

    metrics = {f"{split}_{key}": value for key, value in _metrics(targets, predictions).items()}
    retrieval_metrics = {
        f"{split}_retrieval_{key}": value
        for key, value in _metrics(targets, retrieval_predictions).items()
    }
    metrics.update(retrieval_metrics)
    metrics[f"{split}_examples"] = float(len(targets))
    return metrics, rows


@torch.no_grad()
def _evaluate_multiclip(
    model: ViscosityContrastiveModel,
    *,
    args: argparse.Namespace,
    device: torch.device,
    split: str,
    clip_starts: list[int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    accum: dict[int, dict[str, Any]] = {}
    grid = torch.linspace(
        args.min_log10_mu,
        args.max_log10_mu,
        args.retrieval_grid_size,
        device=device,
        dtype=torch.float32,
    )
    grid_embedding = model.encode_property(grid)

    for start_frame in clip_starts:
        dataset = _make_dataset(args, split=split, train=False, fixed_start_frame=start_frame)
        loader = _make_loader(dataset, args=args, train=False)
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_idx >= args.max_eval_batches:
                break
            video = _normalize_video(batch["video"].to(device, non_blocking=True))
            target = batch["target"].to(device, non_blocking=True)
            output = model(video, target)
            pred = output["pred_log10_mu"].detach().cpu()
            embedding = output["video_embedding"].detach().cpu()
            target_cpu = target.detach().cpu()
            for i in range(target_cpu.shape[0]):
                index = int(batch["index"][i])
                record = accum.setdefault(
                    index,
                    {
                        "split": split,
                        "index": index,
                        "viscosity_index": int(batch["viscosity_index"][i]),
                        "run_id": batch["run_id"][i],
                        "action_index": int(batch["action_index"][i]),
                        "action_id": batch["action_id"][i],
                        "target": float(target_cpu[i]),
                        "video_path": batch["video_path"][i],
                        "predictions": [],
                        "embeddings": [],
                    },
                )
                record["predictions"].append(float(pred[i]))
                record["embeddings"].append(embedding[i])

    targets: list[float] = []
    predictions: list[float] = []
    retrieval_predictions: list[float] = []
    rows: list[dict[str, Any]] = []
    for index in sorted(accum):
        record = accum[index]
        target_value = float(record["target"])
        pred_value = float(np.mean(record["predictions"]))
        mean_embedding = torch.stack(record["embeddings"], dim=0).mean(dim=0).to(device)
        mean_embedding = torch.nn.functional.normalize(mean_embedding, dim=0)
        similarity = mean_embedding[None, :] @ grid_embedding.T
        retrieval_value = float(grid[torch.argmax(similarity, dim=1)[0]].detach().cpu())
        targets.append(target_value)
        predictions.append(pred_value)
        retrieval_predictions.append(retrieval_value)
        rows.append(
            {
                "split": split,
                "index": index,
                "viscosity_index": record["viscosity_index"],
                "run_id": record["run_id"],
                "action_index": record["action_index"],
                "action_id": record["action_id"],
                "target_log10_mu": f"{target_value:.8f}",
                "pred_log10_mu": f"{pred_value:.8f}",
                "retrieval_log10_mu": f"{retrieval_value:.8f}",
                "abs_error_log10_mu": f"{abs(pred_value - target_value):.8f}",
                "retrieval_abs_error_log10_mu": f"{abs(retrieval_value - target_value):.8f}",
                "video_path": record["video_path"],
            }
        )

    metrics = {f"{split}_{key}": value for key, value in _metrics(targets, predictions).items()}
    retrieval_metrics = {
        f"{split}_retrieval_{key}": value
        for key, value in _metrics(targets, retrieval_predictions).items()
    }
    metrics.update(retrieval_metrics)
    metrics[f"{split}_examples"] = float(len(targets))
    metrics[f"{split}_eval_clips"] = float(len(clip_starts))
    return metrics, rows


def _evaluate_final_split(
    model: ViscosityContrastiveModel,
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    device: torch.device,
    split: str,
    eval_clip_starts: list[int] | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if eval_clip_starts is None:
        return _evaluate(model, loader, args=args, device=device, split=split)
    return _evaluate_multiclip(
        model,
        args=args,
        device=device,
        split=split,
        clip_starts=eval_clip_starts,
    )


def _save_checkpoint(
    path: Path,
    *,
    model: ViscosityContrastiveModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": serializable_args,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--temporal-start", type=int, default=40)
    parser.add_argument("--temporal-end", type=int, default=240)
    parser.add_argument("--crop-x", type=int, default=280)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=720)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--min-log10-mu", type=float, default=-3.0)
    parser.add_argument("--max-log10-mu", type=float, default=math.log10(0.03))
    parser.add_argument("--retrieval-grid-size", type=int, default=512)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint to evaluate when --epochs 0, or to override the final best checkpoint load.",
    )
    parser.add_argument(
        "--eval-clip-starts",
        default=None,
        help="Comma-separated fixed start frames for multi-clip final evaluation, e.g. 40,60,80,100,120.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args(argv)

    _set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "config.json", vars(args))
    eval_clip_starts = _parse_eval_clip_starts(args.eval_clip_starts)

    device = torch.device(args.device)
    train_dataset = _make_dataset(args, split="train", train=True)
    val_dataset = _make_dataset(args, split="val", train=False)
    test_dataset = _make_dataset(args, split="test", train=False)
    train_loader = _make_loader(train_dataset, args=args, train=True)
    val_loader = _make_loader(val_dataset, args=args, train=False)
    test_loader = _make_loader(test_dataset, args=args, train=False)

    model = ViscosityContrastiveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    best_val = float("inf")
    best_epoch = 0

    print(
        f"training sanity model: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}, device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            args=args,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
        )
        val_metrics, _ = _evaluate(model, val_loader, args=args, device=device, split="val")
        metrics = {**train_metrics, **val_metrics}
        _append_jsonl(args.output_dir / "metrics.jsonl", metrics)
        _save_checkpoint(
            args.output_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            metrics=metrics,
        )
        val_mae = metrics["val_mae_log10_mu"]
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            _save_checkpoint(
                args.output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                metrics=metrics,
            )
        print(
            f"epoch {epoch:03d} "
            f"loss={metrics['train_loss']:.4f} "
            f"val_mae_log10={metrics['val_mae_log10_mu']:.4f} "
            f"val_mult={metrics['val_typical_multiplicative_error']:.3f}",
            flush=True,
        )

    checkpoint_path = args.checkpoint_path or args.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state"])
    best_epoch = int(best_checkpoint.get("epoch", best_epoch))
    best_val = float(best_checkpoint.get("metrics", {}).get("val_mae_log10_mu", best_val))
    val_metrics, val_rows = _evaluate_final_split(
        model,
        val_loader,
        args=args,
        device=device,
        split="val",
        eval_clip_starts=eval_clip_starts,
    )
    test_metrics, test_rows = _evaluate_final_split(
        model,
        test_loader,
        args=args,
        device=device,
        split="test",
        eval_clip_starts=eval_clip_starts,
    )
    final_metrics = {
        **val_metrics,
        **test_metrics,
        "best_val_mae_log10_mu": best_val,
        "best_epoch": float(best_epoch),
    }
    _write_json(args.output_dir / "final_metrics.json", final_metrics)
    _write_predictions(args.output_dir / "predictions_val.csv", val_rows)
    _write_predictions(args.output_dir / "predictions_test.csv", test_rows)
    print(json.dumps(final_metrics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
