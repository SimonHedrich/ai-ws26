"""
pruning.py — Structured channel pruning experiment on yolov8m.

Uses the `torch-pruning` library (pip install torch-pruning) to apply L1
magnitude-based structured pruning to yolov8m at three sparsity ratios:
  - 0.0  (baseline — no pruning)
  - 0.3  (30 % of channels removed)
  - 0.5  (50 % of channels removed)

No fine-tuning is performed — the goal is to observe the raw effect of
pruning on inference speed and parameter count.

Usage:
    python pruning.py [--data data/MOT17/train] [--limit 200] [--device cpu]

Output:
    results/pruning.csv
    results/pruned_<ratio>.pt   (saved pruned weights)

Requirements:
    pip install torch-pruning
"""

import argparse
import copy
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

MODEL_NAME = "yolov8m.pt"
WARMUP = 10
IMGSZ = 640
CONF = 0.25
PRUNING_RATIOS = [0.0, 0.3, 0.5]


def collect_frames(data_root: str, limit: int | None = None) -> list[str]:
    pattern = str(Path(data_root) / "*/img1/*.jpg")
    frames = sorted(glob(pattern))
    if not frames:
        raise FileNotFoundError(
            f"No frames found at {pattern!r}.\n"
            "Download MOT17 and place it at data/MOT17/train/ (see README)."
        )
    if limit:
        frames = frames[:limit]
    return frames


def measure_inference(model: YOLO, frames: list[str], device: str) -> tuple[float, float, float]:
    """Return (mean_ms, std_ms, total_s) after warm-up."""
    for img in frames[:WARMUP]:
        model.predict(img, imgsz=IMGSZ, conf=CONF, device=device, verbose=False)
    bench = frames[WARMUP:]
    times_ms: list[float] = []
    for img in bench:
        t0 = time.perf_counter()
        model.predict(img, imgsz=IMGSZ, conf=CONF, device=device, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    total_s = sum(times_ms) / 1000.0
    return mean_ms, std_ms, total_s


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def apply_pruning(torch_model: torch.nn.Module, ratio: float) -> torch.nn.Module:
    """Apply L1 structured channel pruning at the given sparsity ratio."""
    try:
        import torch_pruning as tp
    except ImportError:
        raise ImportError(
            "torch-pruning is required for pruning experiments.\n"
            "Install it with: pip install torch-pruning"
        )

    example_input = torch.randn(1, 3, IMGSZ, IMGSZ)

    # Identify the detection head layers to keep intact
    ignored_layers = []
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)
        # Keep the final detection head Conv layers
        if hasattr(m, "cv3") or hasattr(m, "dfl"):
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        torch_model,
        example_input,
        importance=tp.importance.MagnitudeImportance(p=1),  # L1 norm
        iterative_steps=1,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return torch_model


def main():
    parser = argparse.ArgumentParser(description="Pruning experiment on MOT17")
    parser.add_argument("--data", default="data/MOT17/train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    frames = collect_frames(args.data, args.limit)
    print(f"Found {len(frames)} frames.\n")

    Path("results").mkdir(exist_ok=True)
    rows: list[dict] = []

    for ratio in PRUNING_RATIOS:
        print(f"\n{'=' * 60}")
        print(f"  Pruning ratio: {ratio:.0%}")
        print(f"{'=' * 60}")

        # Load a fresh copy of the model for each ratio
        yolo_model = YOLO(MODEL_NAME)
        params_before = count_params(yolo_model.model)

        if ratio > 0.0:
            print(f"  Applying {ratio:.0%} channel pruning …")
            yolo_model.model = apply_pruning(yolo_model.model, ratio)
            params_after = count_params(yolo_model.model)
            print(f"  Params: {params_before:.2f} M  ->  {params_after:.2f} M")

            # Save pruned weights
            pruned_path = Path(f"results/pruned_{int(ratio * 100)}.pt")
            torch.save(yolo_model.model.state_dict(), pruned_path)
        else:
            params_after = params_before
            print("  Baseline (no pruning).")

        # Measure inference
        print(f"  Benchmarking …")
        mean_ms, std_ms, total_s = measure_inference(yolo_model, frames, args.device)

        rows.append(
            {
                "model": MODEL_NAME,
                "pruning_ratio": ratio,
                "params_before_M": round(params_before, 2),
                "params_after_M": round(params_after, 2),
                "imgsz": IMGSZ,
                "conf": CONF,
                "device": args.device,
                "n_images": len(frames) - WARMUP,
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(std_ms, 3),
                "total_s": round(total_s, 3),
            }
        )
        print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img  |  total {total_s:.1f} s")

    out_path = Path("results/pruning.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
