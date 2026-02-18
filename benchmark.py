"""
benchmark.py — Baseline model comparison on MOT17 frames.

Compares YOLOv8, YOLO11, and YOLO12 (nano / small / medium) for person detection.
Measures per-image inference time in milliseconds, skipping the first WARMUP
frames to avoid cold-start / JIT-compilation bias.

Usage:
    python benchmark.py [--data data/MOT17/train] [--warmup 10] [--limit 200] [--device cpu]

Output:
    results/baseline.csv
"""

import argparse
import time
from glob import glob
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model list – update YOLO26 identifier if the installed ultralytics version
# uses a different name (e.g. yolo26n.pt vs yolov26n.pt).
# ---------------------------------------------------------------------------
MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
]

WARMUP = 10  # frames to skip at the start of each benchmark run


def collect_frames(data_root: str, limit: int | None = None) -> list[str]:
    """Return sorted list of .jpg frame paths from MOT17 train sequences."""
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


def benchmark_model(model_name: str, frames: list[str], device: str) -> dict:
    """Load a model and measure per-image inference time."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 60}")

    model = YOLO(model_name)

    # Parameter count
    total_params = sum(p.numel() for p in model.model.parameters())
    params_m = round(total_params / 1e6, 2)

    # ---- Warm-up ----
    print(f"  Warming up on {WARMUP} frames …")
    assert len(frames) > WARMUP, (
        f"Need more than {WARMUP} frames for warm-up; got {len(frames)}."
    )
    for img_path in frames[:WARMUP]:
        model.predict(img_path, device=device, verbose=False)

    # ---- Benchmark ----
    bench_frames = frames[WARMUP:]
    times_ms: list[float] = []
    print(f"  Benchmarking on {len(bench_frames)} frames …")
    for img_path in bench_frames:
        t0 = time.perf_counter()
        model.predict(img_path, device=device, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    total_s = sum(times_ms) / 1000.0

    result = {
        "model": model_name,
        "params_M": params_m,
        "imgsz": 640,
        "conf": 0.25,
        "batch": 1,
        "precision": "fp32",
        "pruning_ratio": 0.0,
        "device": device,
        "n_images": len(bench_frames),
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "total_s": round(total_s, 3),
    }
    print(
        f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img  |  "
        f"{total_s:.1f} s total  |  {params_m} M params"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Baseline YOLO model benchmark on MOT17")
    parser.add_argument("--data", default="data/MOT17/train", help="Path to MOT17 train directory")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Warm-up frames to skip")
    parser.add_argument("--limit", type=int, default=None, help="Max frames to use (None = all)")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, mps, 0 (CUDA)")
    args = parser.parse_args()

    frames = collect_frames(args.data, args.limit)
    print(f"Found {len(frames)} frames under {args.data!r}")

    rows = []
    for model_name in MODELS:
        try:
            row = benchmark_model(model_name, frames, args.device)
            rows.append(row)
        except Exception as exc:
            print(f"  [SKIP] {model_name}: {exc}")

    out_path = Path("results/baseline.csv")
    out_path.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
