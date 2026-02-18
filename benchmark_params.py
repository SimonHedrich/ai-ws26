"""
benchmark_params.py — One-at-a-time parameter sweep on MOT17 frames.

Holds a representative model (yolov8m) fixed and varies one parameter at a time:
  - imgsz : [320, 640, 1280]  — image resolution fed to the model
  - conf  : [0.1, 0.25, 0.5]  — confidence threshold
  - batch : [1, 4, 8]         — images processed per forward pass

Baseline: imgsz=640, conf=0.25, batch=1

Usage:
    python benchmark_params.py [--data data/MOT17/train] [--warmup 10] [--limit 200] [--device cpu]

Output:
    results/params.csv
"""

import argparse
import time
from glob import glob
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

MODEL_NAME = "yolov8m.pt"
WARMUP = 10

# Baseline values
BASELINE = {"imgsz": 640, "conf": 0.25, "batch": 1}

# Parameter grid – only one parameter varies per experiment
PARAM_GRID = {
    "imgsz": [320, 640, 1280],
    "conf": [0.1, 0.25, 0.5],
    "batch": [1, 4, 8],
}


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


def run_experiment(
    model: YOLO,
    frames: list[str],
    imgsz: int,
    conf: float,
    batch: int,
    device: str,
) -> dict:
    """Warm up then benchmark with the given hyper-parameters."""
    assert len(frames) > WARMUP

    # Warm-up
    for img_path in frames[:WARMUP]:
        model.predict(img_path, imgsz=imgsz, conf=conf, device=device, verbose=False)

    bench_frames = frames[WARMUP:]
    times_ms: list[float] = []

    if batch == 1:
        for img_path in bench_frames:
            t0 = time.perf_counter()
            model.predict(img_path, imgsz=imgsz, conf=conf, device=device, verbose=False)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    else:
        # Batch inference: collect batches, time the whole batch, divide by batch size
        for i in range(0, len(bench_frames), batch):
            batch_paths = bench_frames[i : i + batch]
            t0 = time.perf_counter()
            model.predict(batch_paths, imgsz=imgsz, conf=conf, device=device, verbose=False)
            elapsed_per_img = (time.perf_counter() - t0) * 1000.0 / len(batch_paths)
            times_ms.extend([elapsed_per_img] * len(batch_paths))

    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    total_s = sum(times_ms) / 1000.0

    return {
        "model": MODEL_NAME,
        "imgsz": imgsz,
        "conf": conf,
        "batch": batch,
        "device": device,
        "n_images": len(bench_frames),
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "total_s": round(total_s, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep benchmark on MOT17")
    parser.add_argument("--data", default="data/MOT17/train")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    frames = collect_frames(args.data, args.limit)
    print(f"Found {len(frames)} frames. Model: {MODEL_NAME}")

    model = YOLO(MODEL_NAME)
    rows = []

    for param_name, values in PARAM_GRID.items():
        for value in values:
            # Build kwargs with baseline except for the swept parameter
            kwargs = dict(BASELINE)
            kwargs[param_name] = value
            label = f"{param_name}={value}"
            print(f"\n  Experiment: {label}  (other params at baseline)")

            row = run_experiment(model, frames, device=args.device, **kwargs)
            row["varied_param"] = param_name
            rows.append(row)
            print(
                f"  -> {row['mean_ms']:.2f} ± {row['std_ms']:.2f} ms/img"
                f"  |  total {row['total_s']:.1f} s"
            )

    out_path = Path("results/params.csv")
    out_path.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
