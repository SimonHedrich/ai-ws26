"""
benchmark_quantized.py — Quantization experiments on MOT17 frames.

Tests three precision formats for yolov8m:
  - fp32  : default PyTorch inference (baseline)
  - fp16  : ONNX export with half=True, run via onnxruntime
              Works on macOS (CPU ONNX) and Jetson (CUDA EP)
  - int8  : Requires NVIDIA CUDA — skipped automatically on macOS
              Uses TensorRT INT8 export via ultralytics
  - trt   : TensorRT FP16 engine — CUDA only, skipped on macOS

Usage:
    python benchmark_quantized.py [--data data/MOT17/train] [--limit 200] [--device cpu]

Output:
    results/quantized.csv
"""

import argparse
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


def time_yolo_model(model: YOLO, frames: list[str], device: str, label: str) -> dict:
    """Warm-up then benchmark a YOLO model (any format supported by ultralytics)."""
    print(f"  [{label}] warming up …")
    for img in frames[:WARMUP]:
        model.predict(img, imgsz=IMGSZ, conf=CONF, device=device, verbose=False)

    bench = frames[WARMUP:]
    times_ms: list[float] = []
    print(f"  [{label}] benchmarking {len(bench)} frames …")
    for img in bench:
        t0 = time.perf_counter()
        model.predict(img, imgsz=IMGSZ, conf=CONF, device=device, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    return {
        "model": MODEL_NAME,
        "precision": label,
        "imgsz": IMGSZ,
        "conf": CONF,
        "device": device,
        "n_images": len(bench),
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "total_s": round(sum(times_ms) / 1000.0, 3),
    }


def benchmark_fp32(frames: list[str], device: str) -> dict:
    model = YOLO(MODEL_NAME)
    return time_yolo_model(model, frames, device, "fp32")


def benchmark_fp16_onnx(frames: list[str], device: str) -> dict:
    """Export to ONNX with half precision, benchmark via onnxruntime."""
    import cv2
    import onnxruntime as ort

    onnx_path = Path("results/yolov8m_fp16.onnx")
    if not onnx_path.exists():
        print("  [fp16] Exporting ONNX FP16 model …")
        model = YOLO(MODEL_NAME)
        model.export(format="onnx", half=True, imgsz=IMGSZ, dynamic=False)
        # ultralytics saves next to the .pt file — move to results/
        exported = Path(MODEL_NAME).with_suffix(".onnx")
        exported.rename(onnx_path)
    else:
        print(f"  [fp16] Reusing cached {onnx_path}")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device != "cpu"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name

    def infer(img_path: str):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMGSZ, IMGSZ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = (img.astype("float16") / 255.0).transpose(2, 0, 1)[None]
        sess.run(None, {input_name: blob})

    # Warm-up
    print("  [fp16] warming up …")
    for img in frames[:WARMUP]:
        infer(img)

    # Benchmark
    bench = frames[WARMUP:]
    times_ms: list[float] = []
    print(f"  [fp16] benchmarking {len(bench)} frames …")
    for img in bench:
        t0 = time.perf_counter()
        infer(img)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    return {
        "model": MODEL_NAME,
        "precision": "fp16_onnx",
        "imgsz": IMGSZ,
        "conf": CONF,
        "device": device,
        "n_images": len(bench),
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "total_s": round(sum(times_ms) / 1000.0, 3),
    }


def benchmark_tensorrt(frames: list[str], format: str, half: bool, int8: bool) -> dict:
    """Export and benchmark with TensorRT (CUDA required)."""
    label = "trt_int8" if int8 else "trt_fp16"
    engine_path = Path(f"results/yolov8m_{label}.engine")

    if not engine_path.exists():
        print(f"  [{label}] Exporting TensorRT engine (this may take several minutes) …")
        model = YOLO(MODEL_NAME)
        model.export(
            format="engine",
            imgsz=IMGSZ,
            half=half,
            int8=int8,
            data="config/mot17.yaml",  # needed for INT8 calibration
        )
        exported = Path(MODEL_NAME).with_suffix(".engine")
        exported.rename(engine_path)
    else:
        print(f"  [{label}] Reusing cached {engine_path}")

    trt_model = YOLO(str(engine_path))
    return time_yolo_model(trt_model, frames, device="0", label=label)


def main():
    parser = argparse.ArgumentParser(description="Quantization benchmark on MOT17")
    parser.add_argument("--data", default="data/MOT17/train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu | mps | 0 (CUDA index)")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    print(f"Device: {args.device}")

    frames = collect_frames(args.data, args.limit)
    print(f"Found {len(frames)} frames.\n")

    Path("results").mkdir(exist_ok=True)
    rows: list[dict] = []

    # ---- FP32 (always) ----
    print("=== FP32 ===")
    rows.append(benchmark_fp32(frames, args.device))

    # ---- FP16 via ONNX (Mac + Jetson) ----
    print("\n=== FP16 (ONNX) ===")
    try:
        rows.append(benchmark_fp16_onnx(frames, args.device))
    except Exception as exc:
        print(f"  [fp16_onnx] SKIPPED: {exc}")

    # ---- INT8 + TensorRT (Jetson / CUDA only) ----
    if has_cuda:
        print("\n=== TensorRT FP16 ===")
        try:
            rows.append(benchmark_tensorrt(frames, "engine", half=True, int8=False))
        except Exception as exc:
            print(f"  [trt_fp16] SKIPPED: {exc}")

        print("\n=== TensorRT INT8 ===")
        try:
            rows.append(benchmark_tensorrt(frames, "engine", half=False, int8=True))
        except Exception as exc:
            print(f"  [trt_int8] SKIPPED: {exc}")
    else:
        print("\n[INFO] CUDA not available — TensorRT and INT8 experiments skipped.")
        print("       Re-run this script on the Jetson with --device 0")

    out_path = Path("results/quantized.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
