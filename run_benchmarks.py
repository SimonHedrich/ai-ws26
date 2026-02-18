"""
run_benchmarks.py — Single entry point for all YOLO benchmarking experiments.

Edit the CONFIG section below to adjust models, paths, and experiment settings.
Then simply run:  python run_benchmarks.py
"""

import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# =============================================================================
# CONFIG — edit everything here
# =============================================================================

# Path to the MOT17 train directory (contains MOT17-XX-YYY/img1/*.jpg)
DATA_ROOT = "data/MOT17/train"

# Device: "cpu" | "mps" (Apple Silicon) | "0" (first CUDA GPU on Jetson)
DEVICE = "cpu"

# Number of warm-up frames to discard before timing starts
WARMUP = 10

# Maximum number of frames to use per experiment (None = all frames)
# Reduce for quick tests, set to None for full benchmark
FRAME_LIMIT = 200

# --- Baseline model comparison ---
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

# --- Parameter sweep (one-at-a-time, other two held at baseline) ---
PARAM_MODEL = "yolov8m.pt"          # model used for parameter sweep
PARAM_BASELINE = {"imgsz": 640, "conf": 0.25, "batch": 1}
PARAM_GRID = {
    "imgsz": [320, 640, 1280],
    "conf":  [0.1, 0.25, 0.5],
    "batch": [1, 4, 8],
}

# --- Quantization ---
QUANT_MODEL = "yolov8m.pt"          # model used for quantization experiments
QUANT_IMGSZ = 640
QUANT_CONF   = 0.25

# --- Pruning ---
PRUNE_MODEL  = "yolov8m.pt"         # model used for pruning experiments
PRUNE_RATIOS = [0.0, 0.3, 0.5]      # 0.0 = baseline (no pruning)
PRUNE_IMGSZ  = 640
PRUNE_CONF   = 0.25

# Which experiments to run (set to False to skip)
RUN_BASELINE    = True
RUN_PARAMS      = True
RUN_QUANTIZED   = True
RUN_PRUNING     = True

# =============================================================================
# Helpers
# =============================================================================

def collect_frames() -> list[str]:
    pattern = str(Path(DATA_ROOT) / "*/img1/*.jpg")
    frames = sorted(glob(pattern))
    if not frames:
        raise FileNotFoundError(
            f"No frames found at {pattern!r}.\n"
            f"Download MOT17 and extract it so that images live at:\n"
            f"  {DATA_ROOT}/MOT17-XX-YYY/img1/XXXXXX.jpg"
        )
    if FRAME_LIMIT:
        frames = frames[:FRAME_LIMIT]
    print(f"[frames] {len(frames)} frames from {DATA_ROOT!r}")
    return frames


def time_predict(model: YOLO, frames: list[str], imgsz: int = 640, conf: float = 0.25) -> tuple[float, float, float]:
    """Warm-up then measure per-image inference. Returns (mean_ms, std_ms, total_s)."""
    assert len(frames) > WARMUP, f"Need > {WARMUP} frames; got {len(frames)}."
    for img in frames[:WARMUP]:
        model.predict(img, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
    bench = frames[WARMUP:]
    times_ms = []
    for img in bench:
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(times_ms))
    std_ms  = float(np.std(times_ms))
    total_s = sum(times_ms) / 1000.0
    return mean_ms, std_ms, total_s


def save(df: pd.DataFrame, name: str):
    out = Path("results") / name
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    print(df.to_string(index=False))


# =============================================================================
# Experiment 1 — Baseline model comparison
# =============================================================================

def run_baseline(frames: list[str]):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: Baseline model comparison")
    print("=" * 65)
    rows = []
    for model_name in MODELS:
        print(f"\n  Model: {model_name}")
        try:
            model = YOLO(model_name)
            params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
            mean_ms, std_ms, total_s = time_predict(model, frames)
            print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img | {total_s:.1f} s | {params_m:.2f} M params")
            rows.append({
                "model":         model_name,
                "params_M":      round(params_m, 2),
                "imgsz":         640,
                "conf":          0.25,
                "batch":         1,
                "precision":     "fp32",
                "pruning_ratio": 0.0,
                "device":        DEVICE,
                "n_images":      len(frames) - WARMUP,
                "mean_ms":       round(mean_ms, 3),
                "std_ms":        round(std_ms, 3),
                "total_s":       round(total_s, 3),
            })
        except Exception as exc:
            print(f"  [SKIP] {exc}")
    save(pd.DataFrame(rows), "baseline.csv")


# =============================================================================
# Experiment 2 — Parameter sweep
# =============================================================================

def run_params(frames: list[str]):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: Parameter sweep (one-at-a-time)")
    print("=" * 65)
    model = YOLO(PARAM_MODEL)
    rows = []

    for param_name, values in PARAM_GRID.items():
        for value in values:
            kwargs = dict(PARAM_BASELINE)
            kwargs[param_name] = value
            imgsz = kwargs["imgsz"]
            conf  = kwargs["conf"]
            batch = kwargs["batch"]
            print(f"\n  {param_name}={value}  (others: baseline)")

            if batch == 1:
                mean_ms, std_ms, total_s = time_predict(model, frames, imgsz=imgsz, conf=conf)
            else:
                # Batch inference: time the whole batch, divide by n
                for img in frames[:WARMUP]:
                    model.predict(img, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
                bench = frames[WARMUP:]
                times_ms = []
                for i in range(0, len(bench), batch):
                    chunk = bench[i : i + batch]
                    t0 = time.perf_counter()
                    model.predict(chunk, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
                    per_img = (time.perf_counter() - t0) * 1000.0 / len(chunk)
                    times_ms.extend([per_img] * len(chunk))
                mean_ms = float(np.mean(times_ms))
                std_ms  = float(np.std(times_ms))
                total_s = sum(times_ms) / 1000.0

            print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img | {total_s:.1f} s")
            rows.append({
                "model":       PARAM_MODEL,
                "varied_param": param_name,
                "imgsz":       imgsz,
                "conf":        conf,
                "batch":       batch,
                "device":      DEVICE,
                "n_images":    len(frames) - WARMUP,
                "mean_ms":     round(mean_ms, 3),
                "std_ms":      round(std_ms, 3),
                "total_s":     round(total_s, 3),
            })

    save(pd.DataFrame(rows), "params.csv")


# =============================================================================
# Experiment 3 — Quantization
# =============================================================================

def run_quantized(frames: list[str]):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: Quantization")
    print("=" * 65)
    has_cuda = torch.cuda.is_available()
    rows = []

    # --- FP32 (always) ---
    print("\n  [fp32] baseline")
    model = YOLO(QUANT_MODEL)
    mean_ms, std_ms, total_s = time_predict(model, frames, imgsz=QUANT_IMGSZ, conf=QUANT_CONF)
    print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img")
    rows.append({"precision": "fp32", "mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3), "total_s": round(total_s, 3)})

    # --- FP16 via ONNX (Mac + Jetson) ---
    print("\n  [fp16_onnx] exporting & benchmarking")
    try:
        import cv2
        import onnxruntime as ort

        onnx_path = Path("results/quant_fp16.onnx")
        if not onnx_path.exists():
            m = YOLO(QUANT_MODEL)
            m.export(format="onnx", half=True, imgsz=QUANT_IMGSZ, dynamic=False)
            Path(QUANT_MODEL).with_suffix(".onnx").rename(onnx_path)

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda
                     else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        inp  = sess.get_inputs()[0].name

        def infer_fp16(path):
            img = cv2.imread(path)
            img = cv2.resize(img, (QUANT_IMGSZ, QUANT_IMGSZ))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = (img.astype("float16") / 255.0).transpose(2, 0, 1)[None]
            sess.run(None, {inp: blob})

        for img in frames[:WARMUP]:
            infer_fp16(img)
        bench = frames[WARMUP:]
        times_ms = []
        for img in bench:
            t0 = time.perf_counter()
            infer_fp16(img)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = float(np.mean(times_ms))
        std_ms  = float(np.std(times_ms))
        total_s = sum(times_ms) / 1000.0
        print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img")
        rows.append({"precision": "fp16_onnx", "mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3), "total_s": round(total_s, 3)})
    except Exception as exc:
        print(f"  [SKIP] fp16_onnx: {exc}")

    # --- TensorRT FP16 + INT8 (Jetson / CUDA only) ---
    if has_cuda:
        for label, half, int8 in [("trt_fp16", True, False), ("trt_int8", False, True)]:
            print(f"\n  [{label}]")
            try:
                eng_path = Path(f"results/quant_{label}.engine")
                if not eng_path.exists():
                    m = YOLO(QUANT_MODEL)
                    m.export(format="engine", imgsz=QUANT_IMGSZ, half=half, int8=int8,
                             data="config/mot17.yaml")
                    Path(QUANT_MODEL).with_suffix(".engine").rename(eng_path)
                trt_model = YOLO(str(eng_path))
                mean_ms, std_ms, total_s = time_predict(trt_model, frames, imgsz=QUANT_IMGSZ, conf=QUANT_CONF)
                print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img")
                rows.append({"precision": label, "mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3), "total_s": round(total_s, 3)})
            except Exception as exc:
                print(f"  [SKIP] {label}: {exc}")
    else:
        print("\n  [INFO] CUDA not available — TensorRT/INT8 skipped (run on Jetson with DEVICE='0')")

    df = pd.DataFrame(rows)
    df.insert(0, "model", QUANT_MODEL)
    df["device"] = DEVICE
    df["n_images"] = len(frames) - WARMUP
    save(df, "quantized.csv")


# =============================================================================
# Experiment 4 — Structured pruning
# =============================================================================

def run_pruning(frames: list[str]):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4: Structured channel pruning")
    print("=" * 65)
    try:
        import torch_pruning as tp
    except ImportError:
        print("  [SKIP] torch-pruning not installed. Run: pip install torch-pruning")
        return

    rows = []
    for ratio in PRUNE_RATIOS:
        print(f"\n  Pruning ratio: {ratio:.0%}")
        yolo = YOLO(PRUNE_MODEL)
        params_before = sum(p.numel() for p in yolo.model.parameters()) / 1e6

        if ratio > 0.0:
            example_input = torch.randn(1, 3, PRUNE_IMGSZ, PRUNE_IMGSZ)
            ignored = [m for m in yolo.model.modules()
                       if isinstance(m, (torch.nn.Linear,))]
            pruner = tp.pruner.MagnitudePruner(
                yolo.model,
                example_input,
                importance=tp.importance.MagnitudeImportance(p=1),
                pruning_ratio=ratio,
                ignored_layers=ignored,
            )
            pruner.step()
            params_after = sum(p.numel() for p in yolo.model.parameters()) / 1e6
            print(f"  Params: {params_before:.2f} M -> {params_after:.2f} M")
        else:
            params_after = params_before
            print("  Baseline — no pruning.")

        mean_ms, std_ms, total_s = time_predict(yolo, frames, imgsz=PRUNE_IMGSZ, conf=PRUNE_CONF)
        print(f"  -> {mean_ms:.2f} ± {std_ms:.2f} ms/img | {total_s:.1f} s")
        rows.append({
            "model":              PRUNE_MODEL,
            "pruning_ratio":      ratio,
            "params_before_M":    round(params_before, 2),
            "params_after_M":     round(params_after, 2),
            "device":             DEVICE,
            "n_images":           len(frames) - WARMUP,
            "mean_ms":            round(mean_ms, 3),
            "std_ms":             round(std_ms, 3),
            "total_s":            round(total_s, 3),
        })

    save(pd.DataFrame(rows), "pruning.csv")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("YOLO Benchmark Suite")
    print(f"  Data:   {DATA_ROOT}")
    print(f"  Device: {DEVICE}")
    print(f"  Warmup: {WARMUP} frames")
    print(f"  Limit:  {FRAME_LIMIT} frames")

    frames = collect_frames()

    if RUN_BASELINE:
        run_baseline(frames)

    if RUN_PARAMS:
        run_params(frames)

    if RUN_QUANTIZED:
        run_quantized(frames)

    if RUN_PRUNING:
        run_pruning(frames)

    print("\nAll done. Results in results/")
