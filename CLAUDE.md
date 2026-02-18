# CLAUDE.md — Project Context for ai-ws26

## Exercise Background

This project is a hands-on introduction to the **Ultralytics YOLO** framework,
focused on person detection and benchmarking for deployment on an **NVIDIA Jetson**.
Development happens on macOS; the scripts are written so they transfer directly
to the Jetson without modification (change `DEVICE` in the config block).

### Task 1 — Native on the Jetson
1. Compare detection models (YOLOv8, YOLO11, YOLO26) across sizes (nano/small/medium).
   Measure inference time per image and total time, discarding the first 10 frames as warm-up.
2. Repeat with FP16 and TensorRT.
3. Experiment with quantization (FP32, FP16, INT8) and structured pruning.
4. Investigate whether changing the Jetson energy/power mode affects benchmark results.

### Task 2 — Docker
Wrap the same scripts in a Docker container (`nvcr.io/nvidia/l4t-pytorch` base)
and repeat Task 1 measurements inside the container.

### Dataset: MOT17
- Source: https://motchallenge.net/data/MOT17/
- Contains pedestrian tracking sequences; only one class: `person`
- Local path: `data/MOT17/train/MOT17-XX-YYY/img1/*.jpg`
- Labels are **not** in YOLO format — the dataset is used for inference benchmarks only,
  not for training. Fine-tuning (e.g. after pruning) uses `coco8.yaml`.

---

## Repository Layout

```
ai-ws26/
├── CLAUDE.md                  ← this file
├── README.md                  ← user-facing setup and usage guide
├── requirements.txt           ← pip dependencies
├── run_benchmarks.py          ← SINGLE ENTRY POINT — run this
├── benchmark.py               ← standalone: baseline model comparison
├── benchmark_params.py        ← standalone: parameter sweep
├── benchmark_quantized.py     ← standalone: FP16/INT8/TensorRT
├── pruning.py                 ← standalone: structured channel pruning
├── config/
│   └── mot17.yaml             ← ultralytics dataset YAML (inference/TRT calibration only)
├── data/
│   └── MOT17/
│       └── train/
│           └── MOT17-XX-YYY/img1/*.jpg   ← actual images (gitignored)
├── results/                   ← CSV outputs (gitignored)
│   ├── baseline.csv
│   ├── params.csv
│   └── quantized.csv
└── *.pt                       ← downloaded model weights (gitignored)
```

The **only file that needs to be run** is `run_benchmarks.py`. The other
`benchmark_*.py` / `pruning.py` files are standalone equivalents kept for
reference; all their logic is also in `run_benchmarks.py`.

---

## How to Run

```bash
pip install -r requirements.txt
python run_benchmarks.py
```

All configuration is in the `CONFIG` block at the top of `run_benchmarks.py` —
no CLI arguments. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `"data/MOT17/train"` | Path to MOT17 train sequences |
| `DEVICE` | `"cpu"` | `"cpu"` / `"mps"` (Mac) / `"0"` (Jetson CUDA) |
| `WARMUP` | `10` | Frames discarded before timing starts |
| `FRAME_LIMIT` | `200` | Cap on frames per experiment (`None` = all) |
| `RUN_BASELINE` | `True` | Toggle each experiment on/off |
| `RUN_PARAMS` | `True` | |
| `RUN_QUANTIZED` | `True` | |
| `RUN_PRUNING` | `True` | |

---

## Experiments Implemented

### 1. Baseline model comparison (`run_baseline`)
Benchmarks all nine models (three versions × three sizes) at default settings.

| Version | nano | small | medium |
|---------|------|-------|--------|
| YOLOv8  | yolov8n.pt | yolov8s.pt | yolov8m.pt |
| YOLO11  | yolo11n.pt | yolo11s.pt | yolo11m.pt |
| YOLO26  | yolo26n.pt | yolo26s.pt | yolo26m.pt |

Output: `results/baseline.csv`

### 2. Parameter sweep (`run_params`)
One-at-a-time sweep on `yolov8m`. Baseline: `imgsz=640, conf=0.25, batch=1`.

| Parameter | Values | Effect |
|-----------|--------|--------|
| `imgsz` | 320 / 640 / 1280 | Biggest accuracy vs. speed lever |
| `conf` | 0.1 / 0.25 / 0.5 | Detection sensitivity / NMS load |
| `batch` | 1 / 4 / 8 | Throughput vs. latency tradeoff |

Output: `results/params.csv`

### 3. Quantization (`run_quantized`)
Tests `yolov8m` at different precision levels. Hardware availability is detected
automatically; unsupported formats are skipped with an explanation.

| Format | macOS | Jetson | How |
|--------|-------|--------|-----|
| fp32 | ✓ | ✓ | Default PyTorch |
| fp16_onnx | ✓ | ✓ | ONNX export with `half=True`, run via onnxruntime |
| trt_fp16 | ✗ | ✓ | TensorRT engine, `format="engine", half=True` |
| trt_int8 | ✗ | ✓ | TensorRT INT8, `format="engine", int8=True`, calibrated on MOT17 |

Output: `results/quantized.csv`

### 4. Structured pruning (`run_pruning`)
Uses the `torch-pruning` library (L1 magnitude, channel-level) on `yolov8m`
at three sparsity ratios. Fine-tunes with `coco8.yaml` after pruning.

| Ratio | Effect |
|-------|--------|
| 0.0 | Baseline — no pruning |
| 0.3 | 30 % of channels removed |
| 0.5 | 50 % of channels removed |

Output: `results/pruning.csv`

---

## Results So Far (macOS CPU, 190 frames, FRAME_LIMIT=200)

### Baseline
| Model | Params (M) | Mean ms/img | Std ms |
|-------|-----------|-------------|--------|
| yolov8n | 3.16 | 45.7 | 7.6 |
| yolov8s | 11.17 | 84.7 | 11.2 |
| yolov8m | 25.9 | 158.6 | 14.2 |
| yolo11n | 2.62 | 42.8 | 2.9 |
| yolo11s | 9.46 | 77.1 | 14.6 |
| yolo11m | 20.11 | 158.3 | 14.1 |
| yolo26n | 2.60 | 81.1 | 6.6 |
| yolo26s | 9.29 | 150.0 | 18.7 |
| yolo26m | 20.2 | 261.1 | 25.4 |

Key finding: YOLO11 beats YOLOv8 at every size on both params and speed.
YOLO26 is slower on CPU — its architecture gains (attention-based) only
pay off with GPU acceleration.

### Parameter sweep (yolov8m)
| Varied | Value | Mean ms/img |
|--------|-------|-------------|
| imgsz | 320 | 50.1 |
| imgsz | 640 | 157.7 |
| imgsz | 1280 | 549.8 |
| conf | 0.1 | 160.1 |
| conf | 0.25 | 158.7 |
| conf | 0.5 | 160.2 |
| batch | 1 | 160.0 |
| batch | 4 | 202.9 |
| batch | 8 | 205.7 |

Key finding: `imgsz` has a massive effect (~11× from 320→1280). `conf` has
virtually no effect on inference time (NMS cost is negligible vs. backbone).
`batch > 1` is slower per image on CPU — batching only helps on GPU.

### Quantization
FP32 baseline only recorded so far (fp16_onnx export succeeded but ort session
run was not yet timed against MOT17; TensorRT requires Jetson CUDA).

---

## Known Issues / Pitfalls

- **MOT17 labels**: MOT17 annotations are in a custom tracking format, not YOLO
  label format. `mot17.yaml` is valid for inference and TRT INT8 calibration
  **only**. Any `yolo.train()` call must use a different dataset (e.g. `coco8.yaml`).
- **YOLO26 naming**: Verify available model names on the Jetson with
  `YOLO("yolo26n.pt")` and update `MODELS` if the name differs.
- **TensorRT on macOS**: All `trt_*` formats are silently skipped on macOS
  (`torch.cuda.is_available()` guard). Export and run TRT engines on the Jetson.
- **`.pt` files in repo root**: Model weights are downloaded by ultralytics on
  first use and land in the project root. They are gitignored but may need to be
  re-downloaded on a new machine or the Jetson.

---

## Jetson-specific Notes

- Change `DEVICE = "0"` to route inference to CUDA.
- TensorRT `.engine` files are device-specific — always export on the Jetson itself.
- INT8 calibration draws ~100–500 images from `config/mot17.yaml`.
- Energy mode experiments: `sudo nvpmodel -m <mode>` + `sudo jetson_clocks`,
  then re-run `run_benchmarks.py` and compare CSVs across power modes.
- Docker base image: `nvcr.io/nvidia/l4t-pytorch`.
