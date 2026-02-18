# Ultralytics YOLO — Benchmarking on MOT17

This project benchmarks the Ultralytics YOLO framework for **person detection**
on the [MOT17](https://motchallenge.net/data/MOT17/) dataset. The code is
developed on macOS and is designed to transfer directly to an NVIDIA Jetson.

---

## Exercise Overview

### Task 1 — Native on the Jetson (or macOS for development)

1. **Baseline model comparison** — compare YOLOv8, YOLO11, YOLO26 (nano / small / medium)
   - Metrics: parameter count, mean ms/image, std, total time
   - First 10 frames excluded from timing (warm-up)

2. **Parameter sweep** — one-at-a-time experiment with three parameters:

   | Parameter | Values tested | Why it matters |
   |-----------|--------------|---------------|
   | `imgsz`   | 320, 640, 1280 | Resolution directly controls accuracy vs. speed |
   | `conf`    | 0.1, 0.25, 0.5 | Confidence threshold affects NMS load and detections |
   | `batch`   | 1, 4, 8        | Throughput vs. latency tradeoff on Jetson |

3. **Quantization** — test three precision formats:

   | Format       | macOS | Jetson | Notes |
   |-------------|-------|--------|-------|
   | FP32         | ✓    | ✓     | Default PyTorch inference |
   | FP16 (ONNX)  | ✓    | ✓     | ONNX export with `half=True` |
   | INT8 (TRT)   | ✗    | ✓     | TensorRT INT8 with calibration |
   | TRT FP16     | ✗    | ✓     | TensorRT FP16 engine |

4. **Pruning** — structured L1 channel pruning via `torch-pruning`:
   - Sparsity ratios: 0 %, 30 %, 50 %
   - Fine-tuning after pruning to recover accuracy

5. **Energy mode** (Jetson only):
   - Compare benchmark results across Jetson power modes
   - Use `sudo nvpmodel -m <mode>` and `sudo jetson_clocks`

### Task 2 — Docker

Wrap the scripts in a `Dockerfile` based on `nvcr.io/nvidia/l4t-pytorch`
and repeat Task 1 measurements inside the container.

---

## Models Used

| Version | Size variants | Model IDs |
|---------|--------------|-----------|
| YOLOv8  | nano / small / medium | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` |
| YOLO11  | nano / small / medium | `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` |
| YOLO26  | nano / small / medium | `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt` |

---

## Project Structure

```
ai-ws26/
├── requirements.txt          # Python dependencies
├── config/
│   └── mot17.yaml            # Dataset YAML for ultralytics
├── data/                     # MOT17 dataset (gitignored — see setup below)
│   └── MOT17/
│       └── train/
│           ├── MOT17-02-DPM/img1/*.jpg
│           ├── MOT17-04-DPM/img1/*.jpg
│           └── …
├── results/                  # Output CSVs (gitignored)
├── benchmark.py              # Baseline model comparison
├── benchmark_params.py       # Parameter sweep
├── benchmark_quantized.py    # Quantization experiments
└── pruning.py                # Structured pruning experiments
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the MOT17 dataset

1. Go to https://motchallenge.net/data/MOT17/ and download `MOT17.zip`
2. Extract it so the directory layout matches:
   ```
   data/MOT17/train/MOT17-02-DPM/img1/000001.jpg
   data/MOT17/train/MOT17-04-DPM/img1/000001.jpg
   …
   ```

### 3. (Optional) Create a flat images folder for ultralytics val

If you want to use `model.val()` with `config/mot17.yaml`, create a flat
images directory:

```bash
mkdir -p data/MOT17/train/images
find data/MOT17/train -name "*.jpg" -exec ln -s "$(pwd)/{}" data/MOT17/train/images/ \;
```

---

## Running the Benchmarks

All scripts accept `--device cpu` (default), `--device mps` (macOS Apple
Silicon), or `--device 0` (CUDA GPU index on Jetson).

### Baseline model comparison

```bash
python benchmark.py --data data/MOT17/train --limit 200 --device cpu
# Output: results/baseline.csv
```

### Parameter sweep

```bash
python benchmark_params.py --data data/MOT17/train --limit 200 --device cpu
# Output: results/params.csv
```

### Quantization

```bash
python benchmark_quantized.py --data data/MOT17/train --limit 200 --device cpu
# On Jetson:
python benchmark_quantized.py --data data/MOT17/train --limit 200 --device 0
# Output: results/quantized.csv
```

### Pruning

```bash
python pruning.py --data data/MOT17/train --finetune-epochs 10 --limit 200 --device cpu
# Output: results/pruning.csv
```

---

## Understanding the Results

Each script produces a CSV with these key columns:

| Column | Description |
|--------|-------------|
| `model` | Model identifier |
| `params_M` | Parameter count (millions) |
| `imgsz` | Image resolution |
| `conf` | Confidence threshold |
| `batch` | Batch size |
| `precision` | fp32 / fp16_onnx / trt_fp16 / trt_int8 |
| `pruning_ratio` | 0.0 / 0.3 / 0.5 |
| `mean_ms` | Mean inference time per image (ms) |
| `std_ms` | Standard deviation (ms) |
| `total_s` | Total benchmark time (s) |
| `n_images` | Images measured (after warm-up) |

Load and compare results with pandas:

```python
import pandas as pd
df = pd.read_csv("results/baseline.csv")
df.sort_values("mean_ms").to_string(index=False)
```

---

## Notes for Jetson Deployment

- Always pass `device=0` to route inference to the CUDA GPU
- TensorRT `.engine` files are device-specific — export them **on the Jetson**
- INT8 calibration uses ~100–500 images from MOT17 (handled automatically)
- Energy mode experiments:
  ```bash
  sudo nvpmodel -m 0   # max performance
  sudo nvpmodel -m 1   # 10 W mode (Jetson Nano) or 15 W (Xavier)
  sudo jetson_clocks   # lock clocks to max
  ```
- Repeat all benchmarks in each power mode and append results to CSVs
