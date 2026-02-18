"""Quick diagnostic: checks which devices work for YOLO inference."""
import time
import torch
from ultralytics import YOLO

MODEL = "yolov8n.pt"   # smallest model for a fast test
IMGSZ = 320            # small image size for speed

print("=" * 50)
print("  PyTorch device diagnostic")
print("=" * 50)
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  →  {torch.cuda.get_device_name(0)}", end="")
print()
print(f"  MPS      : {torch.backends.mps.is_available()}")
print()

candidates = []
if torch.cuda.is_available():
    candidates.append("0")
if torch.backends.mps.is_available():
    candidates.append("mps")
candidates.append("cpu")

print(f"  Loading {MODEL} ...")
model = YOLO(MODEL)

# Use a small synthetic image so no dataset is needed
import numpy as np
img = (np.random.rand(480, 640, 3) * 255).astype("uint8")

for device in candidates:
    print(f"\n  Testing device='{device}' ...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        result = model.predict(img, imgsz=IMGSZ, device=device, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        boxes = len(result[0].boxes) if result[0].boxes else 0
        print(f"OK  ({ms:.0f} ms,  {boxes} detections)")
    except Exception as e:
        print(f"FAIL  →  {e}")

print()
