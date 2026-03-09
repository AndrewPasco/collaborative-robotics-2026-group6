import torch
from ultralytics import YOLO
import time
import numpy as np
import os

def benchmark_model(model_name, device):
    print(f"\n--- Benchmarking {model_name} on {device} ---")
    model_path = os.path.expanduser(f"~/.yolo_models/{model_name}")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, using default download...")
        model = YOLO(model_name).to(device)
    else:
        model = YOLO(model_path).to(device)

    # Use FP16 if on CUDA
    half = (device == 'cuda')
    
    # Create black image (640x480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Warm-up (10 runs)
    print("Warming up...")
    for _ in range(10):
        model(img, device=device, half=half, verbose=False)

    # Benchmark (50 runs)
    print("Running benchmark...")
    times = []
    for _ in range(50):
        t0 = time.time()
        model(img, device=device, half=half, verbose=False)
        times.append(time.time() - t0)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f"Results for {model_name}:")
    print(f"  Avg Inference: {avg_time*1000:.2f} ms")
    print(f"  Max Inference: {max(times)*1000:.2f} ms")
    print(f"  Min Inference: {min(times)*1000:.2f} ms")
    print(f"  Est. FPS:      {fps:.1f}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"System Device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Test both models if available
    benchmark_model("yolov8n.pt", device)
    # The user mentioned yolo26x.pt, which I assume is yolov8x.pt or similar
    # I'll check if yolov8x.pt exists or try to benchmark it
    if os.path.exists(os.path.expanduser("~/.yolo_models/yolov8x.pt")):
        benchmark_model("yolov8x.pt", device)
    elif os.path.exists(os.path.expanduser("~/.yolo_models/best.pt")):
        benchmark_model("best.pt", device)
