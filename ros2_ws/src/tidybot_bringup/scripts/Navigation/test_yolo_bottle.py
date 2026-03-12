#!/usr/bin/env python3
"""
Usage:
uv run python src/tidybot_bringup/scripts/Navigation/test_yolo_bottle.py

Description:
    Loads the YOLO model (yolo26x.pt) and runs it on the bottle.png image.
    Outputs the detection results and saves an annotated image.
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
YOLO_MODEL_NAME = "yolo26x.pt"
IMAGE_PATH = os.path.expanduser("~/collaborative-robotics-2026-group6/examples/bottle.png")
MODEL_DIR = os.path.expanduser("~/.yolo_models")
RESULT_PATH = os.path.expanduser("~/collaborative-robotics-2026-group6/examples/yolo_bottle_result.jpg")

def main():
    # 1. Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, YOLO_MODEL_NAME)

    # 2. Check for model (Download if missing - similar to vision_yolo_gemini.py)
    if not os.path.exists(model_path):
        print(f"Model {YOLO_MODEL_NAME} not found. Attempting to download...")
        try:
            import urllib.request
            # Note: This URL follows the pattern in vision_yolo_gemini.py
            # If yolo26x.pt is a custom Stanford model, this might fail unless hosted there.
            url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{YOLO_MODEL_NAME}"
            urllib.request.urlretrieve(url, model_path)
            print(f"Download complete: {model_path}")
        except Exception as e:
            print(f"Failed to download model from GitHub: {e}")
            print("Falling back to locating any local .pt files...")
            # If the specific download fails, the user might need to provide the file.
            pass

    # 3. Load Model
    print(f"Loading YOLO model from {model_path}...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found at {IMAGE_PATH}")
        return
    
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Failed to load image at {IMAGE_PATH}")
        return

    # 5. Run Inference
    print("Running inference...")
    results = model.predict(img, conf=0.25)

    # 6. Process and Draw Results
    found_any = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            found_any = True
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(f"Found {cls_name} with confidence {conf:.2f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

            # Draw bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not found_any:
        print("No objects detected in the image.")
    else:
        # 7. Save result
        cv2.imwrite(RESULT_PATH, img)
        print(f"Annotated result saved to: {RESULT_PATH}")

if __name__ == "__main__":
    main()
