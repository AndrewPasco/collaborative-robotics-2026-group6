import sys
print("Starting script...")
import torch
print(f"Torch loaded. CUDA: {torch.cuda.is_available()}")
from ultralytics import SAM
print("Ultralytics SAM imported.")
model = SAM("sam2_t.pt")
print("Model initialized.")
img_path = "ros2_ws/src/tidybot_bringup/rosbags/rosbag2_2026_02_03-21_13_35/extracted_images/rgb/1770182015805584311.png"
print(f"Running predict on {img_path}...")
results = model.predict(img_path, points=[[320, 240]], labels=[1])
print("Predict finished.")
print(f"Results: {results}")
