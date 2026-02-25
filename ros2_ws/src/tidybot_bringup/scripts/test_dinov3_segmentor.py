import sys
import os
import torch

repo_dir = "/home/adi_linux/autonomy_projects/shared_models/dinov3"
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

# Import the module to ensure our patch in segmentors.py is effective
try:
    from dinov3.hub.segmentors import dinov3_vitl16_ms
    print("Found dinov3_vitl16_ms in segmentors")
except ImportError:
    print("dinov3_vitl16_ms NOT found in segmentors")

try:
    # Try loading the model via hub (this triggers the _make function)
    print(f"Loading dinov3_vitl16_ms from {repo_dir}")
    # We use local source to use our hacked segmentors.py
    # We pass check_hash=False because we don't know the hash and want to see if it downloads anything
    # We default to ADE20K weights
    model = torch.hub.load(repo_dir, "dinov3_vitl16_ms", source="local", pretrained=True, check_hash=False)
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
