import sys
import os
import torch

repo_dir = "/home/adi_linux/autonomy_projects/shared_models/dinov3"
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

print(f"Python executable: {sys.executable}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Try to import dinov3 module first
    import dinov3
    print("Imported dinov3 package")
    
    # Try loading model via hub
    print(f"Loading model from {repo_dir}")
    model = torch.hub.load(repo_dir, "dinov3_vit7b16_ms", source="local", pretrained=False)
    print("Model loaded successfully (pretrained=False)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
