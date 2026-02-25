# TidyBot2 Vision Pipeline: Extraction and Segmentation

This directory contains the scripts and documentation for extracting images from ROS 2 bags and performing 2D/3D segmentation on them.

## 1. Image Extraction from Rosbag
Use `extract_images_from_bag.py` to convert ROS 2 `.db3` bag files into sets of synchronized RGB and Depth PNG images.

```bash
python3 scripts/extract_images_from_bag.py \
  --bag path/to/rosbag \
  --output path/to/output_dir \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw
```
*Creates `rgb/` and `depth/` subdirectories with timestamp-based filenames.*

## 2. Batch Segmentation (Full Pipeline)
Use `batch_segmentation.py` to process an entire folder of extracted images. It automatically matches RGB frames to the closest Depth frames and generates 3D point clouds.

```bash
uv run python scripts/batch_segmentation.py \
  --input-dir path/to/extracted_images \
  --output-dir path/to/segmented_pointclouds \
  --segmentation-script ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/segmentation_std.py
```

## 3. Segmentation Models

### Standard: DeepLabV3 (`segmentation_std.py`)
A robust fallback model using `torchvision`. Best for general scene understanding and verifying pipeline logic without heavy VRAM requirements.

### Precision: SAM 2 Tiny (`segment_sam2.py`)
The recommended choice for high-precision tasks (e.g., segmenting a bottle cap).
- **Auto-mode**: Uses YOLOv11 to find objects and SAM2 to refine the mask.
- **Manual-mode**: Provide specific pixel coordinates for surgical precision.

```bash
uv run python ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/segment_sam2.py \
  --rgb-path image.png \
  --auto
```

### Research: DINOv3 (`segmentation_v1.py`)
Advanced semantic segmentation (ADE20K classes). 
*Note: Requires ~28GB VRAM for the 7B model. Weights are gated by Meta.*

## 4. Troubleshooting & Best Practices
- **Hardware**: For laptop GPUs (RTX 3060), use **SAM 2 Tiny** or **DINOv2** instead of DINOv3.
- **Alignment**: Ensure intrinsics and extrinsics are correctly set in the scripts if point clouds look distorted.
- **Environments**: Always use `uv run` to ensure all model dependencies are correctly loaded.

---
*For detailed usage of the original DINOv3 script, see [segmentation_v1_README.md](./segmentation_v1_README.md).*
