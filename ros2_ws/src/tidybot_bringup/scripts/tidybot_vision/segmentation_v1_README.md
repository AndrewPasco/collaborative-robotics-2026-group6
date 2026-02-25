# RGB‑D DINOv3 Segmentation → Object Point Cloud

This script loads an RGB image and a depth image, aligns depth to RGB, runs DINOv3 semantic segmentation, masks the object, and builds an object‑only point cloud for visualization or grasping.

## What is an ADE20K class id?

DINOv3’s built‑in segmentor is trained on the ADE20K semantic segmentation dataset. Each pixel is assigned a class index in the range 0–149 (150 classes). That index is the “ADE20K class id.” The script uses that id to select the object mask.

- If you do not pass a class id, the script picks the largest predicted class in the image.
- If you know the desired class id, pass it with `--class-id`.

## Files

- Script: [ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/segmentation_v1.py](ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/segmentation_v1.py)
- Depth alignment helper: [ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/align_depth_fncs.py](ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/align_depth_fncs.py)

## Inputs

You need:

1) RGB image path
2) Depth image path (raw depth)
3) Intrinsics for RGB and depth cameras
4) Depth‑to‑RGB extrinsic transform
5) DINOv3 weights (segmentor + backbone)

### Intrinsics JSON format

Example:

{
  "fx": 1297.67,
  "fy": 1298.63,
  "cx": 620.91,
  "cy": 238.28
}

### Transform JSON format (depth → RGB)

Example:

[
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
]

## DINOv3 weights

The DINOv3 repo is already local at:

/home/adi_linux/autonomy_projects/shared_models/dinov3

You still need to provide actual weights for the segmentor and backbone. Use local paths or URLs supported by DINOv3. The script forwards these to torch.hub.load() as `weights` and `backbone_weights`.

### Text‑prompt (DINOTxt) weights

If you want text prompts, pass `--text-prompt` and the DINOTxt weights via `--dinov3-text-weights` (plus `--dinov3-backbone-weights`).

## Usage

Basic usage (ADE20K class id):

python segmentation_v1.py \
  --rgb-path /path/to/rgb.png \
  --depth-path /path/to/depth.png \
  --rgb-intrinsics /path/to/rgb_intrinsics.json \
  --depth-intrinsics /path/to/depth_intrinsics.json \
  --cam2cam /path/to/depth_to_rgb_transform.json \
  --dinov3-weights /path/to/segmentor_weights.pth \
  --dinov3-backbone-weights /path/to/backbone_weights.pth \
  --class-id 3 \
  --output-ply object.ply \
  --visualize

Text‑prompt usage (recommended):

python segmentation_v1.py \
  --rgb-path /path/to/rgb.png \
  --depth-path /path/to/depth.png \
  --rgb-intrinsics /path/to/rgb_intrinsics.json \
  --depth-intrinsics /path/to/depth_intrinsics.json \
  --cam2cam /path/to/depth_to_rgb_transform.json \
  --dinov3-text-weights /path/to/dinotxt_weights.pth \
  --dinov3-backbone-weights /path/to/backbone_weights.pth \
  --text-prompt "red bottle" \
  --output-ply object.ply \
  --visualize

Notes:

- If depth units are in millimeters, keep the default `--depth-scale 0.001`.
- If you have your own binary mask, you can skip DINOv3 with `--mask-path`.
- If you omit `--class-id`, the script selects the largest predicted class.

## ROS integration

The script includes a ROS2 subscriber stub (commented out). When you’re ready:

1) Uncomment the ROS block in [ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/segmentation_v1.py](ros2_ws/src/tidybot_bringup/scripts/tidybot_vision/segmentation_v1.py)
2) Replace the file‑load logic with the incoming ROS images
3) Reuse the same alignment, segmentation, and point‑cloud pipeline

## Output

- A PLY file is written to the location specified by `--output-ply`.
- Optional Open3D visualization with `--visualize`.

## Troubleshooting

- If the mask is empty, confirm the class id or try the largest‑class selection.
- If alignment looks wrong, verify intrinsics and the depth‑to‑RGB transform.
- If DINOv3 fails to load, verify weights paths and PyTorch installation.
