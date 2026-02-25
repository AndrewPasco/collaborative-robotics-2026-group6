"""
RGB-D segmentation to object point cloud using standard torchvision models.

- Loads RGB + depth from disk.
- Aligns depth to RGB using align_depth_fncs.py.
- Segments with DeepLabV3 (ResNet50) from torchvision.
- Builds an object-only point cloud for visualization/grasping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import models, transforms

from align_depth_fncs import align_depth


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_tuple(cls, values: Tuple[float, float, float, float]) -> "CameraIntrinsics":
        return cls(*values)


def load_intrinsics(path: Optional[str], fallback: CameraIntrinsics) -> CameraIntrinsics:
    if not path:
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
    )


def load_transform(path: Optional[str]) -> np.ndarray:
    if not path:
        return np.eye(4, dtype=np.float32)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = np.array(data, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix.")
    return matrix


def load_rgb_depth(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("Failed to read RGB image.")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to read depth image.")
    return rgb, depth


def load_mask(mask_path: Optional[str], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError("Failed to read mask image.")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def segment_with_standard_model(
    rgb: np.ndarray,
    class_id: Optional[int],
    device: Optional[str],
) -> Tuple[np.ndarray, int]:
    """
    Run DeepLabV3 segmentation and return a binary mask and selected class id.
    trained on COCO (21 classes).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.eval().to(device)

    # Preprocess
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Resize mask back to original image size
    mask_resized = cv2.resize(output_predictions, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    if class_id is None:
        # If no class specified, pick the most frequent class that is NOT background (0)
        flat = mask_resized.flatten()
        flat = flat[flat != 0] # Exclude background
        if flat.size == 0:
            # Only background found
            class_id = 0
        else:
            counts = np.bincount(flat)
            class_id = int(np.argmax(counts))
            print(f"Auto-selected dominant class ID: {class_id}")

    mask = (mask_resized == class_id).astype(np.uint8)
    return mask, class_id


def depth_to_point_cloud(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray,
) -> np.ndarray:
    """Create Nx3 point cloud from aligned depth in meters and a binary mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    h, w = depth_m.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[ys, xs]
    valid = z > 0
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]

    x = (xs - intrinsics.cx) * z / intrinsics.fx
    y = (ys - intrinsics.cy) * z / intrinsics.fy

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def save_ply(points: np.ndarray, path: str) -> None:
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def visualize_point_cloud(points: np.ndarray) -> None:
    if points.size == 0:
        print("No points to visualize.")
        return
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
    except Exception:
        print("Open3D not available. Skipping visualization.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB-D object segmentation to point cloud (Standard Model)")
    parser.add_argument("--rgb-path", required=True, help="Path to RGB image")
    parser.add_argument("--depth-path", required=True, help="Path to depth image")
    parser.add_argument("--mask-path", default=None, help="Optional binary mask path")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Scale to convert depth units to meters (e.g., 0.001 for mm)",
    )
    parser.add_argument("--rgb-intrinsics", default=None, help="JSON with fx,fy,cx,cy")
    parser.add_argument("--depth-intrinsics", default=None, help="JSON with fx,fy,cx,cy")
    parser.add_argument("--cam2cam", default=None, help="JSON 4x4 depth->rgb transform")
    parser.add_argument("--output-ply", default="object.ply", help="Output PLY path")
    
    # Arg compatibility with batch script
    parser.add_argument("--dinov3-repo", default=None, help="Ignored")
    parser.add_argument("--dinov3-weights", default=None, help="Ignored")
    parser.add_argument("--dinov3-backbone-weights", default=None, help="Ignored")
    
    parser.add_argument("--class-id", type=int, default=None, help="COCO class id to extract")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--visualize", action="store_true", help="Visualize point cloud")
    args, unknown = parser.parse_known_args()

    rgb, depth = load_rgb_depth(args.rgb_path, args.depth_path)

    # Sample fallback intrinsics (update for your camera or provide JSONs)
    rgb_intr = load_intrinsics(args.rgb_intrinsics, CameraIntrinsics(1297.67, 1298.63, 620.91, 238.28))
    depth_intr = load_intrinsics(args.depth_intrinsics, CameraIntrinsics(360.01, 360.01, 243.87, 137.92))
    cam2cam = load_transform(args.cam2cam)

    aligned_depth = align_depth(
        depth=depth,
        depth_K=(depth_intr.fx, depth_intr.fy, depth_intr.cx, depth_intr.cy),
        rgb=rgb,
        rgb_K=(rgb_intr.fx, rgb_intr.fy, rgb_intr.cx, rgb_intr.cy),
        cam2cam_transform=cam2cam,
    )

    depth_m = aligned_depth.astype(np.float32) * args.depth_scale

    mask = load_mask(args.mask_path, (rgb.shape[0], rgb.shape[1]))
    if mask is None:
        mask, class_id = segment_with_standard_model(
            rgb=rgb,
            class_id=args.class_id,
            device=args.device,
        )
        print(f"Standard model selected class id: {class_id}")

    points = depth_to_point_cloud(depth_m, rgb_intr, mask)
    save_ply(points, args.output_ply)
    print(f"Saved point cloud with {points.shape[0]} points to {args.output_ply}")

    if args.visualize:
        visualize_point_cloud(points)


if __name__ == "__main__":
    main()
