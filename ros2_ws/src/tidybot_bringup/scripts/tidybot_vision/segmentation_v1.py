"""
RGB-D segmentation to object point cloud.

- Loads RGB + depth from disk (ROS image subscription stub is commented out).
- Aligns depth to RGB using align_depth_fncs.py.
- Segments with DINOv3 (placeholder) or loads a mask from disk.
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


def segment_with_dinov3(
	rgb: np.ndarray,
	repo_dir: str,
	weights: Optional[str],
	backbone_weights: Optional[str],
	class_id: Optional[int],
	device: Optional[str],
) -> Tuple[np.ndarray, int]:
	"""
	Run DINOv3 segmentation and return a binary mask and selected class id.

	Uses the Mask2Former (m2f) ADE20K segmentor provided by the DINOv3 repo.
	If class_id is None, the largest predicted class is selected.
	"""
	import torch

	if repo_dir not in sys.path:
		sys.path.insert(0, repo_dir)

	from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
	from dinov3.eval.segmentation.inference import make_inference

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model = torch.hub.load(
		repo_dir,
		"dinov3_vit7b16_ms",
		source="local",
		weights=weights,
		backbone_weights=backbone_weights,
	)
	model.eval().to(device)

	img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
	mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
	std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
	img_t = (img_t - mean) / std
	img_t = img_t.to(device)

	pred = make_inference(
		img_t,
		model,
		inference_mode="whole",
		decoder_head_type="m2f",
		rescale_to=(rgb.shape[0], rgb.shape[1]),
		n_output_channels=150,
	)
	pred = pred.squeeze(0)
	cls_map = torch.argmax(pred, dim=0).cpu().numpy().astype(np.int32)

	if class_id is None:
		flat = cls_map.reshape(-1)
		counts = np.bincount(flat, minlength=150)
		class_id = int(np.argmax(counts))

	mask = (cls_map == class_id).astype(np.uint8)
	return mask, class_id


def segment_with_dinov3_text(
	rgb: np.ndarray,
	repo_dir: str,
	weights: Optional[str],
	backbone_weights: Optional[str],
	text_prompt: str,
	device: Optional[str],
	image_size: int,
	threshold: float,
	topk_percent: Optional[float],
) -> np.ndarray:
	"""
	Run DINOv3 DINOTxt to get a text-prompted mask.
	Returns a binary mask (H, W).
	"""
	import torch
	import torch.nn.functional as F

	if repo_dir not in sys.path:
		sys.path.insert(0, repo_dir)

	from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model, tokenizer = torch.hub.load(
		repo_dir,
		"dinov3_vitl16_dinotxt_tet1280d20h24l",
		source="local",
		weights=weights,
		backbone_weights=backbone_weights,
	)
	model.eval().to(device)

	img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
	img = img.astype(np.float32) / 255.0
	img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
	mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
	std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
	img_t = (img_t - mean) / std
	img_t = img_t.to(device)

	tokens = tokenizer.tokenize(text_prompt).to(device)
	with torch.inference_mode():
		_, text_features, _, patch_tokens, _ = model(img_t, tokens)

	patch_tokens = F.normalize(patch_tokens, dim=-1)
	text_features = F.normalize(text_features, dim=-1)
	sim = torch.matmul(patch_tokens, text_features.T).squeeze(-1).squeeze(0)
	patch_h = img_t.shape[2] // getattr(model.visual_model.backbone, "patch_size", 16)
	patch_w = img_t.shape[3] // getattr(model.visual_model.backbone, "patch_size", 16)
	if patch_h * patch_w != sim.shape[1]:
		patch_h = int(np.sqrt(sim.shape[1]))
		patch_w = sim.shape[1] // patch_h
	sim = sim.view(patch_h, patch_w).cpu().numpy()
	sim = cv2.resize(sim, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
	sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)

	if topk_percent is not None:
		thresh = np.percentile(sim, 100.0 - topk_percent)
		mask = sim >= thresh
	else:
		mask = sim >= threshold
	return mask.astype(np.uint8)


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
	parser = argparse.ArgumentParser(description="RGB-D object segmentation to point cloud")
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
	parser.add_argument(
		"--dinov3-repo",
		default="/home/adi_linux/autonomy_projects/shared_models/dinov3",
		help="Path to the local DINOv3 repo",
	)
	parser.add_argument("--dinov3-weights", default=None, help="Path or URL to DINOv3 segmentor weights")
	parser.add_argument(
		"--dinov3-backbone-weights",
		default=None,
		help="Path or URL to DINOv3 backbone weights",
	)
	parser.add_argument("--class-id", type=int, default=None, help="ADE20K class id to extract")
	parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
	parser.add_argument("--visualize", action="store_true", help="Visualize point cloud")
	args = parser.parse_args()

	parser.add_argument("--text-prompt", default=None, help="Text prompt for object selection")
	parser.add_argument("--dinov3-text-weights", default=None, help="Path or URL to DINOTxt weights")
	parser.add_argument("--text-image-size", type=int, default=224, help="Resize size for DINOTxt")
	parser.add_argument("--text-threshold", type=float, default=0.5, help="Mask threshold in [0,1]")
	parser.add_argument("--text-topk", type=float, default=None, help="Keep top-k percent similarity (0-100)")
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
		mask, class_id = segment_with_dinov3(
			rgb=rgb,
			repo_dir=args.dinov3_repo,
			weights=args.dinov3_weights,
			backbone_weights=args.dinov3_backbone_weights,
			class_id=args.class_id,
			device=args.device,
		)
		print(f"DINOv3 selected class id: {class_id}")

	points = depth_to_point_cloud(depth_m, rgb_intr, mask)
	save_ply(points, args.output_ply)
	print(f"Saved point cloud with {points.shape[0]} points to {args.output_ply}")

	if args.visualize:
		visualize_point_cloud(points)


if __name__ == "__main__":
	# ROS2 image subscription stub (commented out for now)
	#
	# import rclpy
	# from rclpy.node import Node
	# from sensor_msgs.msg import Image
	# from cv_bridge import CvBridge
	#
	# class RGBDSubscriber(Node):
	#     def __init__(self):
	#         super().__init__("rgbd_subscriber")
	#         self.bridge = CvBridge()
	#         self.rgb = None
	#         self.depth = None
	#         self.rgb_sub = self.create_subscription(Image, "/camera/color/image_raw", self.rgb_cb, 10)
	#         self.depth_sub = self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 10)
	#
	#     def rgb_cb(self, msg: Image) -> None:
	#         self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
	#
	#     def depth_cb(self, msg: Image) -> None:
	#         self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
	#
	# rclpy.init()
	# node = RGBDSubscriber()
	# rclpy.spin(node)
	# rclpy.shutdown()

	main()
