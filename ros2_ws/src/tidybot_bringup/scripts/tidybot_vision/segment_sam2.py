#!/usr/bin/env python3
"""
Segment objects (bottles, caps) using SAM2 Tiny.
Allows for manual prompting (points/boxes) or automatic detection.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import SAM

def main():
    parser = argparse.ArgumentParser(description="Segment images using SAM2 Tiny")
    parser.add_argument("--rgb-path", required=True, help="Path to input RGB image")
    parser.add_argument("--output-dir", default="segmented_images", help="Directory to save output")
    parser.add_argument("--model", default="sam2_t.pt", help="SAM2 model version (e.g., sam2_t.pt)")
    
    # Prompting arguments
    parser.add_argument("--points", type=str, help="Comma-separated points 'x1,y1,x2,y2'")
    parser.add_argument("--labels", type=str, help="Comma-separated labels (1 for foreground, 0 for background)")
    parser.add_argument("--box", type=str, help="Bounding box as 'x1,y1,x2,y2'")
    parser.add_argument("--auto", action="store_true", help="Auto-detect bottle/cup using YOLO")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load SAM2 model
    print(f"Loading SAM2 model: {args.model}")
    model = SAM(args.model)

    # Prepare prompts
    predict_args = {}
    
    if args.auto:
        from ultralytics import YOLO
        print("Using YOLO to auto-detect bottle/cup...")
        yolo_model = YOLO("yolo11n.pt")
        yolo_results = yolo_model.predict(args.rgb_path, classes=[39, 41]) # bottle, cup
        if yolo_results[0].boxes:
            # Use the first detected box
            box = yolo_results[0].boxes.xyxy[0].cpu().numpy()
            predict_args["bboxes"] = [box.tolist()]
            print(f"Auto-detected box: {predict_args['bboxes']}")
        else:
            print("YOLO found no bottles or cups. Falling back to center point.")
            predict_args["points"] = [[320, 240]]
            predict_args["labels"] = [1]
    
    if args.points:
        pts = [float(p) for p in args.points.split(",")]
        # Reshape to [[x1, y1], [x2, y2], ...]
        predict_args["points"] = np.array(pts).reshape(-1, 2).tolist()
        
        if args.labels:
            predict_args["labels"] = [int(l) for l in args.labels.split(",")]
        else:
            # Default all to foreground
            predict_args["labels"] = [1] * len(predict_args["points"])
            
    if args.box and not predict_args.get("bboxes"):
        predict_args["bboxes"] = [float(b) for b in args.box.split(",")]

    # Run inference
    print(f"Running SAM2 inference on: {args.rgb_path} with args: {predict_args}")
    results = model.predict(args.rgb_path, **predict_args)
    
    # Process results
    img_name = os.path.basename(args.rgb_path)
    name_no_ext = os.path.splitext(img_name)[0]
    
    # Save the visualized result
    output_vis = os.path.join(args.output_dir, f"{name_no_ext}_sam2_vis.jpg")
    results[0].save(filename=output_vis)
    print(f"Saved visualization to {output_vis}")
    
    # Save the masks
    if results[0].masks:
        for i, mask in enumerate(results[0].masks.data):
            # Mask is a torch tensor
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            output_mask = os.path.join(args.output_dir, f"{name_no_ext}_mask_{i}.png")
            cv2.imwrite(output_mask, mask_np)
            print(f"Saved mask {i} to {output_mask}")
    else:
        print("No masks found in the image.")

if __name__ == "__main__":
    main()
