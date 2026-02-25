#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys

def find_closest_depth(rgb_timestamp, depth_files, tolerance=1e8):
    best_match = None
    min_diff = float('inf')
    
    for depth_file in depth_files:
        depth_name = os.path.basename(depth_file)
        try:
            # Handle both "timestamp.png" and "frame_timestamp.png"
            name_no_ext = os.path.splitext(depth_name)[0]
            if name_no_ext.startswith("frame_"):
                ts_str = name_no_ext.replace("frame_", "")
            else:
                ts_str = name_no_ext
                
            depth_ts = float(ts_str)
            diff = abs(rgb_timestamp - depth_ts)
            if diff < min_diff:
                min_diff = diff
                best_match = depth_file
        except ValueError:
            continue
            
    if min_diff <= tolerance:
        return best_match
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch process images with segmentation_v1.py")
    parser.add_argument("--input-dir", required=True, help="Root directory containing rgb and depth folders")
    parser.add_argument("--output-dir", required=True, help="Directory to save ply files")
    parser.add_argument("--segmentation-script", default="tidybot_vision/segmentation_v1.py", help="Path to segmentation_v1.py")
    # All other args are passed to the segmentation script
    args, unknown = parser.parse_known_args()

    rgb_dir = os.path.join(args.input_dir, "rgb")
    depth_dir = os.path.join(args.input_dir, "depth")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_files_list = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    print(f"Found {len(rgb_files)} RGB images and {len(depth_files_list)} Depth images.")
    
    processed_count = 0
    
    for rgb_file in rgb_files:
        rgb_name = os.path.basename(rgb_file)
        try:
            name_no_ext = os.path.splitext(rgb_name)[0]
            if name_no_ext.startswith("frame_"):
                ts_str = name_no_ext.replace("frame_", "")
            else:
                ts_str = name_no_ext
                
            rgb_ts = float(ts_str)
        except ValueError:
            print(f"Skipping {rgb_name}, cannot parse timestamp")
            continue
            
        depth_match = find_closest_depth(rgb_ts, depth_files_list)
        
        if depth_match:
            output_ply = os.path.join(args.output_dir, f"{ts_str}.ply")
            
            if os.path.exists(output_ply):
                print(f"Skipping {rgb_name}, output exists: {output_ply}")
                continue

            print(f"Processing {rgb_name} -> {output_ply}")
            
            # Construct command
            # Ensure we use the same python interpreter if possible, or assume uv run
            # Here we just use sys.executable assuming we are running in the correct env
            
            script_path = args.segmentation_script
            if not os.path.isabs(script_path):
                 # Assume relative to this script directory if not found in cwd
                 current_dir = os.path.dirname(os.path.abspath(__file__))
                 candidate = os.path.join(current_dir, script_path)
                 if os.path.exists(candidate):
                     script_path = candidate
            
            cmd = [
                sys.executable, script_path,
                "--rgb-path", rgb_file,
                "--depth-path", depth_match,
                "--output-ply", output_ply
            ] + unknown
            
            try:
                subprocess.check_call(cmd)
                processed_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error processing {rgb_name}: {e}")
        else:
            print(f"No matching depth for {rgb_name} within tolerance")
            
    print(f"Batch processing complete. Processed {processed_count} pairs.")

if __name__ == "__main__":
    main()
