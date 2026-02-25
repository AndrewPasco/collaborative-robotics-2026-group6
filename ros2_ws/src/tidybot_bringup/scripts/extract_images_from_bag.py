#!/usr/bin/env python3

import argparse
import os
import cv2
import csv
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Image

def get_topics(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {topic_metadata.name: topic_metadata.type for topic_metadata in topic_types}
    return topic_type_map

def image_msg_to_cv2(msg):
    """
    Manually convert ROS Image message to OpenCV image (numpy array).
    Handles bgr8, rgb8, and 16UC1 (depth) encodings.
    """
    dtype = np.uint8
    n_channels = 1
    
    if msg.encoding == 'bgr8':
        dtype = np.uint8
        n_channels = 3
    elif msg.encoding == 'rgb8':
        dtype = np.uint8
        n_channels = 3
    elif msg.encoding == '16UC1':
        dtype = np.uint16
        n_channels = 1
    elif msg.encoding == 'mono8':
        dtype = np.uint8
        n_channels = 1
    else:
        print(f"Warning: Unsupported encoding {msg.encoding}. blindly trying to reshape based on step.")
        # heuristic fallback
        if msg.step == msg.width * 3:
            dtype = np.uint8
            n_channels = 3
        elif msg.step == msg.width * 2:
            dtype = np.uint16
            n_channels = 1
        elif msg.step == msg.width:
            dtype = np.uint8
            n_channels = 1
            
    try:
        # np.frombuffer is zero-copy if possible, but msg.data might be a list or bytes
        # In ROS2 python msg.data is usually array.array or bytes
        img_data = np.frombuffer(msg.data, dtype=dtype)
        
        # Reshape
        # Handle cases where step > width * channels * itemsize (padding)
        # But usually for cameras it's packed.
        # If padded, we need to extract row by row, which is slower.
        # For now assume packed or simple reshape
        
        expected_size = msg.height * msg.width * n_channels
        if img_data.size != expected_size:
             # This happens if there is padding (step > width * bpp)
             # We need to use stride
             # msg.step is bytes per row
             bpp = np.dtype(dtype).itemsize * n_channels
             if msg.step != msg.width * bpp:
                 # Padded
                 # We have to reshape to (height, step/itemsize) then crop
                 row_size_in_items = msg.step // np.dtype(dtype).itemsize
                 img_full = img_data.reshape((msg.height, row_size_in_items))
                 img = img_full[:, :msg.width*n_channels]
                 img = img.reshape((msg.height, msg.width, n_channels))
             else:
                 print(f"Error: Data size {img_data.size} does not match expected {expected_size} for {msg.height}x{msg.width}x{n_channels}")
                 return None
        else:
             img = img_data.reshape((msg.height, msg.width, n_channels))

        # Color conversion if needed
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def extract_images(bag_path, output_dir, rgb_topic='/camera/color/image_raw', depth_topic='/camera/depth/image_raw'):
    print(f"Extracting images from {bag_path} to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)

    if os.path.isfile(bag_path):
        # rosbag2_py expects the directory if it's a directory-based format, 
        # or the file path if it's a single file. 
        # .db3 implies sqlite3 which usually works with the file path + storage options.
        # But typically SequentialReader takes the URI.
        # If it fails, try dirname.
        pass

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return

    topics = [rgb_topic, depth_topic]
    storage_filter = StorageFilter(topics=topics)
    reader.set_filter(storage_filter)
    
    count_rgb = 0
    count_depth = 0
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        try:
            msg = deserialize_message(data, Image)
            cv_image = None
            
            # Using timestamp for filename to match rgb and depth later
            timestamp = t

            if topic == rgb_topic:
                cv_image = image_msg_to_cv2(msg)
                if cv_image is not None:
                    filename = os.path.join(rgb_dir, f"{timestamp}.png")
                    cv2.imwrite(filename, cv_image)
                    count_rgb += 1
            elif topic == depth_topic:
                cv_image = image_msg_to_cv2(msg)
                if cv_image is not None:
                    filename = os.path.join(depth_dir, f"{timestamp}.png")
                    # Save depth as is (16-bit png)
                    cv2.imwrite(filename, cv_image)
                    count_depth += 1
                
        except Exception as e:
            print(f"Error processing message on topic {topic} at time {t}: {e}")

    print(f"Extracted {count_rgb} RGB images.")
    print(f"Extracted {count_depth} Depth images.")

def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS 2 bag")
    parser.add_argument("--bag", required=True, help="Path to bag file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--rgb_topic", default="/camera/color/image_raw", help="RGB topic name")
    parser.add_argument("--depth_topic", default="/camera/depth/image_raw", help="Depth topic name")
    
    args = parser.parse_args()
    extract_images(args.bag, args.output, args.rgb_topic, args.depth_topic)

if __name__ == "__main__":
    main()
