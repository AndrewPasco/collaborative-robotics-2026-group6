#!/usr/bin/env python3
"""
Audio Item Detection Test Script

This script demonstrates how the audio item detector will be used in the real system.
It can test with pre-recorded WAV files or simulate live recording with ROS2.

Prerequisites:
    - Gemini API key at /mnt/hgfs/CS339R/api.txt
    - Microphone node running (for live recording):
        ros2 launch tidybot_bringup real.launch.py
      or standalone:
        ros2 run tidybot_control microphone_node

Usage:
    # Test with a pre-recorded WAV file
    python test_stt.py --file /mnt/hgfs/CS339R/Lab/apple.wav
    
    # Test with live microphone recording (requires ROS2 microphone service)
    python test_stt.py --duration 5.0

Author: TidyBot Team
Date: February 2026
"""

import argparse
import os
import sys
from audio_item_detector import ItemExtractorBase, ItemExtractorROS, import_ros2


def test_with_file(audio_file_path: str, api_key_path=None):
    """
    Test the detector with a pre-recorded audio file.
    Simulates how the real system will use the detector with audio input.
    
    Args:
        audio_file_path: Path to the WAV audio file
        api_key_path: Optional path to API key file
        
    Returns:
        Extracted item name
    """
    print(f'\n[TEST MODE: File-based Detection]')
    print(f'Audio file: {audio_file_path}')
    
    if not os.path.exists(audio_file_path):
        print(f'Error: File not found: {audio_file_path}')
        return None
    
    # Initialize the detector (simulating stage 1 of real system)
    print('\nInitializing Audio Item Detector...')
    detector = ItemExtractorBase(api_key_path=api_key_path)
    
    # Extract item from audio
    print('\nProcessing audio to extract item name...')
    item_name = detector.extract_item_from_audio(audio_file_path)
    
    # Print result (simulating what would be passed to stage 2)
    print('\n' + '='*60)
    print(f'EXTRACTED ITEM: "{item_name}"')
    print(f'(This would be passed to the navigation/manipulation stage)')
    print('='*60 + '\n')
    
    return item_name


def test_with_recording(duration: float, api_key_path=None):
    """
    Test the detector with live microphone recording.
    Simulates how the real system will use ROS2 to record and detect items.
    
    Args:
        duration: Recording duration in seconds
        api_key_path: Optional path to API key file
        
    Returns:
        Extracted item name
    """
    print(f'\n[TEST MODE: Live Recording Detection]')
    print(f'Recording duration: {duration} seconds')
    
    # Import ROS2 modules
    print('\nImporting ROS2 modules...')
    rclpy, Node, AudioRecord, success = import_ros2()
    
    if not success:
        print('\nError: ROS2 not available. Cannot record audio.')
        print('Make sure to source ROS2 environment:')
        print('  cd ros2_ws && source setup_env.bash')
        return None
    
    # Initialize ROS2
    print('Initializing ROS2...')
    rclpy.init()
    
    try:
        # Initialize the detector (simulating stage 1 of real system)
        print('Initializing Audio Item Detector with ROS2...')
        detector = ItemExtractorROS(
            api_key_path=api_key_path,
            rclpy_module=rclpy,
            Node=Node,
            AudioRecord=AudioRecord
        )
        
        # Wait for microphone service
        print('\nWaiting for microphone service...')
        if not detector.wait_for_microphone(timeout=5.0):
            print('Error: Microphone service not available.')
            print('Make sure to run: ros2 run tidybot_control microphone_node')
            return None
        
        # Record audio
        print(f'\nRecording audio for {duration} seconds...')
        print('Speak now: "I want the [item name]"')
        response = detector.record_audio(duration)
        
        # Save recording to temp file
        temp_dir = os.path.join(os.getcwd(), 'recordings')
        os.makedirs(temp_dir, exist_ok=True)
        wav_file = os.path.join(temp_dir, 'temp_recording.wav')
        
        detector.save_wav(wav_file, response.audio_data, response.sample_rate)
        print(f'Saved recording to: {wav_file}')
        
        # Extract item from audio
        print('\nProcessing audio to extract item name...')
        item_name = detector.extract_item_from_audio(wav_file)
        
        # Print result (simulating what would be passed to stage 2)
        print('\n' + '='*60)
        print(f'EXTRACTED ITEM: "{item_name}"')
        print(f'(This would be passed to the navigation/manipulation stage)')
        print('='*60 + '\n')
        
        return item_name
        
    except Exception as e:
        print(f'\nError during recording/detection: {e}')
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        detector.destroy_node()
        rclpy.shutdown()
        print('\nROS2 shutdown complete.')


def main():
    """
    Test script demonstrating how the audio item detector will be used in the real system.
    
    This simulates stage 1 of the robotics task system:
      Stage 1: Audio -> Item Name (this script)
      Stage 2: Item Name -> Navigation/Manipulation (not implemented here)
    """
    parser = argparse.ArgumentParser(
        description='Test Audio Item Detector - Simulates Stage 1 of Robotics Task System'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Test with existing WAV file instead of recording'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=5.0,
        help='Recording duration in seconds for live test (default: 5.0)'
    )
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        default=None,
        help='Path to API key file (default: /mnt/hgfs/CS339R/api.txt)'
    )
    args = parser.parse_args()
    
    print('='*60)
    print('AUDIO ITEM DETECTOR TEST')
    print('Simulating Stage 1 of Robotics Task System')
    print('='*60)
    
    try:
        if args.file:
            # Test with file
            result = test_with_file(args.file, api_key_path=args.api_key)
        else:
            # Test with live recording
            result = test_with_recording(args.duration, api_key_path=args.api_key)
        
        # Exit with appropriate code
        if result:
            print(f'Test completed successfully. Item detected: "{result}"')
            sys.exit(0)
        else:
            print('Test failed or no item detected.')
            sys.exit(1)
            
    except KeyboardInterrupt:
        print('\n\nTest interrupted by user.')
        sys.exit(1)
    except Exception as e:
        print(f'\nTest failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

