#!/usr/bin/env python3
"""
Audio Item Detector Module

Extracts item names from audio using Gemini API.
Can be used standalone or with ROS2 for live microphone recording.

"""

import os
import time
import wave
import json
import io
import numpy as np
import google.generativeai as genai
from google.cloud import speech
from google.api_core.exceptions import ResourceExhausted

# Configure Google Cloud Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mnt/hgfs/CS339R/collaborative-robotics-485523-05dfbadc5d79.json"


def import_ros2():
    """Import ROS2 modules only when needed."""
    try:
        import rclpy
        from rclpy.node import Node
        from tidybot_msgs.srv import AudioRecord
        return rclpy, Node, AudioRecord, True
    except ImportError as e:
        return None, None, None, False


class ItemExtractorBase:
    """Base class for item extraction from audio using Gemini API."""
    
    def __init__(self, api_key_path=None):
        """
        Initialize the item extractor.
        
        Args:
            api_key_path: Path to file containing Gemini API key (default: /mnt/hgfs/CS339R/api.txt)
        """
        # Setup API key from file
        if api_key_path is None:
            api_key_path = "/mnt/hgfs/CS339R/api.txt"
        
        if not os.path.exists(api_key_path):
            raise FileNotFoundError(f'API key file not found: {api_key_path}')
        
        # Read API key from file
        with open(api_key_path, 'r') as f:
            api_key = f.readline().strip()
        
        if not api_key:
            raise ValueError("API key file is empty")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Initialize Google Cloud Speech client
        self.speech_client = speech.SpeechClient()
        print('Audio Item Detector (STT + Gemini-Text) initialized')
    
    def log(self, message):
        """Log a message."""
        print(message)
    
    def save_wav(self, filename: str, audio_data: list, sample_rate: int):
        """
        Save float32 audio data to a 16-bit WAV file.
        
        Args:
            filename: Output WAV file path
            audio_data: List of float32 audio samples
            sample_rate: Sample rate in Hz
        """
        audio = np.array(audio_data, dtype=np.float32)
        # Clamp and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        int16_data = (audio * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())
    
    def transcribe_audio(self, wav_path: str) -> str:
        """Transcribe WAV file via Google Cloud Speech-to-Text."""
        try:
            with io.open(wav_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )

            response = self.speech_client.recognize(config=config, audio=audio)

            if not response.results:
                return ""
            
            # Extract transcript from the first alternative of the first result
            transcript = response.results[0].alternatives[0].transcript
            return transcript
        except Exception as e:
            self.log(f"STT Error: {str(e)}")
            return ""

    def extract_item_from_audio(self, wav_path: str):
        """
        Two-stage pipeline:
        1. Transcribe WAV via STT.
        2. Extract item from transcript via Gemini-Text.
        """
        try:
            self.log(f'Processing audio: {wav_path}')
            
            # --- Stage 1: Transcription ---
            transcript = self.transcribe_audio(wav_path).strip()
            self.log(f'STT Transcript: "{transcript}"')
            
            if not transcript or transcript.upper() == "SILENCE":
                self.log('No speech detected (Silence).')
                return "ERROR", "SILENCE"
            
            # --- Stage 2: Item Extraction ---
            prompt = f"""Extract the item name mentioned in this sentence. 
Return ONLY the item name in lowercase, or "ERROR" if no valid item is found.

Rules:
- If a sentence refers to an object (e.g., "get me the banana"), return "banana".
- If no clear item is mentioned, return "ERROR".
- Return ONLY the word, nothing else.

Sentence: "{transcript}"
Result:"""

            self.log('Extracting item via Gemini-Text...')
            response = self.gemini_model.generate_content(prompt)
            
            if not response.candidates or not response.candidates[0].content.parts:
                return "ERROR", transcript
            
            item_name = response.candidates[0].content.parts[0].text.strip().lower()
            
            # Robust extraction: split and take clean last word if model rambles
            item_name = item_name.split()[-1].strip('".?,') 
            
            self.log(f'Extracted Item: "{item_name}"')
            return item_name, transcript

        except Exception as e:
            self.log(f'ERROR in extraction: {str(e)}')
            return "ERROR", ""


class ItemExtractorROS:
    """ROS2 version with microphone recording capability."""
    
    def __init__(self, api_key_path=None, rclpy_module=None, Node=None, AudioRecord=None):
        """
        Initialize ROS2-enabled item extractor.
        
        Args:
            api_key_path: Path to API key file
            rclpy_module: rclpy module (pass from import_ros2())
            Node: Node class (pass from import_ros2())
            AudioRecord: AudioRecord service (pass from import_ros2())
        """
        # Store ROS2 modules
        self.rclpy = rclpy_module
        self.Node = Node
        self.AudioRecord = AudioRecord
        
        # Initialize ROS2 node
        self.node = self.Node('item_extractor')
        
        # Setup base extractor functionality
        self.base = ItemExtractorBase(api_key_path)
        
        # Microphone service client
        self.mic_client = self.node.create_client(self.AudioRecord, '/microphone/record')
        self.node.get_logger().info('Audio Item Detector (ROS2) initialized')
    
    def log(self, message):
        """Log using ROS logger."""
        self.node.get_logger().info(message)
    
    def wait_for_microphone(self, timeout=5.0):
        """
        Wait for microphone service to be available.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if service is available, False otherwise
        """
        self.log('Waiting for /microphone/record service...')
        if not self.mic_client.wait_for_service(timeout_sec=timeout):
            self.log('Microphone service not available!')
            return False
        self.log('Microphone service connected.')
        return True
    
    def record_audio(self, duration: float):
        """
        Record audio using the microphone service.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            AudioRecord.Response with audio data
        """
        # Start recording
        start_req = self.AudioRecord.Request()
        start_req.start = True
        start_future = self.mic_client.call_async(start_req)
        self.rclpy.spin_until_future_complete(self.node, start_future, timeout_sec=5.0)
        
        if start_future.result() is None or not start_future.result().success:
            raise RuntimeError('Failed to start recording')
        
        self.log(f'Recording for {duration:.1f} seconds...')
        time.sleep(duration)
        
        # Stop recording
        stop_req = self.AudioRecord.Request()
        stop_req.start = False
        stop_future = self.mic_client.call_async(stop_req)
        self.rclpy.spin_until_future_complete(self.node, stop_future, timeout_sec=30.0)
        
        if stop_future.result() is None:
            raise RuntimeError('Failed to stop recording')
        
        response = stop_future.result()
        if not response.success:
            raise RuntimeError(f'Recording failed: {response.message}')
        
        self.log(
            f'Recorded {len(response.audio_data)} samples, '
            f'{response.duration:.2f}s @ {response.sample_rate} Hz'
        )
        
        return response
    
    def save_wav(self, filename: str, audio_data: list, sample_rate: int):
        """Save audio to WAV file."""
        return self.base.save_wav(filename, audio_data, sample_rate)
    
    def extract_item_from_audio(self, audio_file_path: str) -> str:
        """Extract item name from audio file."""
        return self.base.extract_item_from_audio(audio_file_path)
    
    def destroy_node(self):
        """Cleanup ROS2 node."""
        self.node.destroy_node()
