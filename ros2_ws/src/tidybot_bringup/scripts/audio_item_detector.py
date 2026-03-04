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
import os
import glob
import numpy as np
import google.generativeai as genai
from google.cloud import speech
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env if present (searches parent directories)
load_dotenv(find_dotenv())



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
    
    def __init__(self):
        """
        Initialize the item extractor.
        """
        # 1. Try Environment Variable first
        api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key not found in GEMINI_API_KEY env or file. Please check your .env file.")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Default model for initialization (will be overridden in extraction if it fails)
        model_name = os.environ.get('GEMINI_MODEL', 'models/gemini-2.0-flash-lite')
        self.gemini_model = genai.GenerativeModel(model_name)
        
        # Setup Google Cloud Credentials if not provided via env
        gcp_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not gcp_creds:
            # Look specifically for GOOGLE_CREDENTIALS.json in common locations
            try:
                # 1. Check user home directory
                home_path = os.path.expanduser('~/GOOGLE_CREDENTIALS.json')
                
                # 2. Check current working directory
                cwd_path = os.path.join(os.getcwd(), 'GOOGLE_CREDENTIALS.json')

                # 3. Check repo root (from env)
                repo_root = os.environ.get('TIDYBOT_REPO_ROOT')
                repo_path = os.path.join(repo_root, 'GOOGLE_CREDENTIALS.json') if repo_root else None

                # 4. Check parent of this script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                parent_path = os.path.join(script_dir, '..', '..', '..', '..', 'GOOGLE_CREDENTIALS.json')
                
                if os.path.exists(home_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = home_path
                    self.log(f"Using STT credentials from home: {home_path}")
                elif os.path.exists(cwd_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cwd_path
                    self.log(f"Using STT credentials from CWD: {cwd_path}")
                elif repo_path and os.path.exists(repo_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = repo_path
                    self.log(f"Using STT credentials from repo root: {repo_path}")
                elif os.path.exists(parent_path):
                    parent_path = os.path.abspath(parent_path)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = parent_path
                    self.log(f"Using STT credentials from parent dir: {parent_path}")
            except Exception:
                pass
        
        # Initialize Google Cloud Speech client
        try:
            self.speech_client = speech.SpeechClient()
            self.log('Audio Item Detector (STT + Gemini-Text) initialized')
        except Exception as e:
            self.log(f'WARNING: Failed to initialize Google Cloud SpeechClient: {e}')
            self.log('Please ensure a .json credential file is in the git directory or configured in .env.')
            self.speech_client = None
    
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
        if not getattr(self, 'speech_client', None):
            self.log("STT Error: SpeechClient not initialized (missing GOOGLE_APPLICATION_CREDENTIALS?).")
            return ""
        try:
            # Detect sample rate from WAV header
            with wave.open(wav_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                self.log(f"Detected sample rate: {sample_rate} Hz")

            with io.open(wav_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
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
           Falls back to last-word heuristic if Gemini is rate-limited.
        """
        transcript = ""
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
            
            # Use chain: 2.0-flash-lite -> 2.0-flash -> 3-flash-preview
            model_chain = [
                'models/gemini-2.0-flash-lite',
                'models/gemini-2.0-flash',
                'models/gemini-3-flash-preview'
            ]
            
            response = None
            for model_name in model_chain:
                self.log(f'Trying Gemini model: {model_name}')
                self.gemini_model = genai.GenerativeModel(model_name)
                
                # Exponential backoff retry for 429 rate limits on this specific model
                for attempt in range(3):
                    try:
                        response = self.gemini_model.generate_content(prompt)
                        break
                    except Exception as e:
                        if '429' in str(e) or 'Resource exhausted' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                            wait = (2 ** attempt) 
                            self.log(f'Gemini 429 rate limit (attempt {attempt+1}/3), retrying in {wait}s...')
                            time.sleep(wait)
                        else:
                            self.log(f'Error with model {model_name}: {e}')
                            break # Try next model in chain

                if response:
                    break

            if response is None:
                self.log('Gemini model chain: All models failed or rate limited.')
                return "ERROR", transcript

            if not response.candidates or not response.candidates[0].content.parts:
                return "ERROR", transcript
            
            item_name = response.candidates[0].content.parts[0].text.strip().lower()
            
            # Robust extraction: split and take clean last word if model rambles
            item_name = item_name.split()[-1].strip('".?,') 
            
            self.log(f'Extracted Item: "{item_name}"')
            return item_name, transcript

        except Exception as e:
            self.log(f'ERROR in Gemini extraction: {str(e)}')
            return "ERROR", transcript

    def extract_sequential_from_audio(self, wav_path: str):
        """
        Extracts two items (payload and destination) from audio.
        Returns: JSON string {"payload": "...", "destination": "..."} or "ERROR".
        """
        transcript = ""
        try:
            self.log(f'Processing audio (sequential): {wav_path}')
            transcript = self.transcribe_audio(wav_path).strip()
            self.log(f'STT Transcript: "{transcript}"')

            if not transcript or transcript.upper() == "SILENCE":
                return "ERROR", "SILENCE"

            prompt = f"""Extract the payload item and the destination container/location from this command.
Return ONLY a JSON object with keys "payload" and "destination", or return "ERROR" if not found.

Example 1: "put the apple in the basket" -> {{"payload": "apple", "destination": "basket"}}
Example 2: "grab the banana and place it in the bowl" -> {{"payload": "banana", "destination": "bowl"}}

Rules:
- Return ONLY the JSON object, nothing else.
- All values should be lowercase.
- If you can't find both, return "ERROR".

Command: "{transcript}"
Result:"""

            self.log('Extracting sequential items via Gemini-Text...')
            
            # Use same model chain logic as single extraction
            model_chain = [
                'models/gemini-2.0-flash-lite',
                'models/gemini-2.0-flash',
                'models/gemini-3-flash-preview'
            ]
            
            response = None
            for model_name in model_chain:
                self.gemini_model = genai.GenerativeModel(model_name)
                for attempt in range(3):
                    try:
                        response = self.gemini_model.generate_content(prompt)
                        break
                    except Exception as e:
                        if '429' in str(e) or 'Resource exhausted' in str(e):
                            time.sleep(2 ** attempt)
                        else: break
                if response: break

            if response is None or not response.candidates or not response.candidates[0].content.parts:
                return "ERROR", transcript
            
            json_str = response.candidates[0].content.parts[0].text.strip()
            # Basic cleanup if model adds markdown blocks
            if json_str.startswith('```json'):
                json_str = json_str.replace('```json', '').replace('```', '').strip()
            elif json_str.startswith('```'):
                json_str = json_str.replace('```', '').strip()
            
            # Verify it parses as JSON
            try:
                json.loads(json_str)
                self.log(f'Extracted Sequential: {json_str}')
                return json_str, transcript
            except json.JSONDecodeError:
                self.log(f'Gemini returned invalid JSON: {json_str}')
                return "ERROR", transcript

        except Exception as e:
            self.log(f'ERROR in sequential extraction: {str(e)}')
            return "ERROR", transcript


class ItemExtractorROS:
    """ROS2 version with microphone recording capability."""
    
    def __init__(self, rclpy_module=None, Node=None, AudioRecord=None):
        """
        Initialize ROS2-enabled item extractor.
        
        Args:
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
        self.base = ItemExtractorBase()
        
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
