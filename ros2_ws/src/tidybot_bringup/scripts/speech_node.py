#!/usr/bin/env python3
"""
Speech Node

Implementation: Threaded Service Client Pattern
- Main thread: rclpy.spin() - Handles subscriptions/timers
- Worker thread: _do_listen() - Handles blocking service calls

This prevents deadlocks because the service client can wait for Future.result()
which is fulfilled by the main thread loop.
"""

import os
import threading
import time
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tidybot_msgs.srv import AudioRecord
from audio_item_detector import ItemExtractorBase


class SpeechNode(Node):
    def __init__(self):
        super().__init__('speech_node')

        # ── Pub/Sub ─────────────────────────────────────────────────────
        self.result_pub = self.create_publisher(String, '/brain/speech_result', 10)
        self.create_subscription(String, '/brain/speech_goal', self._goal_cb, 10)

        # ── Service Client ──────────────────────────────────────────────
        self.mic_client = self.create_client(AudioRecord, '/microphone/record')
        self.get_logger().info('Waiting for /microphone/record service …')
        if self.mic_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Microphone service connected.')
        else:
            self.get_logger().warn('Microphone service not available!')

        # ── State ───────────────────────────────────────────────────────
        self.extractor = ItemExtractorBase()
        self.wav_path = '/tmp/speech_node_recording.wav'
        self.busy_lock = threading.Lock()
        self.is_busy = False

        self.get_logger().info('Speech Node ready (Threaded)')

    def _goal_cb(self, msg: String):
        """Callback runs in Main Thread. Spawns Worker Thread."""
        goal = msg.data.strip()
        
        test_file = None
        if goal.lower() == 'listen':
            pass
        elif goal.startswith('test_audio '):
            test_file = goal.split(' ', 1)[1]
        else:
            return

        with self.busy_lock:
            if self.is_busy:
                self.get_logger().warn('Busy processing audio - ignoring request')
                return
            self.is_busy = True
            self.current_test_file = test_file

        # Run logic in a thread so we don't block the main spin loop
        t = threading.Thread(target=self._run_recording_sequence, daemon=True)
        t.start()

    def _run_recording_sequence(self):
        """
        Worker Thread Logic.
        Can enable blocking service calls because Main Thread is spinning.
        """
        try:
            self.get_logger().info('Starting recording sequence...')
            test_file = getattr(self, 'current_test_file', None)

            if test_file:
                self.get_logger().info(f'Using test audio file: {test_file}')
                if not os.path.exists(test_file):
                    self._fail(f'Make sure test file exists: {test_file}')
                    return
                time.sleep(1.0) # Simulate some delay
                self.wav_path = test_file
            else:
                # 1. FORCE STOP (Clear any stuck state)
                stop_req = AudioRecord.Request()
                stop_req.start = False
                future = self.mic_client.call_async(stop_req)
                try:
                    rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                except Exception:
                    pass

                time.sleep(0.5)

                # 2. START RECORDING
                start_req = AudioRecord.Request()
                start_req.start = True
                future = self.mic_client.call_async(start_req)
                
                if not self._wait_for_future(future, timeout=12.0):
                    self._fail('Start recording timeout')
                    return
                
                result = future.result()
                if not result.success:
                    self._fail(f'Start recording failed: {result.message}')
                    return

                self.get_logger().info(f'Recording started: {result.message}')

                # 3. RECORD DURATION
                time.sleep(5.0)

                # 4. STOP RECORDING
                stop_req = AudioRecord.Request()
                stop_req.start = False
                future = self.mic_client.call_async(stop_req)

                if not self._wait_for_future(future, timeout=5.0):
                    self._fail('Stop recording timeout')
                    return

                result = future.result()
                if not result.success:
                    self._fail(f'Stop recording failed: {result.message}')
                    return
                
                self.get_logger().info(f'Recorded {result.duration:.1f}s')

                # 5. PROCESS
                self.extractor.save_wav(self.wav_path, result.audio_data, result.sample_rate)
                self.get_logger().info(f'Audio saved to: {os.path.abspath(self.wav_path)}')
            
            item, transcript = self.extractor.extract_item_from_audio(self.wav_path)

            if transcript:
                self.get_logger().info(f'[TRANSCRIPT]: "{transcript}"')
            
            if not item or item == 'ERROR':
                self.get_logger().warn('[FAIL] No item detected')
                self._publish('ERROR')
            else:
                self.get_logger().info(f'[OK] Detected item: "{item}"')
                self._publish(item)

        except Exception as e:
            self._fail(f'Exception: {e}\n{traceback.format_exc()}')
        finally:
            with self.busy_lock:
                self.is_busy = False

    def _wait_for_future(self, future, timeout):
        """Helper to wait for future completion."""
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def _fail(self, msg):
        self.get_logger().error(msg)
        self._publish('ERROR')

    def _publish(self, txt):
        msg = String()
        msg.data = txt
        self.result_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpeechNode()
    try:
        # Main thread just spins to handle callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
