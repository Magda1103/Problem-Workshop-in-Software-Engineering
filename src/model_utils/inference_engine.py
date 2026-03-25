import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.model_utils.baseline_model import FRAME_STEP, FRAMES_COUNT
from src.model_utils.model_training import preprocess_frame

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / 'models' / 'best_model.keras'


class InferenceEngine:
    """
        Real-time inference engine using a sliding window approach.

        It uses:
        - Producer thread: reads video frames
        - Consumer thread: performs inference on buffered frames
    """

    def __init__(self, frame_step, frames_limit, path):
        self.frame_step = frame_step
        self.frames_limit = frames_limit
        self.buffer = deque(maxlen=self.frames_limit)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.inference_event = threading.Event()

        self.frame_count = 0
        self.predictions = []

        self.model = tf.keras.models.load_model(str(MODEL_PATH))
        self.path = path

    def run_inference(self):
        """
            Consumer thread:
                - Waits for inference signal
                - Copies current buffer (sliding window)
                - Runs model prediction
        """
        while not self.stop_event.is_set():
            triggered = self.inference_event.wait(timeout=1.0)
            if not triggered:
                continue

            self.inference_event.clear()

            with self.lock:
                window = np.array(list(self.buffer))

            tensor = tf.convert_to_tensor(np.expand_dims(window, axis=0))

            prediction = self.model.predict(tensor)
            predicted_class = np.argmax(prediction)
            self.predictions.append(predicted_class)
            print(f"Predicted clas: {predicted_class}")

    def perform_using_sliding_window(self):
        """
            Producer thread:
                - Reads frames from video
                - Preprocesses frames
                - Maintains sliding window buffer
                - Triggers inference when conditions are met
        """
        inference_thread = threading.Thread(target=self.run_inference)
        inference_thread.start()

        cap = cv2.VideoCapture(self.path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_frame(frame)

            with self.lock:
                self.buffer.append(frame)

            frame_index += 1

            if frame_index % self.frame_step == 0 and len(self.buffer) == self.frames_limit:
                self.inference_event.set()

        cap.release()
        self.stop_event.set()
        inference_thread.join()


if __name__ == "__main__":
    """
        Example usage:
        - Initialize inference engine
        - Run sliding window inference on video
    """

    VIDEO_PATH = BASE_DIR / 'data' / 'videos' / 'person_enters_car' / \
                 '0a4e6f8d-186c-4627-a436-4df638471e644701766132532539918_0.mp4'

    engine = InferenceEngine(frame_step=FRAME_STEP, frames_limit=FRAMES_COUNT,
                             path=VIDEO_PATH)
    engine.perform_using_sliding_window()
