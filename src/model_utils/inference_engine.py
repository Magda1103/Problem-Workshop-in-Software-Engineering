import os
import threading
from collections import deque, Counter
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch

from src.model_utils.baseline_model import FRAME_STEP, FRAMES_COUNT, CLASS_COUNT, ActionRecognition
from src.model_utils.model_training import preprocess_frame, MODEL_OUTPUT, INPUT_FOLDER

BASE_DIR = Path(__file__).resolve().parent.parent.parent


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
        self.buffer = deque(maxlen=self.frames_limit)  # Automatically maintain maxlen elements (deletes older element)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.frame_count = 0
        self.predictions = []
        self.final = None
        self.queue = Queue()  # Thread-safe FIFO

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_loaded = ActionRecognition(num_classes=CLASS_COUNT, backbone_type='resnet18')
        model_loaded.load_state_dict(torch.load(MODEL_OUTPUT))
        model_loaded.to(self.device)
        model_loaded.eval()

        self.model = model_loaded
        self.path = path

        classes = sorted(os.listdir(INPUT_FOLDER))  # Reads class names from directory structure.
        class_to_idx = {cls: i for i, cls in enumerate(classes)}  # Maps class names to numeric labels.
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def run_inference(self):
        """
            Consumer thread:
                - Waits for inference signal
                - Copies current buffer (sliding window)
                - Runs model prediction
        """
        i = 0
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                window = self.queue.get(timeout=1)
            except:
                continue

            # Convert to PyTorch tensor: (B, T, H, W, C) -> (B, C, T, H, W)
            tensor = torch.tensor(np.expand_dims(window, axis=0))  # (1, T, H, W, C)
            tensor = tensor.permute(0, 4, 1, 2, 3).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                _, pred = torch.max(output, 1)

            self.predictions.append(pred.item())
            pred_label = self.idx_to_class[pred.item()]
            print(f"Predicted class ({i}): {pred_label}")
            i = i + 1

        self.final = Counter(self.predictions).most_common(1)[0][0]
        print("=" * 45)
        print(f"Final prediction: {self.idx_to_class[self.final]}")

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
                with self.lock:
                    window = np.array(list(self.buffer), dtype=np.float32)
                self.queue.put(window)

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
