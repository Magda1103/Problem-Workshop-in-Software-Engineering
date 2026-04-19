import os
import threading
from collections import deque, Counter
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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
        
        self.buffers = {}       # separate buffer for each person (person_id -> deque)
        self.frame_counts = {}  # different frame count for each person
        self.last_seen = {}     # GC: stores the last frame index a person was seen (person_id -> frame_index)
        self.yolo_model = YOLO('yolov8n.pt') # detection model to identify people in the video
        
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

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
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                track_id, window = self.queue.get(timeout=1)
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
            
            print(f"Person ID: {int(track_id)} | Action: {pred_label}") 

        if self.predictions:
            self.final = Counter(self.predictions).most_common(1)[0][0]
            print("=" * 45)
            print(f"Final prediction (most frequent): {self.idx_to_class[self.final]}")

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
        frame_index = 0 # GC: Counter to keep track of the current frame number

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_index += 1 # GC: Increment frame counter

            # 1. YOLO searches for people in the current frame and assigns track IDs
            results = self.yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # Check if any people were detected in the frame
            if results[0].boxes is None or results[0].boxes.id is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # 2. Extract the specific person from the frame using the bounding box coords
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                    
                processed_crop = preprocess_frame(person_crop)
                
                with self.lock:
                    if track_id not in self.buffers:
                        self.buffers[track_id] = deque(maxlen=self.frames_limit)
                        self.frame_counts[track_id] = 0
                        
                    self.buffers[track_id].append(processed_crop)
                    self.frame_counts[track_id] += 1
                    self.last_seen[track_id] = frame_index # GC: Update the last seen frame

                    # 4. If we have collected a batch for a specific person, send it for classification
                    if self.frame_counts[track_id] % self.frame_step == 0 and len(self.buffers[track_id]) == self.frames_limit:
                        window = np.array(list(self.buffers[track_id]), dtype=np.float32)
                        self.queue.put((track_id, window))
                        
            # GC: Clean up old tracks (persons not seen for more than 30 frames)
            with self.lock:
                stale_ids = [tid for tid, last_seen_frame in self.last_seen.items() if frame_index - last_seen_frame > 30]
                for tid in stale_ids:
                    del self.buffers[tid]
                    del self.frame_counts[tid]
                    del self.last_seen[tid]

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