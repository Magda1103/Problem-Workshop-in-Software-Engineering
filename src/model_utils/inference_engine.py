import os
import random
import threading
import json
import argparse
from collections import deque, Counter
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from src.model_utils.baseline_model import FRAME_STEP, FRAMES_COUNT, ActionRecognition
from src.model_utils.fine_tuning import preprocess_frame

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class InferenceEngine:
    """
    Real-time inference engine with Temporal Smoothing and Memory Optimization.
    """

    def __init__(self, frame_step, frames_limit, video_path):
        self.frame_step = frame_step
        self.frames_limit = frames_limit
        self.path = video_path
        
        # Memory Management: tracking buffers
        self.buffers = {}       
        self.frame_counts = {}  
        self.last_seen = {}     
        self.latest_predictions = {}
        self.latest_confidences = {}
        self.action_history = {} 
        self.history_length = 5
        
        # Use GPU if available to prevent CPU bottlenecks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLO for person tracking
        self.yolo_model = YOLO('yolov8n.pt') 
        
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.queue = Queue(maxsize=10) # Limit queue size to prevent memory bloat

        # --- FIX: Updated for 4 classes instead of 9 ---
        MY_CLASS_COUNT = 4
        self.idx_to_class = {
            0: 'person_steals_object',
            1: 'person_enters_car',
            2: 'person_rides_bicycle',
            3: 'person_picks_up_object'
        }

        # Load your Fine-Tuned model
        model_path = BASE_DIR / 'models' / 'fine_tuned_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
            
        model_loaded = ActionRecognition(num_classes=MY_CLASS_COUNT, backbone_type='resnet18')
        model_loaded.load_state_dict(torch.load(model_path, map_location=self.device))
        model_loaded.to(self.device)
        model_loaded.eval()
        self.model = model_loaded

    def run_inference(self):
        """
        Consumer thread: Processes frames and applies Majority Vote.
        """
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                track_id, window = self.queue.get(timeout=1)
            except:
                continue

            # Efficient tensor conversion
            tensor = torch.tensor(np.expand_dims(window, axis=0), dtype=torch.float32)
            tensor = tensor.permute(0, 4, 1, 2, 3).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                
            # Memory safety: Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            pred_label = self.idx_to_class[pred.item()] 
            conf_percentage = round(confidence.item() * 100, 2)
            
            with self.lock:
                if track_id not in self.action_history:
                    self.action_history[track_id] = deque(maxlen=self.history_length)
                
                self.action_history[track_id].append(pred_label)
                
                # Majority Vote Logic
                most_common_action = Counter(self.action_history[track_id]).most_common(1)[0][0]
                self.latest_predictions[track_id] = most_common_action
                self.latest_confidences[track_id] = conf_percentage
            
            self.queue.task_done()

    def perform_inference(self):
        """
        Main loop: Video processing and Garbage Collection.
        """
        inference_thread = threading.Thread(target=self.run_inference)
        inference_thread.start()

        cap = cv2.VideoCapture(str(self.path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_index = 0 

        print(f"Processing: {self.path.name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_index += 1 

            # Track persons only
            results = self.yolo_model.track(frame, persist=True, verbose=False, classes=[0], conf=0.5)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size > 0:
                        processed_crop = preprocess_frame(frame) # Preprocess the entire frame for better context, can be changed to person_crop if needed.
                        
                        with self.lock:
                            if track_id not in self.buffers:
                                self.buffers[track_id] = deque(maxlen=self.frames_limit)
                                self.frame_counts[track_id] = 0
                                
                            self.buffers[track_id].append(processed_crop)
                            self.frame_counts[track_id] += 1
                            self.last_seen[track_id] = frame_index 

                            # Sliding window logic
                            if self.frame_counts[track_id] % self.frame_step == 0 and len(self.buffers[track_id]) == self.frames_limit:
                                window = np.array(list(self.buffers[track_id]), dtype=np.float32)
                                if not self.queue.full():
                                    self.queue.put((track_id, window))
                    
                    with self.lock:
                        current_action = self.latest_predictions.get(track_id, "Analyzing...")
                        conf = self.latest_confidences.get(track_id, 0)
                    
                    # Visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{int(track_id)} {current_action}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Garbage Collector: Prevent Memory Leaks by removing stale IDs (30 frames inactivity)
            with self.lock:
                stale_ids = [tid for tid, last_seen_frame in self.last_seen.items() if frame_index - last_seen_frame > 30]
                for tid in stale_ids:
                    del self.buffers[tid]
                    del self.frame_counts[tid]
                    del self.last_seen[tid]
                    self.latest_predictions.pop(tid, None)
                    self.action_history.pop(tid, None)
                    self.latest_confidences.pop(tid, None)

            cv2.imshow("Fine-Tuned Model", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stop_event.set()
        inference_thread.join()
        
        # --- MILESTONE: SAVE JSON LOG ---
        self.save_json_results()

    def save_json_results(self):
        """
        Saves detected actions to a JSON file as per Milestone requirements.
        """
        final_report = []
        for tid, action in self.latest_predictions.items():
            final_report.append({
                "video_source": str(self.path.name),
                "track_id": int(tid),
                "final_action": action,
                "confidence_score": f"{self.latest_confidences.get(tid, 0)}%"
            })

        log_path = BASE_DIR / 'src' / 'model_utils' /'inference_results.json'
        with open(log_path, 'w') as f:
            json.dump(final_report, f, indent=4)
        print(f"\nResults saved to {log_path}")

def get_random_video(base_dir):
    videos_dir = base_dir / 'data' / 'videos'
    allowed_folders = ['person_steals_object', 'person_enters_car', 'person_rides_bicycle', 'person_picks_up_object']
    all_videos = []

    for folder in allowed_folders:
        folder_path = videos_dir / folder
        if folder_path.exists():
            all_videos.extend(folder_path.glob('**/*.mp4'))
    if not all_videos:
        raise FileNotFoundError("No .mp4 files found!")
    return random.choice(all_videos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Recognition CLI Tool")
    parser.add_argument("--video", type=str, help="Path to specific video file")
    args = parser.parse_args()

    # Use specified video or pick a random one
    if args.video:
        video_to_process = Path(args.video)
    else:
        print("No video specified, picking a random one...")
        video_to_process = get_random_video(BASE_DIR)

    engine = InferenceEngine(frame_step=FRAME_STEP, frames_limit=FRAMES_COUNT, video_path=video_to_process)
    engine.perform_inference()