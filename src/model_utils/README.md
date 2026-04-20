# Sprint 2: AI Prototyping & Temporal Modeling (Weeks 4–7)

## Task 1 - Implement a Baseline Model

This model uses a pre-trained ResNet backbone for feature extraction. The original classification head is replaced with four (2+1)D CNN blocks. ResNet weights are frozen, so only the additional layers are trained.

### Architecture Overview:

1. **ResNet Backbone**
 - Processes individual frames (B*T, C, H, W).
- Outputs per-frame feature maps, later reshaped back to (B, C, T, H, W).
2. **Initial 3D Convolution**
- Conv3d with kernel (3,7,7) extracts preliminary spatiotemporal features.
- Followed by BatchNorm3d, ReLU, and MaxPool3d(1,2,2) for spatial downsampling.
3. **(2+1)D Convolution Blocks**
- Each block decomposes 3D convolution into:
  - Spatial (1, H, W) – captures frame-level features.
  - Temporal (T, 1, 1) – captures motion across frames.
- Residual connections add the input to the output of each block.
- Four blocks with filter sizes: 16 → 32 → 64 → 128.
- Spatial downsampling occurs after the second block.
4. **Global Pooling & Classification**
- AdaptiveAvgPool3d(1) averages across temporal and spatial dimensions.
- Fully connected layer maps the 128-channel feature vector to num_classes.

### Key Features

-  (2+1)D convolution reduces computation while preserving temporal modeling.
- Residual connections allow stable training of deep feature blocks.
- Modular backbone allows swapping between ResNet18, ResNet34, or ResNet50.
- Only additional layers are trainable; backbone remains frozen.

## Task 2 - Create the InferenceEngine module.

This module implements **real-time inference** using a **sliding window approach** with a producer-consumer pattern.

### Sliding Window
- Maintains a fixed-size buffer of the most recent frames (`frames_limit`, e.g., 16).  
- FIFO (First-In-First-Out) behavior: new frames push out the oldest ones.  
- `frame_step` controls prediction frequency:
  - `frame_step = 1`: predict on every frame (high computational cost).  
  - `frame_step > 1`: predict every N frames, smoothing predictions and saving resources.

#### Important note
- It is crucial to use the same number of frames per clip (`frames_limit`) for both training and prediction.
- However, a smaller `frame_step` can be used during prediction for smoother, more frequent predictions - as long as the total number of frames per clip remains consistent with training.

### Video Clip Extraction for Training

1. Each video is read using OpenCV (cv2.VideoCapture) and frames are preprocessed (resized, normalized, RGB conversion).
2. Clips consist of `frames_limit` consecutive frames. Clips are overlapping: the window moves by `frame_step` frames.
Example for `frames_limit=16` and `frame_step=5`: `[f0..f15], [f5..f20], [f10..f25], ...`
3. Multiple clips are extracted from each video. One clip is randomly selected per training iteration to improve generalization. If a video is too short to produce a valid clip, another random video is selected.
4. Clips are converted from (T, H, W, C) to (C, T, H, W) for PyTorch. Converted to torch.float32 tensors along with the corresponding label.


### Multi-threading (Producer-Consumer)
- **Producer thread**: captures frames from video and appends to the buffer.  
- **Consumer thread**: takes the latest window from the buffer and runs the model for prediction.  
- Thread safety is ensured with `self.lock` and a thread-safe queue.

### Tensor Preparation
- Frames are preprocessed:
  - Resized to model input dimensions (e.g., 224x224).  
  - Converted from BGR to RGB.  
  - Scaled to `[0,1]`.  
- Converted to 5D tensor `(Batch, Channels, Frames, Height, Width)` for the model.


----------------------------------------------------------------------------

## Task 3 - Fine-Tuning and Data Pipeline Optimization

This task adapts the baseline model to focus on high-priority security events and optimizes the data loading process to prevent memory exhaustion during training.

### Fine-Tuning Strategy
- **Class Reduction:** The model was fine-tuned specifically on 4 target classes (`person_steals_object`, `person_enters_car`, `person_rides_bicycle`, `person_picks_up_object`) to improve accuracy for critical security events.
- **Performance Tracking:** Implemented automated logging of training and validation metrics (`train_loss`, `train_acc`, `val_acc`) to a JSON file per epoch to track model convergence and prevent overfitting.

### Memory Optimization (RAM Bottleneck Fix)
- Addressed severe `ArrayMemoryError` crashes caused by loading entire videos into memory during the clip extraction phase.
- **Single Clip Extraction:** Modified the `VideoDataset` to extract only *one* random `frames_limit` clip directly from the video stream using OpenCV, discarding the remaining frames immediately.
- **Dataset Sampling:** Implemented dynamic subset sampling to train on a manageable, randomized portion of the dataset per epoch. This drastically reduces the memory footprint while maintaining data variance.

## Task 4 - Inference Engine Enhancements & Milestone CLI

This phase upgrades the `InferenceEngine` to support multi-object tracking, stabilizes predictions, and fulfills the Sprint milestone by generating a structured data contract.

### Multi-Object Tracking & Memory Management
- **YOLOv8 Integration:** Integrated a YOLOv8 nano model (`yolov8n.pt`) to detect and track individuals dynamically. The inference engine now tracks specific IDs rather than processing the entire global frame blindly.
- **Garbage Collection:** Implemented a cleanup routine to prevent infinite memory leaks during long video streams. The system monitors the `last_seen` frame for each `track_id` and automatically purges buffers, frame counters, and history queues for any ID inactive for more than 30 frames.

### Temporal Smoothing & Confidence Scoring
- **Majority Vote Buffer:** Created an `action_history` queue that stores the last 5 predictions for each tracked individual. The final output is determined by the most common prediction in this buffer, effectively eliminating UI flickering caused by single-frame misclassifications.
- **Softmax Probabilities:** Converted raw model logits into human-readable percentage confidence scores using `torch.nn.functional.softmax` to provide accurate uncertainty metrics to the downstream API.

### Milestone Delivery
- **JSON Data Logging:** The CLI tool now automatically exports a structured `inference_results.json` log upon completion. This file serves as the data contract for downstream GUI/API development, containing `video_source`, `track_id`, `final_action`, and `confidence_score`.

#### Important note on Data Pipeline
The baseline model was trained on **global context** (full frames with backgrounds) and the Inference Engine currently processes full frames as well

### Automated Environment Setup

To maintain developer experience and prevent Git repository bloat from large model weights and video datasets, an automated setup protocol was established.
- **`setup_data.py`:** A custom Python script utilizing `gdown` to automatically fetch the `.pth` model weights and sample video structures directly from cloud storage, extracting them seamlessly into the correct local directories.