import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from src.model_utils.baseline_model import create_model, WIDTH, HEIGHT, EPOCHS, BATCH_SIZE, FRAME_STEP, FRAMES_COUNT

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FOLDER = BASE_DIR / 'data' / 'videos'
MODEL_OUTPUT = BASE_DIR / 'models' / 'best_model.pth'
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)


def preprocess_frame(frame):
    """
        Preprocess the frame before feeding it to the model:

        Steps:
            - Resize frame to (WIDTH, HEIGHT)
            - Convert color from BGR (OpenCV default) to RGB
            - Normalize pixel values to range [0, 1]
            - Convert to float32
    """
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format from BGR to RGB.
    frame = frame.astype(np.float32) / 255.0
    return frame


def split_dataset(dataset, val_ratio=0.2):
    """
        Split dataset into training and validation sets.

        The dataset is randomly shuffled and then divided according
        to the given validation ratio.
    """
    all_labels = [label for _, label in dataset.samples]
    all_labels = np.array(all_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    return train_subset, val_subset


def create_dataloaders(all_data_dir, batch_size, val_ratio=0.2):
    """
        Build datasets and dataloaders for training and validation.
    """
    classes = sorted(os.listdir(all_data_dir))  # Reads class names from directory structure.
    class_to_idx = {cls: i for i, cls in enumerate(classes)}  # Maps class names to numeric labels.

    # Custom dataset that loads and processes video clips.
    full_dataset = VideoDataset(all_data_dir, FRAMES_COUNT, FRAME_STEP, class_to_idx)
    train_dataset, val_dataset = split_dataset(full_dataset, val_ratio=val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)  # Keep validation deterministic (shuffle=False).

    return train_loader, val_loader, class_to_idx


class VideoDataset(Dataset):
    """
        Converts raw videos into tensors for the model.
    """

    def __init__(self, dataset_dir, frames_limit, frame_step, class_to_idx):
        self.samples = []
        self.frames_limit = frames_limit
        self.frame_step = frame_step
        self.class_to_idx = class_to_idx

        for cls in os.listdir(dataset_dir):
            cls_dir = os.path.join(dataset_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((  # Each sample = video path + class index.
                        os.path.join(cls_dir, fname),
                        class_to_idx[cls]
                    ))

    def extract_clips(self, video_path):
        """
            Extract fixed-length clips from a video file.

            The video is read frame by frame, preprocessed, and then split into
            multiple overlapping clips using a sliding window approach.

            Each clip consists of a fixed number of consecutive frames
            (`self.frames_limit`), sampled every `self.frame_step`.
        """
        cap = cv2.VideoCapture(video_path)  # Open video file.
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()  # If ret=False -> the end of the video.
            if not ret:
                break
            frames.append(preprocess_frame(frame))  # List of all frames (T, H, W, C)

        cap.release()

        clips = []
        for start in range(0, len(frames) - self.frames_limit + 1, self.frame_step):
            clip = frames[start:start + self.frames_limit]
            clips.append(np.array(clip))  # Extracts multiple clips from one video.

        return clips

    def __len__(self):
        """
            Return the length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
            Retrieve a single training sample (clip and label) from the dataset.

            For a given index, the corresponding video is loaded and split into
            multiple clips. One clip is randomly selected to improve generalization.

            If the video is too short to produce any valid clip, another random
            sample is selected.
        """
        video_path, label = self.samples[idx]

        clips = self.extract_clips(video_path)  # Multipal clips from one video.

        if len(clips) == 0:  # If the video is too short -> random picking new one.
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        clip = random.choice(clips)  # Random sample from video for better generalization.

        clip = np.transpose(clip, (3, 0, 1, 2))  # Reorder dimensions for PyTorch: (T, H, W, C) -> (C, T, H, W).

        return torch.tensor(clip, dtype=torch.float32), torch.tensor(label)  # (C, T, H, W), scalar


def train_model(model, train_loader, val_loader, epochs, device):
    """
        Perform training and validation, save the best model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    best_val_acc = 0.0  # Tracking the best validation accuracy.

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # Reset gradients before backprop.
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)  # Get the class with the higher probability.
            correct += (preds == y).sum().item()
            total += y.size(0)  # Dimension 0 = batch size.

        train_acc = correct / total

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(f"Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUTPUT)  # Save best model weights.
            print("Model saved!")


if __name__ == "__main__":
    """
        Entry point for training the video classification model.

        Steps:
            - Set device to GPU if available, otherwise CPU.
            - Create training and validation data loaders.
            - Instantiate the model with the correct number of classes.
            - Train the model using the train_model function.
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        INPUT_FOLDER,
        BATCH_SIZE
    )

    # Create model
    model = create_model(num_classes=len(class_to_idx))

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        EPOCHS,
        device
    )
