import os
import random
from collections import defaultdict
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
BASE_MODEL_INPUT = BASE_DIR / 'models' / 'best_model.pth'
MODEL_OUTPUT = BASE_DIR / 'models' / 'fine_tuned_model.pth'
STATS_OUTPUT = BASE_DIR / 'models' / 'fine_tuning_stats.txt'
FINE_TUNE_SAMPLES = 1000
SEED = 42
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)


def set_seed(seed=SEED):
    """
        Make fine-tuning repeatable between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def split_dataset(dataset_labels, val_ratio=0.2):
    """
        Split dataset into training and validation sets by returning indices.

        The dataset is randomly shuffled and then divided according
        to the given validation ratio.
    """
    all_labels = np.array(dataset_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    return train_idx, val_idx


def balanced_subset_indices(samples, max_samples=FINE_TUNE_SAMPLES, seed=SEED):
    """
        Select a repeatable, class-balanced subset for fine-tuning.
    """
    if max_samples is None or max_samples >= len(samples):
        return list(range(len(samples)))

    rng = random.Random(seed)
    indices_by_label = defaultdict(list)

    for idx, (_, label) in enumerate(samples):
        indices_by_label[label].append(idx)

    for indices in indices_by_label.values():
        rng.shuffle(indices)

    per_class = max(1, max_samples // len(indices_by_label))
    selected_indices = []

    for label in sorted(indices_by_label):
        selected_indices.extend(indices_by_label[label][:per_class])

    remaining_slots = max_samples - len(selected_indices)
    if remaining_slots > 0:
        remaining_indices = []
        for label in sorted(indices_by_label):
            remaining_indices.extend(indices_by_label[label][per_class:])
        rng.shuffle(remaining_indices)
        selected_indices.extend(remaining_indices[:remaining_slots])

    rng.shuffle(selected_indices)
    return selected_indices


def create_dataloaders(all_data_dir, batch_size, val_ratio=0.2):
    """
        Build datasets and dataloaders for training and validation.
    """
    classes = ['person_steals_object', 'person_enters_car', 'person_rides_bicycle', 'person_picks_up_object']  # Fixed class list for fine-tuning.
    class_to_idx = {cls: i for i, cls in enumerate(classes)}  # Maps class names to numeric labels.

    # Separate datasets keep train clips random and validation clips repeatable.
    train_source = VideoDataset(all_data_dir, FRAMES_COUNT, FRAME_STEP, class_to_idx, random_clip=True)
    val_source = VideoDataset(all_data_dir, FRAMES_COUNT, FRAME_STEP, class_to_idx, random_clip=False)

    # Select the same balanced samples every run.
    fine_tune_indices = balanced_subset_indices(train_source.samples)
    fine_tune_labels = [train_source.samples[i][1] for i in fine_tune_indices]

    # Split the selected samples into training and validation sets
    train_local_idx, val_local_idx = split_dataset(fine_tune_labels, val_ratio=val_ratio)

    # Map local indices back to the original full_dataset indices
    train_final_idx = [fine_tune_indices[i] for i in train_local_idx]
    val_final_idx = [fine_tune_indices[i] for i in val_local_idx]

    train_dataset = Subset(train_source, train_final_idx)
    val_dataset = Subset(val_source, val_final_idx)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)  # Keep validation deterministic (shuffle=False).

    return train_loader, val_loader, class_to_idx


class VideoDataset(Dataset):
    """
        Converts raw videos into tensors for the model.
    """

    def __init__(self, dataset_dir, frames_limit, frame_step, class_to_idx, random_clip=True):
        self.samples = []
        self.frames_limit = frames_limit
        self.frame_step = frame_step
        self.class_to_idx = class_to_idx
        self.random_clip = random_clip

        for cls in os.listdir(dataset_dir):
            if cls not in class_to_idx:
                continue
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
            multiple clips. Training can use a random clip, while validation uses
            a fixed center clip to make metrics more stable.

            If the video is too short to produce any valid clip, another random
            sample is selected.
        """
        video_path, label = self.samples[idx]

        clips = self.extract_clips(video_path)  # Multipal clips from one video.

        if len(clips) == 0:  # If the video is too short -> picking another sample.
            next_idx = random.randint(0, len(self.samples) - 1) if self.random_clip else (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)

        if self.random_clip:
            clip = random.choice(clips)  # Random sample from video for better generalization.
        else:
            clip = clips[len(clips) // 2]

        clip = np.transpose(clip, (3, 0, 1, 2))  # Reorder dimensions for PyTorch: (T, H, W, C) -> (C, T, H, W).

        return torch.tensor(clip, dtype=torch.float32), torch.tensor(label)  # (C, T, H, W), scalar


def load_base_model_weights(model, base_model_path=BASE_MODEL_INPUT):
    """
        Initialize fine-tuning from the 9-class baseline model.
        The 4-class classifier layer is skipped because its shape is different.
    """
    if not base_model_path.exists():
        print(f"Base model not found at {base_model_path}. Fine-tuning will start from ImageNet weights.")
        return model

    checkpoint = torch.load(base_model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()

    matching_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }

    model_state.update(matching_state)
    model.load_state_dict(model_state)

    skipped_layers = len(state_dict) - len(matching_state)
    print(f"Loaded {len(matching_state)} layers from {base_model_path.name}; skipped {skipped_layers} incompatible layers.")

    return model


def train_model(model, train_loader, val_loader, epochs, device):
    """
        Perform training and validation, save the best model.
    """
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    best_val_acc = -1.0  # Tracking the best validation accuracy.
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    with open(STATS_OUTPUT, 'w') as f:
        f.write("Epoch,Loss,Train Acc,Val Acc\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        total_loss, correct, total = 0, 0, 0

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

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUTPUT)  # Save best model weights.
            print("Model saved!")
        
        with open(STATS_OUTPUT, 'a') as f:
            f.write(f"{epoch + 1},{avg_loss:.4f},{train_acc:.4f},{val_acc:.4f}\n")
        print(f"Training statistics saved to {STATS_OUTPUT}")


if __name__ == "__main__":
    """
        Entry point for training the video classification model.

        Steps:
            - Set device to GPU if available, otherwise CPU.
            - Create training and validation data loaders.
            - Instantiate the model with the correct number of classes.
            - Train the model using the train_model function.
    """

    set_seed()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        INPUT_FOLDER,
        BATCH_SIZE
    )

    # Create model
    model = create_model(num_classes=len(class_to_idx))
    model = load_base_model_weights(model)

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        EPOCHS,
        device
    )
