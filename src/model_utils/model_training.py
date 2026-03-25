import random
from pathlib import Path
import os

import cv2
import numpy as np
import tensorflow as tf

from src.model_utils.baseline_model import create_model, WIDTH, HEIGHT, EPOCHS, BATCH_SIZE, FRAME_STEP, FRAMES_COUNT

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FOLDER = BASE_DIR / 'data' / 'videos'
OUTPUT_FOLDER = BASE_DIR / 'dataset_split'
MODEL_OUTPUT = BASE_DIR / 'models' / 'best_model.keras'


def preprocess_frame(frame):
    """
        Convert a raw BGR frame (from OpenCV) into a normalized RGB frame.
    """
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return frame


def extract_clips(video_path, frames_limit, frame_step):
    """
        Extract clips (subsequences of frames) from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)

        frames.append(frame)

    cap.release()

    clips = []
    for start in range(0, len(frames) - frames_limit + 1, frame_step):
        clip = frames[start: start + frames_limit]
        clips.append(np.array(clip))

    return clips


def load_dataset(dataset_dir, frames_limit, frame_step, batch_size):
    """
        Load dataset using a generator and convert it into a TensorFlow Dataset.
    """
    classes = sorted(os.listdir(dataset_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def generator():
        for cls in classes:
            cls_dir = os.path.join(dataset_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(('.mp4', '.avi', '.mov')):
                    path = os.path.join(cls_dir, fname)

                    clips = extract_clips(path, frames_limit, frame_step)
                    for clip in clips:
                        yield clip, class_to_idx[cls]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(frames_limit, HEIGHT, WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    return ds.shuffle(200).batch(batch_size).prefetch(tf.data.AUTOTUNE), class_to_idx


def split_dataset(source_dir, output_dir, splits, seed=42):
    """
        Split dataset into train/validation/test subsets.

        Each class is split independently while preserving class balance.
        Symbolic links are created instead of copying files.
    """
    random.seed(seed)

    classes = sorted(os.listdir(source_dir))

    # Create folders for output
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    stats = {split: 0 for split in splits}

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        videos = [f for f in os.listdir(cls_dir)
                  if f.endswith(('.mp4', '.avi', '.mov'))]

        total_needed = sum(splits.values())

        if len(videos) < total_needed:
            print(f"Class '{cls}': has {len(videos)} videos, "
                  f"needed {total_needed}")
            continue

        random.shuffle(videos)

        # Assign video to splits
        idx = 0
        for split, count in splits.items():
            for video in videos[idx:idx + count]:
                src = os.path.join(cls_dir, video)
                dst = os.path.join(output_dir, split, cls, video)
                os.symlink(os.path.abspath(src), dst)
            stats[split] += count
            idx += count


if __name__ == "__main__":
    """
        Main training pipeline:
            - Split dataset (if not already split)
            - Load datasets
            - Build and compile model
            - Train model with callbacks
    """

    if not os.path.exists(OUTPUT_FOLDER) or not os.listdir(OUTPUT_FOLDER):
        split_dataset(
            source_dir=INPUT_FOLDER,
            output_dir=OUTPUT_FOLDER,
            splits={"train": 30, "val": 10, "test": 10}
        )
    else:
        print("Dataset is already split.")

    train_ds, class_to_idx = load_dataset(OUTPUT_FOLDER / 'train', FRAMES_COUNT, FRAME_STEP, BATCH_SIZE)
    val_ds, _ = load_dataset(OUTPUT_FOLDER / 'val', FRAMES_COUNT, FRAME_STEP, BATCH_SIZE)

    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        # Stop if val_loss does not decrease for 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor='val_accuracy',
            save_best_only=True
        ),
        # Reduce learning rate when the model becomes stagnant
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
