import cv2
import torch
import os


def check_env():
    print("--- Environment Check ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.getBuildInformation()[:50]}...")

    data_path = "./data"
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        print(f"Files in data folder: {len(files)}")
    else:
        print("Data folder not found!")


if __name__ == "__main__":
    check_env()