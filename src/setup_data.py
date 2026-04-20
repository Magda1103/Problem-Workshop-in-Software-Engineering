import os
import zipfile
import shutil
from pathlib import Path


GDRIVE_FILE_ID = "1f6pnSHSF341ubSudc04OAAVhiIDMYCPH"
ZIP_FILENAME = "sprint2_data.zip"

BASE_DIR = Path(__file__).resolve().parent

def download_from_gdrive(file_id, destination):
    """
    Downloads a file directly from Google Drive using the gdown library.
    """
    try:
        import gdown
    except ImportError:
        print("Error: 'gdown' library is missing.")
        print("Please run this command in your terminal first:")
        print("pip install gdown")
        exit(1)

    print(f"⬇Downloading data from Google Drive (this may take a moment)...")
    url = f"https://drive.google.com/uc?id={file_id}"
    
    gdown.download(url, str(destination), quiet=False)

def setup_environment():
    """
    Main function to download and extract the dataset and model weights.
    """
    zip_path = BASE_DIR / ZIP_FILENAME

    if not zip_path.exists():
        if GDRIVE_FILE_ID == "YOUR_FILE_ID_HERE":
            print("Error: You forgot to paste your Google Drive File ID in the script!")
            print("Please edit setup_data.py and update GDRIVE_FILE_ID.")
            return
        download_from_gdrive(GDRIVE_FILE_ID, zip_path)

    if not zip_path.exists():
        print("Error: Download failed.")
        return

    print(f"📦 Found '{ZIP_FILENAME}'. Starting extraction...")

    models_dir = BASE_DIR / 'models'
    videos_dir = BASE_DIR / 'data' / 'videos'
    models_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue

                filename = file_info.filename

                if filename.endswith('.pth'):
                    target_path = models_dir / Path(filename).name
                    with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    print(f"Extracted model: {target_path.name} -> models/")

                elif filename.endswith(('.mp4', '.avi')):
                    # Extract the action class name from the zip folder structure
                    parts = Path(filename).parts
                    if len(parts) >= 2:
                        action_name = parts[-2]
                        
                        # Create specific subfolder for the action (e.g., person_steals_object)
                        action_dir = videos_dir / action_name
                        action_dir.mkdir(parents=True, exist_ok=True)
                        
                        target_path = action_dir / Path(filename).name
                        with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        print(f"✅ Extracted video: {Path(filename).name} -> data/videos/{action_name}/")

        print("\nSetup complete! All files are in their correct directories.")
        print("YOLOv8 weights will be downloaded automatically upon first run.")
        print("You can now run the inference engine:")
        print("python -m src.model_utils.inference_engine")

    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    setup_environment()