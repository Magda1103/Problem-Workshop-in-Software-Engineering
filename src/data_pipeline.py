import os
import zipfile
import gdown
import shutil
from typing import List

class DataPipeline:
    """
    Class responsible for automating the download and extraction of the PIP 370k subset.
    Optimized for Google Drive large file downloads using gdown.
    """

    def __init__(self, data_url: str, data_dir: str = "./data"):
        """
        Initializes the pipeline with the direct Google Drive URL and target directory.
        """
        self.data_url = data_url
        self.data_dir = os.path.abspath(data_dir)
        self.zip_path = os.path.join(self.data_dir, "subset_data.zip")

        # Create the data directory structure if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def is_data_already_extracted(self) -> bool:
        """
        Checks if the data is already present in the target directory.
        Prevents redundant 10GB downloads by looking for existing class subfolders.
        """
        if not os.path.exists(self.data_dir):
            return False

        # Look for any subdirectories within the data folder (e.g., class labels)
        items = os.listdir(self.data_dir)
        folders = [i for i in items if os.path.isdir(os.path.join(self.data_dir, i))]

        return len(folders) > 0

    def download_data(self) -> None:
        """
        Downloads the ZIP archive from Google Drive using gdown to bypass virus scan warnings.
        """
        if self.is_data_already_extracted():
            return

        if os.path.exists(self.zip_path) and os.path.getsize(self.zip_path) < 1000000:
            print("[!] Found corrupted small file. Deleting and restarting download...")
            os.remove(self.zip_path)

        if os.path.exists(self.zip_path):
            print(f"[*] Archive {self.zip_path} already exists. Skipping download.")
            return

        print(f"[*] Starting GDOWN download from: {self.data_url}")
        try:
            # gdown handles Google Drive confirmation pages for large files automatically
            gdown.download(self.data_url, self.zip_path, quiet=False)
            print("[+] Download complete.")
        except Exception as e:
            print(f"[!] Gdown download error: {e}")

    def extract_data(self, cleanup: bool = True) -> None:
        """
        Extracts the ZIP archive and optionally removes it to save disk space.
        """
        if self.is_data_already_extracted():
            return

        if not os.path.exists(self.zip_path):
            print("[!] ZIP archive not found.")
            return

        print(f"[*] Extracting to {self.data_dir}...")
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("[+] Extraction successful.")

            macosx_path = os.path.join(self.data_dir, "__MACOSX")
            if os.path.exists(macosx_path):
                shutil.rmtree(macosx_path)
                print("[*] Cleaned up __MACOSX.")

            nested_folder = os.path.join(self.data_dir, "subset_data")
            if os.path.exists(nested_folder):
                print("[*] Moving files from nested folder to root data directory...")
                for item in os.listdir(nested_folder):
                    s = os.path.join(nested_folder, item)
                    d = os.path.join(self.data_dir, item)
                    shutil.move(s, d)
                os.rmdir(nested_folder)
                print("[+] Folder structure flattened.")

            if cleanup:
                os.remove(self.zip_path)
                print("[+] ZIP archive removed.")

        except Exception as e:
            print(f"[!] Error during extraction/cleanup: {e}")

    def run(self):
        """
        Main method to execute the full pipeline: Download -> Extract.
        """
        if self.is_data_already_extracted():
            print("[*] Data is already available in the /data folder. Skipping pipeline.")
            return

        self.download_data()
        self.extract_data(cleanup=True)

if __name__ == "__main__":
    FILE_ID = "1_e0wwfALr0tFiE_REOsAt1UnUJoYQGD6"
    DIRECT_URL = f"https://drive.google.com/uc?id={FILE_ID}"

    pipeline = DataPipeline(data_url=DIRECT_URL)
    pipeline.run()