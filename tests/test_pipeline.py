import os
import zipfile
from src.data_pipeline import DataPipeline


def test_init_creates_directory(tmp_path):
    """
    Tests if initializing the DataPipeline class successfully creates the target directory.
    """
    target_dir = tmp_path / "my_data"
    pipeline = DataPipeline(data_url="http://fake", data_dir=str(target_dir))

    assert os.path.exists(target_dir)
    assert pipeline.zip_path == os.path.join(str(target_dir), "subset_data.zip")


def test_is_data_already_extracted(tmp_path):
    """
    Verifies the pipeline's ability to detect if the dataset has already been extracted
    by checking for the existence of class subdirectories.
    """
    pipeline = DataPipeline(data_url="http://fake", data_dir=str(tmp_path))

    assert pipeline.is_data_already_extracted() is False

    os.makedirs(os.path.join(tmp_path, "person_reads_document"))

    assert pipeline.is_data_already_extracted() is True


def test_download_data_corrupted_file(tmp_path, monkeypatch):
    """
    Tests the error-handling logic that deletes small, corrupted ZIP files
    before attempting a new dataset download.
    """
    pipeline = DataPipeline(data_url="http://fake", data_dir=str(tmp_path))

    with open(pipeline.zip_path, "w") as f:
        f.write("fake corrupted data")

    assert os.path.exists(pipeline.zip_path)

    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr("src.data_pipeline.gdown.download", mock_download)
    pipeline.download_data()

    assert not os.path.exists(pipeline.zip_path)


def test_extract_data_success(tmp_path):
    """
    Tests the extraction process, ensuring that the ZIP archive is unpacked,
    macOS system files are cleaned up, and the nested folder structure is flattened.
    """
    pipeline = DataPipeline(data_url="http://fake", data_dir=str(tmp_path))

    nested_folder_name = "subset_data"
    test_file_name = "test_video.mp4"

    with zipfile.ZipFile(pipeline.zip_path, 'w') as zf:
        zf.writestr(f"{nested_folder_name}/{test_file_name}", "video binary data")
        zf.writestr("__MACOSX/._test_video.mp4", "junk data")

    pipeline.extract_data(cleanup=True)

    assert not os.path.exists(pipeline.zip_path)
    assert not os.path.exists(os.path.join(tmp_path, "__MACOSX"))
    assert not os.path.exists(os.path.join(tmp_path, nested_folder_name))
    assert os.path.exists(os.path.join(tmp_path, test_file_name))