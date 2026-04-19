import os


def test_environment_data_folder_ready():
    """
    Checks if the environment has the necessary data folder before starting the training.
    """
    data_path = "data/videos"
    assert os.path.exists(data_path), f"Directory {data_path} does not exist. Run data_pipeline.py first."

    if os.path.exists(data_path):
        classes = os.listdir(data_path)
        assert len(classes) > 0, "The data folder is empty!"