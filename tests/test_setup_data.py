# tests/test_setup_data.py

from unittest.mock import patch
from src.setup_data import setup_environment


@patch("src.setup_data.download_from_gdrive")
@patch("src.setup_data.Path.exists")
def test_download_called_when_zip_missing(mock_exists, mock_download):
    mock_exists.side_effect = [False, True]

    setup_environment()

    assert mock_download.called

@patch("src.setup_data.Path.exists")
def test_download_failed(mock_exists, capsys):
    mock_exists.return_value = False

    with patch("src.setup_data.download_from_gdrive"):
        setup_environment()

    captured = capsys.readouterr()

    assert "Download failed" in captured.out