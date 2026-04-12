import pytest
from pathlib import Path
from src.eda import extract_scene_payload, _activity_label_text, reservoir_sample_jsonl


def test_extract_scene_payload_invalid():
    """
    Verifies that the function raises a ValueError when provided with an empty record.
    """
    with pytest.raises(ValueError, match="Record is empty"):
        extract_scene_payload({})


def test_activity_label_text():
    """
    Checks if the category label is correctly formatted into a human-readable string.
    """
    label = _activity_label_text("person_reads_document")
    assert label == "person reading a book or document"


def test_reservoir_sample_jsonl(tmp_path):
    """
    Tests the reservoir sampling mechanism by creating a temporary JSONL file
    and ensuring it returns a valid dictionary object.
    """
    fake_jsonl = tmp_path / "fake_annotations.jsonl"
    fake_jsonl.write_text('{"scene_1": {"category": "test"}}\n{"scene_2": {"category": "test2"}}')

    result = reservoir_sample_jsonl(fake_jsonl)
    assert isinstance(result, dict)