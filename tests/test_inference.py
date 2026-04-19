import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.model_utils.inference_engine import InferenceEngine


@patch("src.model_utils.inference_engine.torch.load")
@patch("src.model_utils.inference_engine.ActionRecognition")
@patch("src.model_utils.inference_engine.cv2.VideoCapture")
def test_sliding_window_logic(mock_vidcap, mock_model_class, mock_torch_load):
    """
    Verifies the mathematical logic of the Sliding Window mechanism, including buffer
    size limits and queue appending intervals, using mocked video frames without loading real models.
    """
    mock_cap_instance = MagicMock()
    fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)

    returns = [(True, fake_frame)] * 15 + [(False, None)]
    mock_cap_instance.read.side_effect = returns
    mock_cap_instance.isOpened.return_value = True

    mock_vidcap.return_value = mock_cap_instance

    engine = InferenceEngine(frame_step=5, frames_limit=10, path="fake_video.mp4")

    with patch.object(engine, 'run_inference') as mock_run:
        engine.perform_using_sliding_window()

        assert len(engine.buffer) == 10
        assert engine.queue.qsize() == 2