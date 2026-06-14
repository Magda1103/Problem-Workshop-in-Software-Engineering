import numpy as np
from collections import deque
from queue import Queue
import json
from collections import Counter
from unittest.mock import patch
from pathlib import Path


from src.model_utils.inference_engine import InferenceEngine


@patch("src.model_utils.inference_engine.YOLO")
@patch("src.model_utils.inference_engine.torch.load")
@patch("src.model_utils.inference_engine.ActionRecognition")
def test_sliding_window_logic(mock_model, mock_load, mock_yolo):
    """
    Verify sliding window enqueue condition.
    """

    engine = InferenceEngine(
        frame_step=5,
        frames_limit=10,
        video_path=Path("fake_video.mp4")
    )

    track_id = 1

    engine.buffers[track_id] = deque(
        [np.zeros((224, 224, 3)) for _ in range(10)],
        maxlen=10
    )

    engine.frame_counts[track_id] = 10

    if (
        engine.frame_counts[track_id] % engine.frame_step == 0
        and len(engine.buffers[track_id]) == engine.frames_limit
    ):
        window = np.array(
            list(engine.buffers[track_id]),
            dtype=np.float32
        )

        engine.queue.put((track_id, window, 0))

    assert engine.queue.qsize() == 1

def test_guess_indoor_environment():
    engine = object.__new__(InferenceEngine)

    result = engine._guess_environment({
        "chair": 10,
        "tv": 5
    })

    assert result == "indoors"

def test_guess_outdoor_environment():
    engine = object.__new__(InferenceEngine)

    result = engine._guess_environment({
        "car": 10,
        "bus": 5
    })

    assert result == "outdoors"

def test_guess_unknown_environment():
    engine = object.__new__(InferenceEngine)

    result = engine._guess_environment({})

    assert result == "unknown"

def test_majority_voting():


    history = deque(maxlen=5)

    history.append("person_enters_car")
    history.append("person_enters_car")
    history.append("person_steals_object")
    history.append("person_enters_car")

    result = Counter(history).most_common(1)[0][0]

    assert result == "person_enters_car"




def test_queue_behavior():
    queue = Queue(maxsize=2)

    queue.put("a")
    queue.put("b")

    assert queue.full() is True
    assert queue.qsize() == 2

def test_stale_track_detection():
    frame_index = 100

    last_seen = {
        1: 60,
        2: 95
    }

    stale_ids = [
        tid
        for tid, last_seen_frame in last_seen.items()
        if frame_index - last_seen_frame > 30
    ]

    assert stale_ids == [1]





@patch("src.model_utils.inference_engine.YOLO")
@patch("src.model_utils.inference_engine.torch.load")
@patch("src.model_utils.inference_engine.ActionRecognition")
def test_json_export(mock_model, mock_load, mock_yolo, tmp_path):
    """
    Verify JSON export of inference results.
    """



    engine = InferenceEngine(
        frame_step=5,
        frames_limit=10,
        video_path=Path("fake_video.mp4")
    )

    engine.latest_predictions = {
        1: "person_enters_car"
    }

    engine.latest_confidences = {
        1: 91.2
    }

    engine.latest_alerts = {
        1: {
            "current_alert_state": "SAFE",
            "max_alert_state": "SAFE",
            "anomaly_counter": 0,
            "danger_count": 0,
            "warning_count": 0
        }
    }

    engine.scene_object_counts = Counter({
        "car": 3,
        "person": 1
    })

    engine.people_seen = {1}



    with patch(
        "src.model_utils.inference_engine.BASE_DIR",
        tmp_path
    ):
        (tmp_path / "src" / "model_utils").mkdir(
            parents=True,
            exist_ok=True
        )

        engine.save_json_results()

        assert (
            tmp_path
            / "src"
            / "model_utils"
            / "inference_results.json"
        ).exists()

        data = json.loads(
            (
                tmp_path
                / "src"
                / "model_utils"
                / "inference_results.json"
            ).read_text()
        )

        assert len(data) == 1
        assert data[0]["track_id"] == 1
        assert data[0]["final_action"] == "person_enters_car"