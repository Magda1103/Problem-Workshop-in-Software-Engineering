from src.model_utils.alert_logic import AlertLogic


def test_safe_state():
    """
    Normal action should keep SAFE state.
    """
    logic = AlertLogic()

    result = logic.update(
        track_id=1,
        action="person_enters_car"
    )

    assert result["current_alert_state"] == "SAFE"
    assert result["anomaly_counter"] == 0


def test_warning_state():
    """
    First anomaly should trigger WARNING.
    """
    logic = AlertLogic()

    result = logic.update(
        track_id=1,
        action="person_steals_object"
    )

    assert result["current_alert_state"] == "WARNING"
    assert result["anomaly_counter"] == 1


def test_danger_threshold():
    """
    Exceeding threshold should trigger DANGER.
    """
    logic = AlertLogic(threshold=2)

    logic.update(1, "person_steals_object")
    logic.update(1, "person_steals_object")

    result = logic.update(
        1,
        "person_steals_object"
    )

    assert result["current_alert_state"] == "DANGER"
    assert result["danger_count"] == 1


def test_alert_reset():
    """
    Returning to normal action should reset alert.
    """
    logic = AlertLogic()

    logic.update(1, "person_steals_object")

    result = logic.update(
        1,
        "person_enters_car"
    )

    assert result["current_alert_state"] == "SAFE"
    assert result["anomaly_counter"] == 0