# Actions treated as dangerous anomalies
ANOMALIES = {"person_steals_object"}

#Number of anomaly frames needed to trigger DANGER state
ALERT_THRESHOLD = 5

class AlertLogic:
    """
    Simple alert system for anomaly actions
    The class checks actions for each tracked person
    and changes the alert state when an anomaly
    continues for multiple frames.
    """
    def __init__(self, threshold=ALERT_THRESHOLD, anomalies=None):
        self.threshold = threshold
        self.anomalies = anomalies or ANOMALIES
        self.counters = {}
        self.states = {}
        self.max_states = {}
        self.danger_counts = {}
        self.warning_counts = {}



    def update(self, track_id, action):
        """
        Updates alert state for tracked person
        Args:
            track_id: ID of tracked person
            action: Current predicted action
        """

        # Create default values for new tracked person
        if track_id not in self.counters:
            self.counters[track_id] = 0
            self.states[track_id] = "SAFE"

            self.max_states[track_id] = "SAFE"
            self.danger_counts[track_id] = 0
            self.warning_counts[track_id] = 0

        previous_state = self.states[track_id]

        # Increase anomaly counter if action is dangerous
        if action in self.anomalies:
            self.counters[track_id] += 1

        # Reset counter and state if anomaly disappears
        else:
            self.counters[track_id] = 0
            self.states[track_id] = "SAFE"

        # Trigger DANGER state after threshold is exceeded
        if self.counters[track_id] > self.threshold:
            self.states[track_id] = "DANGER"
            self.max_states[track_id] = "DANGER"
            if previous_state != "DANGER":
                self.danger_counts[track_id] += 1

        # Temporary warning state before danger alert
        elif self.counters[track_id] > 0:
            self.states[track_id] = "WARNING"

            if self.max_states[track_id] != "DANGER":
                self.max_states[track_id] = "WARNING"

            if previous_state != "WARNING":
                self.warning_counts[track_id] += 1

        # Return alert information
        return {
            "track_id": int(track_id),
            "action": action,

            "current_alert_state": self.states[track_id],
            "max_alert_state": self.max_states[track_id],

            "anomaly_counter": self.counters[track_id],

            "danger_count": self.danger_counts[track_id],
            "warning_count": self.warning_counts[track_id],
        }

    def remove_track(self, track_id):
        """
        Removes data for inactive tracked person
        """
        self.counters.pop(track_id, None)
        self.states.pop(track_id, None)

        self.max_states.pop(track_id, None)
        self.danger_counts.pop(track_id, None)
        self.warning_counts.pop(track_id, None)


