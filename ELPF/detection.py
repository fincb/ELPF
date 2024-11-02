class Detection:

    def __init__(self, state_vector, measurement_model, timestamp=None):
        self.state_vector = state_vector
        self.measurement_model = measurement_model
        self.timestamp = timestamp


class TrueDetection(Detection):
    """Class representing a true detection in the measurement space."""


class Clutter(Detection):
    """Class representing clutter in the measurement space."""
