import numpy as np


class Hypothesis:

    def __init__(self, prediction, measurement, measurement_prediction=None):
        self.prediction = prediction
        self.measurement = measurement
        self.measurement_prediction = measurement_prediction


class SingleProbabilityHypothesis(Hypothesis):

    def __init__(self, prediction, measurement, probability, measurement_prediction=None):
        super().__init__(prediction, measurement, measurement_prediction)
        self.probability = probability


class JointProbabilityHypothesis:

    def __init__(self, hypotheses):
        self.hypotheses = hypotheses
        self.probability = np.prod([h.probability for h in hypotheses], axis=0)
