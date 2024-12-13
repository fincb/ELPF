from itertools import product

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from ELPF.detection import MissedDetection
from ELPF.hypothesis import JointProbabilityHypothesis, SingleProbabilityHypothesis


class PDAHypothesiser:
    def __init__(
        self,
        measurement_model,
        detection_probability,
        clutter_spatial_density,
        likelihood_function,
        likelihood_function_args,
        gate_probability=0.95,
        include_all=False,
    ):
        """
        measurement_model : MeasurementModel
            The model used to convert states to measurements and vice versa.
        detection_probability : float
            Probability of detecting the target.
        clutter_spatial_density : float
            Density of clutter (false alarms) in the measurement space.
        probability_gate : float
            Gate threshold for probability of association.
        likelihood_function : function
            Function to calculate the likelihood of a measurement given a predicted measurement.
        likelihood_function_args : dict
            Additional arguments to pass to the likelihood function.
        """
        self.measurement_model = measurement_model
        self.detection_probability = detection_probability
        self.clutter_spatial_density = clutter_spatial_density
        self.likelihood_func = likelihood_function
        self.likelihood_func_args = likelihood_function_args
        self.gate_probability = gate_probability
        self.include_all = include_all

    def hypothesise(self, particle_state, detections):
        """
        Generates single hypotheses for each particle with respect to each measurement
        and missed detection hypothesis.

        Parameters
        ----------
        particle_state : ParticleState
            The state of the particles from which measurement predictions are generated.
        detections : np.ndarray
            Detections to be considered in hypothesis generation.

        Returns
        -------
        list of SingleHypothesis
            Hypotheses for each particle measurement combination, including missed detections.
        """
        # Predict measurements for all particles to compare to actual measurements
        predicted_measurements = self.measurement_model.function(
            particle_state, noise=False
        ).astype(np.float64)
        mean_predicted_measurement = np.mean(predicted_measurements, axis=1)
        covar = np.cov(predicted_measurements, rowvar=True)

        # Create a hypothesis for missed detection with a uniform probability across particles
        missed_detection_prob = np.full(
            (len(particle_state.particles),),
            1 - self.detection_probability * self.gate_probability,
        )
        hypotheses = [
            SingleProbabilityHypothesis(
                particle_state,
                MissedDetection(),
                missed_detection_prob,
                predicted_measurements,
            )
        ]

        gate_threshold = chi2.ppf(self.gate_probability, df=mean_predicted_measurement.shape[0])

        # Create a hypothesis for each particle-measurement pair
        for measurement in detections:
            measure = mahalanobis(
                mean_predicted_measurement.flatten(), measurement.state_vector.flatten(), covar
            )
            valid_measurement = measure <= gate_threshold or self.include_all
            if valid_measurement:
                # Compute differences between predicted and actual measurements for each particle
                diffs = predicted_measurements - measurement.state_vector

                # Calculate likelihood (probability) of this measurement for each particle
                probability = (
                    self.likelihood_func(diffs.T, **self.likelihood_func_args)
                    * self.detection_probability
                    / self.clutter_spatial_density
                )

                # Append hypothesis with computed probability and predicted measurement
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        particle_state, measurement, probability, predicted_measurements
                    )
                )

        return hypotheses


class JPDAHypothesiser(PDAHypothesiser):
    def hypothesise(self, particle_states, detections):
        """
        Generates valid joint hypotheses from single hypotheses, ensuring each measurement
        is assigned to one hypothesis only.

        Parameters
        ----------
        particle_states : list of ParticleState
            The states of the particles from which measurement predictions are generated.
        detections : np.ndarray
            Detections to be considered in hypothesis generation.

        Returns
        -------
        list of JointHypothesis
            Valid joint hypotheses across all particle states.
        """
        hypotheses = {
            particle_state: super().hypothesise(particle_state, detections)
            for particle_state in particle_states
        }

        # Generate valid joint hypotheses across all particle states
        # Create joint hypotheses by taking the Cartesian product of single hypotheses
        joint_hypotheses = list(product(*hypotheses.values()))
        valid_joint_hypotheses = []

        for joint_hypothesis in joint_hypotheses:
            measurements_assigned = set()
            valid = True

            for hypothesis in joint_hypothesis:
                measurement = hypothesis.measurement
                if isinstance(measurement, MissedDetection):
                    continue
                if measurement in measurements_assigned:
                    valid = False  # Invalidate if a measurement is assigned more than once
                    break
                measurements_assigned.add(measurement)

            if valid:
                valid_joint_hypotheses.append(JointProbabilityHypothesis(joint_hypothesis))

        # Normalise joint hypotheses probabilities
        total_prob = np.sum([np.sum(jh.probability) for jh in valid_joint_hypotheses])
        for jh in valid_joint_hypotheses:
            jh.probability /= total_prob

        # Recalculate probabilities for single hypotheses based on joint hypotheses
        self._redistribute_probabilities(hypotheses, valid_joint_hypotheses)

        return hypotheses

    def _redistribute_probabilities(self, hypotheses, joint_hypotheses):
        """
        Recalculates and assigns probabilities to single hypotheses based on the contributions
        from valid joint hypotheses.

        Parameters
        ----------
        hypotheses : dict
            Mapping of particle states to their respective single hypotheses.
        joint_hypotheses : list of JointHypothesis
            Valid joint hypotheses containing combinations of single hypotheses.
        """
        # Iterate over all hypotheses for each particle state to update their probabilities
        for single_hypotheses in hypotheses.values():
            for single_hypothesis in single_hypotheses:
                new_prob = 0.0

                # Accumulate contributions from joint hypotheses
                for joint_hypothesis in joint_hypotheses:
                    # Check if the single hypothesis is part of the current joint hypothesis
                    for jh in joint_hypothesis.hypotheses:
                        if jh == single_hypothesis:
                            new_prob += joint_hypothesis.probability  # Add contribution
                            break

                # Assign the recalculated probability to the single hypothesis
                single_hypothesis.probability = new_prob
