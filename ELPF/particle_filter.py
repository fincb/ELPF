from datetime import timedelta
from itertools import product

import numpy as np

from ELPF.detection import MissedDetection
from ELPF.hypothesis import JointHypothesis, SingleHypothesis
from ELPF.state import Particle, ParticleState


class _ParticleFilter:
    def __init__(self, transition_model, measurement_model, likelihood_function):
        """
        Initialises the Particle Filter with a given transition and measurement model.

        Parameters
        ----------
        transition_model : TransitionModel
            The model used to predict the next state based on the current state.
        measurement_model : MeasurementModel
            The model used to convert states to measurements and vice versa.
        """
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.likelihood_function = likelihood_function

    def predict(self, particle_state: ParticleState, time_interval: timedelta) -> ParticleState:
        """
        Predicts the next state of each particle based on the transition model.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of the particles.
        time_interval : timedelta
            The time interval for the transition model.

        Returns
        -------
        ParticleState
            The predicted state of the particles after applying the transition model.
        """
        new_states = self.transition_model.function(particle_state, time_interval)
        return ParticleState(
            [
                Particle(state, particle.weight)
                for state, particle in zip(new_states.T, particle_state.particles)
            ],
            timestamp=particle_state.timestamp + time_interval,
        )

    def resample(self, particle_state: ParticleState) -> ParticleState:
        """
        Resamples particles based on their weights if the Effective Sample Size (ESS) is below
        a threshold.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of the particles.

        Returns
        -------
        ParticleState
            The resampled particle state, or the original state if no resampling is needed.
        """
        num_particles = particle_state.num_particles
        ess = 1 / np.sum(particle_state.weights**2)

        if ess < (num_particles / 2):
            weights = particle_state.weights
            cdf = np.cumsum(weights)
            cdf[-1] = 1.0  # Ensure last value is 1.0
            u_i = np.random.uniform(0, 1 / num_particles)
            u_j = u_i + (1 / num_particles) * np.arange(num_particles)
            index = np.searchsorted(cdf, u_j)

            # Create new particles based on the resampled indices
            new_state_vector = particle_state.state_vector[:, index]
            new_particles = [
                Particle(state_vector, 1.0 / num_particles) for state_vector in new_state_vector
            ]

            return ParticleState(new_particles, timestamp=particle_state.timestamp)
        return particle_state


class BootstrapParticleFilter(_ParticleFilter):

    def update(self, particle_state: ParticleState, measurement: np.ndarray) -> ParticleState:
        """
        Updates the particle weights based on the measurement and resamples if necessary.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of the particles.
        measurement : np.ndarray
            The measurement received from the environment.

        Returns
        -------
        ParticleState
            The updated particle state after incorporating the measurement.
        """
        predicted_measurements = self.measurement_model.function(particle_state, noise=False)

        # Calculate likelihoods of each particle for the measurement
        likelihoods = self.likelihood_function(
            measurement.state_vector, predicted_measurements, self.measurement_model.covar
        )

        weights = np.array([particle.weight for particle in particle_state.particles])
        new_weights = weights * likelihoods

        # Create new particles with updated weights
        new_particles = [
            Particle(particle.state_vector, new_weight)
            for particle, new_weight in zip(particle_state.particles, new_weights)
        ]

        # Normalise weights to sum to 1
        total_weight = np.sum(new_weights)
        if total_weight > 0:
            # Normalise all weights at once
            for p in new_particles:
                p.weight /= total_weight

        return self.resample(ParticleState(new_particles))


class ExpectedLikelihoodParticleFilter(_ParticleFilter):
    """
    Particle Filter that calculates and updates particle weights based on expected likelihoods
    of multiple measurements. This filter handles missed detections, clutter, and measurement
    probabilities in a multi-hypothesis framework.
    """

    def update(
        self,
        particle_state: ParticleState,
        measurements: np.ndarray,
        detection_probability: float,
        clutter_spatial_density: float,
        likelihood_func_args,
    ) -> ParticleState:
        """
        Updates the weights of particles based on the likelihoods of observed measurements.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of particles to be updated.
        measurements : np.ndarray
            Array of observed measurements.
        detection_probability : float
            Probability of detecting the target.
        clutter_spatial_density : float
            Density of clutter in the measurement space.

        Returns
        -------
        ParticleState
            The updated particle state after resampling based on the new weights.
        """
        # Generate hypotheses for each particle with each measurement and missed detection
        hypotheses = self._generate_single_hypotheses(
            particle_state,
            measurements,
            detection_probability,
            clutter_spatial_density,
            likelihood_func_args,
        )

        return self._update_weights(particle_state, hypotheses)

    def _update_weights(self, particle_state, hypotheses):
        """
        Updates particle weights based on calculated likelihoods from hypotheses and
        normalises them.

        This method calculates the new weights by averaging the likelihoods
        across all hypotheses for each particle and then normalises the weights to ensure
        they sum to one. Finally, it generates a new set of particles based on the updated
        weights and performs resampling.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of particles to be updated, containing individual particle
            weights and state vectors.
        hypotheses : list of SingleHypothesis
            List of hypotheses representing the likelihood of each measurement for each particle.

        Returns
        -------
        ParticleState
            The updated particle state after resampling based on the new weights.
        """
        # Extract particle weights from the particle state
        weights = np.array([particle.weight for particle in particle_state.particles])

        # Compute likelihoods for each hypothesis (array of probabilities for each particle)
        likelihoods = np.array([h.probability for h in hypotheses])

        # Update weights based on mean likelihood across hypotheses and normalize
        new_weights = weights * likelihoods.mean(axis=0)
        new_weights /= np.sum(new_weights)  # Normalise to ensure weights sum to 1

        # Generate new particles with updated weights
        new_particles = [
            Particle(particle.state_vector, weight)
            for particle, weight in zip(particle_state.particles, new_weights)
        ]

        # Resample particles based on updated weights and return new particle state
        return self.resample(ParticleState(new_particles, timestamp=particle_state.timestamp))

    def _generate_single_hypotheses(
        self,
        particle_state,
        measurements,
        detection_probability,
        clutter_spatial_density,
        likelihood_func_args,
    ):
        """
        Generates single hypotheses for each particle with respect to each measurement
        and missed detection hypothesis.

        Parameters
        ----------
        particle_state : ParticleState
            The state of the particles from which measurement predictions are generated.
        measurements : np.ndarray
            Observed measurements to be considered in hypothesis generation.
        detection_probability : float
            Probability of detecting the target.
        clutter_spatial_density : float
            Density of clutter (false alarms) in the measurement space.

        Returns
        -------
        list of SingleHypothesis
            Hypotheses for each particle measurement combination, including missed detections.
        """
        # Predict measurements for all particles to compare to actual measurements
        predicted_measurements = self.measurement_model.function(particle_state, noise=False)

        # Create a hypothesis for missed detection with a uniform probability across particles
        missed_detection_prob = np.full(
            (len(particle_state.particles),), 1 - detection_probability
        )
        hypotheses = [
            SingleHypothesis(
                particle_state,
                MissedDetection(),
                missed_detection_prob,
                predicted_measurements,
            )
        ]

        # Create a hypothesis for each particle-measurement pair
        for measurement in measurements:
            # Compute differences between predicted and actual measurements for each particle
            diffs = predicted_measurements - measurement.state_vector

            # Calculate likelihood (probability) of this measurement for each particle
            probability = (
                self.likelihood_function(diffs.T, **likelihood_func_args)
                * detection_probability
                / clutter_spatial_density
            )

            # Append hypothesis with computed probability and predicted measurement
            hypotheses.append(
                SingleHypothesis(particle_state, measurement, probability, predicted_measurements)
            )

        return hypotheses


class MultiTargetExpectedLikelihoodParticleFilter(ExpectedLikelihoodParticleFilter):
    """
    Particle Filter for multiple targets, using expected likelihoods to update particle states.
    This filter considers multiple hypotheses, handling missed detections, clutter, and
    measurement probabilities in a multi-target environment.
    """

    def update(
        self,
        particle_states,
        measurements,
        detection_probability,
        clutter_spatial_density,
        likelihood_func_args,
    ):
        """
        Updates particle states based on measurements, calculating expected likelihoods for
        each particle-measurement combination.

        Parameters
        ----------
        particle_states : list of ParticleState
            Current states of all particles for each target.
        measurements : np.ndarray
            Array of observed measurements.
        detection_probability : float
            Probability of detecting a target.
        clutter_spatial_density : float
            Density of clutter (false alarms) in the measurement space.

        Returns
        -------
        list of ParticleState
            Updated states for each target after resampling based on new weights.
        """

        # Generate hypotheses for each particle state with each measurement and missed detection
        hypotheses = {
            particle_state: self._generate_single_hypotheses(
                particle_state,
                measurements,
                detection_probability,
                clutter_spatial_density,
                likelihood_func_args,
            )
            for particle_state in particle_states
        }

        # Generate valid joint hypotheses across all particle states
        joint_hypotheses = self._generate_joint_hypotheses(hypotheses)

        # Recalculate probabilities for single hypotheses based on joint hypotheses
        self._redistribute_probabilities(hypotheses, joint_hypotheses)

        # Update weights for each particle state based on the new probabilities
        return [
            self._update_weights(particle_state, hypotheses[particle_state])
            for particle_state in particle_states
        ]

    def _generate_joint_hypotheses(self, hypotheses):
        """
        Generates valid joint hypotheses from single hypotheses, ensuring each measurement
        is assigned to one hypothesis only.

        Parameters
        ----------
        hypotheses : dict
            Mapping of particle states to their respective single hypotheses.

        Returns
        -------
        list of JointHypothesis
            Valid joint hypotheses across all particle states.
        """

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
                valid_joint_hypotheses.append(JointHypothesis(joint_hypothesis))

        # Normalise joint hypotheses probabilities
        total_prob = np.sum([np.sum(jh.probability) for jh in valid_joint_hypotheses])
        for jh in valid_joint_hypotheses:
            jh.probability /= total_prob

        return valid_joint_hypotheses

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
