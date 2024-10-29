import numpy as np
from scipy.stats import multivariate_normal

from state import Particle, ParticleState


class BootstrapParticleFilter:
    def __init__(self, transition_model, measurement_model):
        """
        Initialises the Bootstrap Particle Filter with a given transition and measurement model.

        Parameters
        ----------
        transition_model : ConstantVelocityTransitionModel
            The model used to predict the next state based on the current state.
        measurement_model : CartesianToRangeBearingMeasurementModel
            The model used to convert states to measurements and vice versa.
        """
        self.transition_model = transition_model  # Transition model for state prediction
        self.measurement_model = measurement_model  # Measurement model for state observation

    def predict(self, particle_state: ParticleState) -> ParticleState:
        """
        Predicts the next state of each particle based on the transition model.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of the particles.

        Returns
        -------
        ParticleState
            The predicted state of the particles after applying the transition model.
        """
        new_particles = []
        for particle in particle_state.particles:
            new_state = self.transition_model.function(particle)  # Predict new state
            new_particles.append(Particle(new_state, particle.weight))  # Keep the same weight
        return ParticleState(new_particles)

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
        new_particles = []
        for particle in particle_state.particles:
            # Predict measurement
            predicted_measurement = self.measurement_model.function(particle, noise=False)

            # Calculate the likelihood using the Gaussian PDF
            likelihood = self.gaussian_pdf(measurement, predicted_measurement)

            # Update the particle's weight
            new_particles.append(Particle(particle.state_vector, particle.weight * likelihood))

        # Normalise the weights
        total_weight = np.sum([p.weight for p in new_particles])
        if total_weight > 0:
            for p in new_particles:
                p.weight /= total_weight

        return self.resample(ParticleState(new_particles))

    def gaussian_pdf(self, observed_state: np.ndarray, predicted_state: np.ndarray) -> float:
        """
        Computes the likelihood of the observed state given the predicted state.

        Parameters
        ----------
        observed_state : np.ndarray
            The observed measurement state.
        predicted_state : np.ndarray
            The predicted measurement state from the particle.

        Returns
        -------
        float
            The likelihood of the observed state given the predicted state.
        """
        # Use the noise covariance matrix for the calculation
        covar = self.measurement_model.R

        # Calculate the difference between the observed and predicted states
        diff = observed_state - predicted_state

        # The likelihood is the probability of observing `observed_state` given the predicted
        # `predicted_state`
        likelihood = multivariate_normal.pdf(diff.flatten(), cov=covar)

        return likelihood

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
        threshold = num_particles / 2  # Threshold for resampling
        # Calculate Effective Sample Size (ESS)
        ess = 1 / np.sum(particle_state.weights**2)

        # Resample if the ESS is below the threshold
        if ess < threshold:
            weights = particle_state.weights  # Use weights directly
            cdf = np.cumsum(weights)  # Cumulative distribution function (CDF)
            cdf[-1] = 1.0  # Ensure the last value is exactly 1.0

            # Randomly pick the starting point
            u_i = np.random.uniform(0, 1 / num_particles)

            # Cycle through the cumulative distribution
            u_j = u_i + (1 / num_particles) * np.arange(num_particles)
            index = np.searchsorted(cdf, u_j)

            new_state_vector = np.array([particle_state.state_vector[:, i] for i in index]).T
            new_weight = (
                np.ones(num_particles) / num_particles
            )  # Equal weight for resampled particles
            new_particles = [
                Particle(state_vector, weight)
                for state_vector, weight in zip(new_state_vector.T, new_weight)
            ]
            return ParticleState(new_particles)
        else:
            return particle_state
