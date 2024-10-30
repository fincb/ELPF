import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

from state import Particle, ParticleState


class ParticleFilter:
    def __init__(self, transition_model, measurement_model):
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
        new_states = [
            self.transition_model.function(particle) for particle in particle_state.particles
        ]
        return ParticleState(
            [
                Particle(state, particle.weight)
                for state, particle in zip(new_states, particle_state.particles)
            ]
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
                Particle(state_vector, 1.0 / num_particles) for state_vector in new_state_vector.T
            ]

            return ParticleState(new_particles)
        return particle_state


class BootstrapParticleFilter(ParticleFilter):

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

        # Calculate the likelihood using the multivariate normal distribution
        likelihood = multivariate_normal.pdf(diff.flatten(), cov=covar)

        return likelihood


class ExpectedLikelihoodParticleFilter(ParticleFilter):

    def update(self, particle_state: ParticleState, measurements: np.ndarray) -> ParticleState:
        """
        Updates the weights of each particle based on the association probabilities of
        measurements.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of the particles.
        measurements : np.ndarray
            Array of measurements to be associated with particles.

        Returns
        -------
        ParticleState
            The updated particle state after weighting based on expected likelihoods.
        """
        association_probabilities = np.zeros((particle_state.num_particles, len(measurements)))

        # Calculate likelihoods of each particle for each measurement
        for i, particle in enumerate(particle_state.particles):
            for j, measurement in enumerate(measurements):
                predicted_measurement = self.measurement_model.function(particle, noise=False)
                likelihood = self.t_pdf(measurement.state_vector, predicted_measurement)
                association_probabilities[i, j] = likelihood

        # Update particle weights using PDA and normalise
        new_particles = []
        for i, particle in enumerate(particle_state.particles):
            expected_likelihood = np.mean(
                association_probabilities[i, :]
            )  # Expected likelihood over measurements
            new_weight = particle.weight * expected_likelihood
            new_particles.append(Particle(particle.state_vector, new_weight))

        # Normalise weights to sum to 1
        total_weight = sum(p.weight for p in new_particles)
        if total_weight > 0:
            for p in new_particles:
                p.weight /= total_weight

        # Return resampled ParticleState if needed
        return self.resample(ParticleState(new_particles))

    def t_pdf(self, observed_state: np.ndarray, predicted_state: np.ndarray) -> float:
        """
        Computes the likelihood of the observed state given the predicted state
        using a multivariate Student's t-distribution.

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
        # Use the noise covariance matrix from the measurement model
        covar = self.measurement_model.R

        # Degrees of freedom for the t-distribution
        df = observed_state.size

        # Calculate the difference
        diff = observed_state - predicted_state

        # Ensure the covariance is a 2D matrix
        if covar.ndim == 1:  # Convert 1D array to 2D covariance matrix if necessary
            covar = np.diag(covar)

        # Calculate the likelihood using the multivariate t-distribution
        likelihood = multivariate_t.pdf(diff.flatten(), df=df, shape=covar)

        return likelihood
