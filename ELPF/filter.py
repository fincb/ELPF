from datetime import timedelta

import numpy as np

from ELPF.state import Particle, ParticleState


class _ParticleFilter:
    def __init__(self, transition_model, measurement_model, likelihood_function=None):
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
        hypotheses: list,
    ) -> ParticleState:
        """
        Updates the weights of particles based on the likelihoods of observed measurements.

        Parameters
        ----------
        particle_state : ParticleState
            The current state of particles to be updated.
        detections : np.ndarray
            Array of detections.
        hypotheses : list of SingleHypothesis

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


class MultiTargetExpectedLikelihoodParticleFilter(ExpectedLikelihoodParticleFilter):
    """
    Particle Filter for multiple targets, using expected likelihoods to update particle states.
    This filter considers multiple hypotheses, handling missed detections, clutter, and
    measurement probabilities in a multi-target environment.
    """

    def update(self, particle_states, hypotheses):
        """
        Updates particle states based on measurements, calculating expected likelihoods for
        each particle-measurement combination.

        Parameters
        ----------
        particle_states : list of ParticleState
            Current states of all particles for each target.
        detections : np.ndarray
            Array of detections.
        hypotheses

        Returns
        -------
        list of ParticleState
            Updated states for each target after resampling based on new weights.
        """
        updated_states = []

        for particle_state in particle_states:
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
            updated_states.append(
                self.resample(ParticleState(new_particles, timestamp=particle_state.timestamp))
            )

        # Update weights for each particle state based on the new probabilities
        return updated_states


class PHDFilter:
    def __init__(self, initiator, measurement_model, num_particles=100):
        self.initiator = initiator
        self.measurement_model = measurement_model
        self.num_particles = num_particles

    def initiate(self, detections, timestamp):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            A list of unassociated detections
        timestamp: datetime.datetime
            Current timestamp

        Returns
        -------
        : set of :class:`~.Track`
            A set of new tracks with initial :class:`~.ParticleState`
        """
        tracks = set()

        for detection in detections:
            # Use inverse function to get the state vector from the detection
            state_vector = self.measurement_model.inverse_function(detection)

            # Get the model's covariance (for particle uncertainty)
            model_covar = self.measurement_model.covar()

            # Flatten state vector to make it compatible with sampling
            state_vector = state_vector.flatten()

            # Sample particles from a multivariate normal distribution around the state vector
            samples = np.random.multivariate_normal(
                mean=state_vector, cov=model_covar, size=self.num_particles
            )

            # Assign equal weights to the particles
            weights = np.ones(self.num_particles) / self.num_particles

            # Create particle objects
            particles = [Particle(sample, weight) for sample, weight in zip(samples, weights)]

            # Create the initial particle state
            prior = ParticleState(particles, timestamp=timestamp)

            # Add the prior (ParticleState) to the set of tracks
            tracks.add(prior)

        return tracks
