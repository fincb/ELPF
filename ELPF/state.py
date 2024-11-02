from datetime import datetime

import numpy as np

from ELPF.array import StateVector, StateVectors


class State:
    def __init__(self, state_vector: np.ndarray, timestamp: datetime = None):
        """
        Initialise a State instance with a given state vector.

        Parameters
        ----------
        state_vector : np.ndarray
            The state vector representing the state of the system.
        timestamp : datetime, optional
            The timestamp associated with the state.
        """
        self.state_vector = state_vector
        self.timestamp = timestamp


class GroundTruthState(State):
    def __init__(self, state_vector: np.ndarray, timestamp: datetime):
        """
        Initialise a GroundTruthState instance with a state vector and a timestamp.

        Parameters
        ----------
        state_vector : np.ndarray
            The state vector representing the ground truth state.
        timestamp : datetime, optional
            The timestamp associated with the ground truth state.
        """
        super().__init__(StateVector(state_vector), timestamp)


class Particle(State):
    def __init__(self, state_vector: np.ndarray, weight: float):
        """
        Initialise a Particle instance with a state vector and a weight.

        Parameters
        ----------
        state_vector : np.ndarray
            The state vector representing the particle's state.
            It should be a 1-dimensional array, which will be reshaped
            into a column vector (2D array with one column).
        weight : float
            The weight of the particle, representing its importance in the
            particle filter. The weight should be non-negative.
        timestamp : datetime, optional
            The timestamp associated with the particle state.
        """
        super().__init__(StateVector(state_vector))
        self.weight = weight


class ParticleState(State):
    def __init__(self, particles: list, timestamp: datetime):
        """
        Initialise a ParticleState instance with a list of particles.

        Parameters
        ----------
        particles : list
            A list of Particle objects that represent the particles
            in the particle filter.
        timestamp : datetime, optional
            The timestamp associated with the particle state.
        """
        self.particles = particles
        self.timestamp = timestamp  # Initialize timestamp in the State base class

    @property
    def state_vector(self) -> np.ndarray:
        """
        Returns the concatenated state vector of all particles.

        Returns
        -------
        np.ndarray
            A 2D array containing the concatenated state vectors of
            all particles.
        """
        return StateVectors([particle.state_vector for particle in self.particles])

    @property
    def weights(self) -> np.ndarray:
        """
        Returns the weights of all particles.

        Returns
        -------
        np.ndarray
            A 1D array containing the weights of each particle.
        """
        return np.array([particle.weight for particle in self.particles])

    @property
    def mean(self) -> np.ndarray:
        """
        Calculates the weighted mean of the particle states.

        Returns
        -------
        np.ndarray
            A 1D array representing the weighted mean of the particle states.
        """
        return np.average(self.state_vector, axis=1, weights=self.weights)

    @property
    def num_particles(self) -> int:
        """
        Returns the number of particles.

        Returns
        -------
        int
            The number of particles in the particle state.
        """
        return len(self.particles)
