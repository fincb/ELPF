import numpy as np


class State:
    def __init__(self, state_vector: np.ndarray):
        """
        Initialise a State instance with a given state vector.

        Parameters
        ----------
        state_vector : np.ndarray
            The state vector representing the state of the system.
            It should be a 1-dimensional array, which will be reshaped
            into a column vector (2D array with one column).

        Attributes
        ----------
        state_vector : np.ndarray
            The state vector stored as a column vector (shape: (n, 1)),
            where n is the number of elements in the input state vector.
        """
        self.state_vector = state_vector.reshape(-1, 1)


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

        Attributes
        ----------
        weight : float
            The weight of the particle, which is used during the resampling
            process in the particle filter.
        """
        super().__init__(state_vector)
        self.weight = weight


class ParticleState(State):
    def __init__(self, particles: list):
        """
        Initialise a ParticleState instance with a list of particles.

        Parameters
        ----------
        particles : list
            A list of Particle objects that represent the particles
            in the particle filter.
        """
        self.particles = particles  # Store the list of particles
        self._state_vector = self.state_vector  # Initialise the state vector from particles

    @property
    def state_vector(self) -> np.ndarray:
        """
        Returns the concatenated state vector of all particles.

        This property creates a 2D array (column vector) that includes
        the state vectors of each particle, arranged side by side.

        Returns
        -------
        np.ndarray
            A 2D array containing the concatenated state vectors of
            all particles.
        """
        return np.hstack([particle.state_vector for particle in self.particles])

    @property
    def weights(self) -> np.ndarray:
        """
        Returns the weights of all particles.

        This property extracts the weights associated with each
        particle and returns them as a 1D NumPy array.

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

        This property computes the weighted average of the particle
        states using the particle weights.

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

        This property counts the total number of particles present in
        the particle state.

        Returns
        -------
        int
            The number of particles in the particle state.
        """
        return len(self.particles)
