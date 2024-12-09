"""
This module defines classes for representing states in a state estimation framework,
including ground truth states, particles, and collections of particles. The classes are designed
to facilitate the management and manipulation of state information in particle filtering.

This module is inspired by the Stone Soup library, which is an open-source framework
for state estimation and tracking.

Classes:
    State: A base class for representing the state of a system, including a state vector
           and an optional timestamp.
    GroundTruthState: A subclass of State that represents the true state of the system
                      at a specific timestamp, ensuring that the state vector is a
                      StateVector.
    Particle: A subclass of State that represents a single particle in a particle filter,
              storing a state vector and an associated weight that indicates its importance.
    ParticleState: A subclass of State that represents the collective state of multiple
                   particles, providing methods to access the concatenated state vector,
                   weights, mean state, and the number of particles.

Usage:
    The State class serves as a foundation for other state representations. The GroundTruthState
    class is used to track the true state of the system, while the Particle class is essential
    for the particle filter algorithm. The ParticleState class encapsulates multiple particles,
    facilitating operations that require aggregate information.

References:
    Stone Soup Library: https://stonesoup.readthedocs.io/
"""

from datetime import datetime
import numpy as np
from collections.abc import MutableSequence
from typing import Any
from ELPF.array_type import StateVector, StateVectors


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


class GroundTruthPath:
    """
    Path class for representing the path of an object.

    Parameters
    ----------
    states: list
        The list of states in the path
    """

    def __init__(self, states=None):
        if states is None:
            states = []
        if not isinstance(states, MutableSequence):
            states = [states]
        self.states = states

    def __getitem__(self, item):
        # If item is a datetime, return the corresponding state
        if isinstance(item, datetime):
            for state in self.states:
                if state.timestamp == item:
                    return state
            raise IndexError('Timestamp not found in states.')
        return self.states[item]
    
    def append(self, value):
        self.states.append(value)

    def __len__(self):
        return len(self.states)

    def __str__(self):
        return f"GroundTruthPath with {len(self.states)} states"

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
