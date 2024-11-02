from datetime import timedelta

import numpy as np
from scipy.linalg import block_diag

from ELPF.state import State


class ConstantVelocity:
    """
    Constant Velocity Model for linear state transitions with Gaussian process noise.

    This class generates the state transition matrix and process noise covariance
    matrix based on a constant velocity model, with configurable process noise variance.
    """

    def __init__(self, process_noise_variance: float):
        """
        Initialise the Constant Velocity model.

        Parameters
        ----------
        process_noise_variance : float
            The variance of the process noise, used to model uncertainty in the state transitions.
        """
        self.process_noise_variance = process_noise_variance

    def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the state transition matrix (F) and the process noise covariance matrix (Q).

        The matrices represent a 1D constant velocity model for a given time step.

        Parameters
        ----------
        dt : float
            The time step for the transition model.

        Returns
        -------
        tuple
            A tuple containing:
            - F : np.ndarray
                The state transition matrix, modeling how the state evolves.
            - Q : np.ndarray
                The process noise covariance matrix, scaled by process noise variance.
        """
        # State transition matrix (for constant velocity in one dimension)
        F = np.array([[1, dt], [0, 1]])

        # Process noise covariance matrix, scaled by the process noise variance
        Q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * self.process_noise_variance

        return F, Q


class CombinedLinearGaussianTransitionModel:
    """
    Combined Linear Gaussian Transition Model for multidimensional state transitions.
    """

    def __init__(self, transition_models: list[ConstantVelocity]):
        self.transition_models = transition_models
        self.F_combined = None  # Placeholder for the combined F matrix
        self.Q_combined = None  # Placeholder for the combined Q matrix
        self.last_time_interval = None  # Store the last time interval used

    def _update_combined_matrices(self, dt: float):
        """
        Update the combined state transition and process noise matrices.

        Parameters
        ----------
        dt : float
            The time step for the transition.
        """
        # Retrieve transition and noise matrices from each model
        F_list, Q_list = zip(*[model.matrices(dt) for model in self.transition_models])

        # Stack the individual matrices to form combined F and Q matrices
        self.F_combined = block_diag(*F_list)
        self.Q_combined = block_diag(*Q_list)

    def function(self, state: State, time_interval: timedelta, noise: bool = True) -> np.ndarray:
        """
        Predict the next state based on the current state and time step.

        Combines the state transition matrices and process noise covariance matrices
        of the individual models to compute the overall transition in multiple dimensions.

        Parameters
        ----------
        state : State
            The current state of the system, including the state vector.
        time_interval : float
            The time step for the transition.
        noise : bool, optional
            If True, add process noise to the state transition (default is True).

        Returns
        -------
        np.ndarray
            The predicted next state as a column vector, optionally with process noise.
        """
        dt = time_interval.total_seconds()
        # Update combined matrices only if the time interval has changed
        if dt != self.last_time_interval or self.F_combined is None or self.Q_combined is None:
            self._update_combined_matrices(dt)
            self.last_time_interval = time_interval

        # Calculate next state with optional noise
        if noise:
            # Generate noise with the same number of samples as state_vector's second dimension
            noise_vector = np.random.multivariate_normal(
                mean=np.zeros(self.F_combined.shape[0]),
                cov=self.Q_combined,
                size=state.state_vector.shape[1],
            ).T
            return self.F_combined @ state.state_vector + noise_vector
        else:
            return self.F_combined @ state.state_vector
