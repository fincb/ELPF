import numpy as np

from ELPF.angle import Bearing
from ELPF.state import State


class CartesianToRangeBearingMeasurementModel:
    def __init__(self, measurement_noise: np.ndarray, mapping: tuple):
        """
        Initializes the Cartesian to Range-Bearing Measurement Model.

        Parameters
        ----------
        measurement_noise : np.ndarray
            The measurement noise covariance matrix used to model
            uncertainty in the measurements.
        mapping : tuple
            A tuple indicating which elements of the state vector to use
            for the measurement (e.g., positions in the state vector).
        """
        self.measurement_noise = measurement_noise  # Measurement noise covariance
        self.mapping = mapping  # Mapping to the measurement space

    def function(self, state: State, noise: bool = True) -> np.ndarray:
        """
        Converts the state from Cartesian coordinates to range and bearing.

        Parameters
        ----------
        state : State
            The current state of the system, which includes the state vector.
        noise : bool, optional
            If True, adds measurement noise to the output (default is True).

        Returns
        -------
        np.ndarray
            The measurement in range and bearing as a column vector,
            optionally including measurement noise.
        """
        state_vector = state.state_vector[self.mapping, :]
        x, y = state_vector[0, :], state_vector[1, :]

        # Calculate range (rho) and bearing (phi)
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        # Ensure phi is wrapped in a Bearing class
        if np.isscalar(phi):
            phi = np.array([[Bearing(phi)]])
        else:
            phi = np.array([[Bearing(i) for i in phi]])  # Ensure phi is a 2D array

        # Stack rho and phi into a single array
        measurement = np.vstack((rho, phi.flatten()))  # Use flatten to ensure it's 1D

        # Add noise if specified
        if noise:
            noise = np.random.multivariate_normal(np.zeros(2), self.covar).reshape(-1, 1)
            measurement += noise

        return measurement

    def inverse_function(self, state: State) -> np.ndarray:
        """
        Converts the measurement in range and bearing back to Cartesian coordinates.

        Parameters
        ----------
        state : State
            The measurement state containing range and bearing.

        Returns
        -------
        np.ndarray
            The corresponding Cartesian coordinates as a column vector.
        """
        state_vector = state.state_vector
        rho, phi = state_vector[0, 0], state_vector[1, 0]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.array([[x], [y]])

    @property
    def covar(self) -> np.ndarray:
        """
        Measurement noise covariance matrix.

        Returns
        -------
        np.ndarray
            The covariance matrix used for the measurement noise.
        """
        return self.measurement_noise
