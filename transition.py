import numpy as np

from state import State


class ConstantVelocityTransitionModel:
    def __init__(self, dt: float, process_noise: float):
        """
        Initializes the Constant Velocity Transition Model.

        Parameters
        ----------
        dt : float
            The time step for the transition model.
        process_noise : float
            The standard deviation of the process noise, used to model
            uncertainty in the state transitions.
        """
        self.dt = dt  # Time step
        self.process_noise = process_noise  # Process noise standard deviation

    def function(self, state: State) -> np.ndarray:
        """
        Computes the next state based on the current state and process noise.

        The next state is calculated using the state transition matrix
        (F) and adding process noise.

        Parameters
        ----------
        state : State
            The current state of the system, which includes the state vector.

        Returns
        -------
        np.ndarray
            The predicted next state as a column vector, including
            process noise.
        """
        noise = np.random.multivariate_normal(np.zeros(4), self.Q).reshape(-1, 1)
        return self.F @ state.state_vector + noise

    @property
    def F(self) -> np.ndarray:
        """
        State transition matrix.

        The transition matrix describes how the state evolves over one
        time step given constant velocity.

        Returns
        -------
        np.ndarray
            A 4x4 state transition matrix.
        """
        return np.array([[1, self.dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])

    @property
    def Q(self) -> np.ndarray:
        """
        Process noise covariance matrix.

        The covariance matrix models the process noise for the state
        transitions, taking into account the time step and process noise.

        Returns
        -------
        np.ndarray
            A 4x4 covariance matrix for the process noise.
        """
        return (
            np.array(
                [
                    [self.dt**3 / 3, self.dt**2 / 2, 0, 0],
                    [self.dt**2 / 2, self.dt, 0, 0],
                    [0, 0, self.dt**3 / 3, self.dt**2 / 2],
                    [0, 0, self.dt**2 / 2, self.dt],
                ]
            )
            * self.process_noise**2
        )
