import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

from angle import Bearing

sns.set_style("whitegrid")


class State:
    def __init__(self, state_vector: np.ndarray):
        """
        Initialize a State instance with a given state vector.

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
        state_vector = state.state_vector[self.mapping,]
        x, y = state_vector[0, 0], state_vector[1, 0]

        # Calculate range (rho) and bearing (phi)
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        # Ensure phi is wrapped in a Bearing class
        phi = [Bearing(phi)] if np.isscalar(phi) else [Bearing(i) for i in phi]

        # Add noise if specified
        if noise:
            noise = np.random.multivariate_normal(np.zeros(2), self.R).reshape(-1, 1)
            return np.array([[rho], phi]) + noise
        else:
            return np.array([[rho], phi])

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
    def R(self) -> np.ndarray:
        """
        Measurement noise covariance matrix.

        Returns
        -------
        np.ndarray
            The covariance matrix used for the measurement noise.
        """
        return self.measurement_noise


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


# Set random seed
np.random.seed(1999)

# Define the mapping between the state vector and the measurement space
mapping = (0, 2)

# Create a transition model
dt = 1
process_noise = 0.05
transition_model = ConstantVelocityTransitionModel(dt, process_noise)

# Create a measurement model
measurement_noise = np.diag([1, 0.01])  # Diagonal noise covariance for range and bearing
measurement_model = CartesianToRangeBearingMeasurementModel(measurement_noise, mapping)

# Number of steps
num_steps = 30

# Generate ground truth
truth = [State(np.array([0, 1, 0, 1]))]
for _ in range(1, num_steps):
    state = transition_model.function(truth[-1])
    truth.append(State(state))

# Generate measurements
measurements = [measurement_model.function(state, noise=True) for state in truth]

# Define number of particles
num_particles = 1000

# Create a prior state
samples = np.random.multivariate_normal(
    mean=[0, 1, 0, 1], cov=np.diag([1.5, 0.5, 1.5, 0.5]), size=num_particles
)

weights = np.ones(num_particles) / num_particles
particles = np.array([Particle(sample, weight) for sample, weight in zip(samples, weights)])
prior = ParticleState(particles)

# Create a particle filter
pf = BootstrapParticleFilter(transition_model, measurement_model)

track = [prior]

for measurement in measurements:
    # Predict the new state
    prior = pf.predict(track[-1])

    # Update the state
    posterior = pf.update(prior, measurement)

    track.append(posterior)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))


def update(k):
    ax.clear()
    # Plot the ground truth
    ax.plot(
        [state.state_vector[0] for state in truth[:k]],
        [state.state_vector[2] for state in truth[:k]],
        color="C3",
        linestyle="--",
        label="Ground Truth",
    )

    # Plot the particles
    ax.scatter(
        track[k].state_vector[mapping[0], :],
        track[k].state_vector[mapping[1], :],
        color="C2",
        label="Particles",
        s=track[k].weights * 1000,
        alpha=0.5,
    )

    # Plot the estimated track up to the current time step
    ax.plot(
        [state.mean[0] for state in track[:k]],
        [state.mean[2] for state in track[:k]],
        color="C2",
        label="Estimated Track",
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.set_xlim(
        min([state.state_vector[0] for state in truth]) - 10,
        max([state.state_vector[0] for state in truth]) + 10,
    )
    ax.set_ylim(
        min([state.state_vector[2] for state in truth]) - 10,
        max([state.state_vector[2] for state in truth]) + 10,
    )

    return ax


ani = FuncAnimation(fig, update, frames=num_steps, repeat=False)
plt.show()
