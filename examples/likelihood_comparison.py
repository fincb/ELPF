import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
from tqdm import tqdm

from ELPF.detection import Clutter, TrueDetection
from ELPF.likelihood import gaussian_pdf, t_pdf
from ELPF.measurement import CartesianToRangeBearingMeasurementModel
from ELPF.particle_filter import ExpectedLikelihoodParticleFilter
from ELPF.state import Particle, ParticleState, State
from ELPF.transition import ConstantVelocityTransitionModel


def compute_error(particle_state: ParticleState, true_state: State, mapping: tuple) -> float:
    """
    Compute the error between the estimated state and the true state.

    Parameters
    ----------
    estimated_state : ParticleState
        The estimated state from the particle filter.
    true_state : State
        The true state at the current time step.
    mapping : tuple
        The mapping of state indices to measurement indices.

    Returns
    -------
    float
        The computed error (Euclidean distance).
    """
    # Extract the position from the estimated state (mean of the particles)
    estimated_position = particle_state.mean[mapping]
    true_position = true_state.state_vector[mapping]

    # Calculate Euclidean distance
    error = np.linalg.norm(estimated_position - true_position)
    return error


if __name__ == "__main__":
    # Set random seed
    np.random.seed(1999)

    # Define the mapping between the state vector and the measurement space
    mapping = [0, 2]

    # Create the transition model
    dt = 1
    process_noise = 0.05
    transition_model = ConstantVelocityTransitionModel(dt, process_noise)

    # Create the measurement model
    measurement_noise = np.diag([1, np.deg2rad(0.2)])  # Noise covariance for range and bearing
    measurement_model = CartesianToRangeBearingMeasurementModel(measurement_noise, mapping)

    # Number of steps
    num_steps = 100

    # Generate ground truth
    truth = [State(np.array([0, 1, 0, 1]))]
    for _ in range(1, num_steps):
        state = transition_model.function(truth[-1])
        truth.append(State(state))

    prob_detect = 0.8  # 80% chance of detection

    # Generate measurements
    all_measurements = []
    for state in truth:
        measurement_set = set()

        # Generate detection
        if np.random.rand() <= prob_detect:
            measurement = measurement_model.function(state, noise=True)
            measurement_set.add(
                TrueDetection(state_vector=measurement, measurement_model=measurement_model)
            )

        # Generate clutter
        truth_x = state.state_vector[mapping[0]]
        truth_y = state.state_vector[mapping[1]]
        for _ in range(np.random.poisson(5)):
            x = uniform.rvs(truth_x - 10, 40)
            y = uniform.rvs(truth_y - 10, 40)
            clutter = measurement_model.function(State(np.array([x, 0, y, 0])), noise=False)
            measurement_set.add(
                Clutter(
                    clutter,
                    measurement_model=measurement_model,
                )
            )

        all_measurements.append(measurement_set)

    # Define number of particles
    num_particles = 1000

    # Create a prior state
    samples = np.random.multivariate_normal(
        mean=[0, 1, 0, 1], cov=np.diag([1.5, 0.5, 1.5, 0.5]), size=num_particles
    )

    weights = np.ones(num_particles) / num_particles
    particles = np.array([Particle(sample, weight) for sample, weight in zip(samples, weights)])
    prior = ParticleState(particles)

    # Create two particle filters
    pf_t = ExpectedLikelihoodParticleFilter(transition_model, measurement_model, t_pdf)
    pf_gaussian = ExpectedLikelihoodParticleFilter(
        transition_model, measurement_model, gaussian_pdf
    )

    track_t = [prior]
    track_gaussian = [prior]

    error_t = []
    error_gaussian = []

    # Perform the particle filter
    for measurements, true_state in tqdm(zip(all_measurements, truth), total=num_steps):
        # Predict and update for t_pdf
        prior_t = pf_t.predict(track_t[-1])
        posterior_t = pf_t.update(prior_t, measurements)
        current_error_t = compute_error(posterior_t, true_state, mapping)
        error_t.append(current_error_t)
        track_t.append(posterior_t)

        # Predict and update for gaussian_pdf
        prior_gaussian = pf_gaussian.predict(track_gaussian[-1])
        posterior_gaussian = pf_gaussian.update(prior_gaussian, measurements)
        current_error_gaussian = compute_error(posterior_gaussian, true_state, mapping)
        error_gaussian.append(current_error_gaussian)
        track_gaussian.append(posterior_gaussian)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(error_t, label="Error (Student's t)")
    plt.plot(error_gaussian, label="Error (Gaussian)")
    plt.title("Tracking Error Comparison Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error (Distance)")
    plt.legend()
    plt.grid()
    plt.show()
