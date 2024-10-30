import numpy as np
from scipy.stats import uniform
from tqdm import tqdm

from detection import Clutter, TrueDetection
from measurement import CartesianToRangeBearingMeasurementModel
from particle_filter import ExpectedLikelihoodParticleFilter
from plotting import plot
from state import Particle, ParticleState, State
from transition import ConstantVelocityTransitionModel

if __name__ == "__main__":
    # Set random seed
    np.random.seed(1999)

    # Define the mapping between the state vector and the measurement space
    mapping = (0, 2)

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

    # Create a particle filter
    pf = ExpectedLikelihoodParticleFilter(transition_model, measurement_model)

    track = [prior]

    # Perform the particle filter
    for measurements in tqdm(all_measurements, desc="Filtering"):
        # Predict the new state
        prior = pf.predict(track[-1])

        # Update the state
        posterior = pf.update(prior, measurements)

        track.append(posterior)

    # Plot the results
    plot(track, truth, all_measurements, mapping, save=False)
