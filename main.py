import numpy as np

from measurement import CartesianToRangeBearingMeasurementModel
from particle_filter import BootstrapParticleFilter
from plotting import plot
from state import Particle, ParticleState, State
from transition import ConstantVelocityTransitionModel

if __name__ == "__main__":
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

    inv_measurements = [
        measurement_model.inverse_function(State(measurement)) for measurement in measurements
    ]

    # Plot the results
    plot(track, truth, inv_measurements, mapping, save=False)
