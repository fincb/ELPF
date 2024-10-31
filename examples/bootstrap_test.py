import numpy as np
from tqdm import tqdm

from ELPF.detection import TrueDetection
from ELPF.likelihood import gaussian_pdf
from ELPF.measurement import CartesianToRangeBearingMeasurementModel
from ELPF.particle_filter import BootstrapParticleFilter
from ELPF.plotting import plot
from ELPF.state import Particle, ParticleState, State
from ELPF.transition import CombinedLinearGaussianTransitionModel, ConstantVelocity

if __name__ == "__main__":
    # Set random seed
    np.random.seed(1999)

    # Define the mapping between the state vector and the measurement space
    mapping = (0, 2)

    # Create a transition model
    time_interval = 1
    process_noise = 0.005
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(process_noise), ConstantVelocity(process_noise)]
    )

    # Create a measurement model
    measurement_noise = np.diag([1, np.deg2rad(0.2)])
    measurement_model = CartesianToRangeBearingMeasurementModel(measurement_noise, mapping)

    # Number of steps
    num_steps = 100

    # Generate ground truth
    truth = [State(np.array([100, 1, 100, 1]))]
    for _ in range(1, num_steps):
        state = transition_model.function(truth[-1], time_interval)
        truth.append(State(state))

    # Generate measurements
    measurements = [
        TrueDetection(measurement_model.function(state, noise=True), measurement_model)
        for state in truth
    ]

    # Define number of particles
    num_particles = 1000

    # Create a prior state
    samples = np.random.multivariate_normal(
        mean=[100, 1, 100, 1], cov=np.diag([1.5, 0.5, 1.5, 0.5]), size=num_particles
    )

    weights = np.ones(num_particles) / num_particles
    particles = np.array([Particle(sample, weight) for sample, weight in zip(samples, weights)])
    prior = ParticleState(particles)

    # Create a particle filter
    pf = BootstrapParticleFilter(transition_model, measurement_model, gaussian_pdf)

    track = [prior]

    for measurement in tqdm(measurements, desc="Filtering"):
        # Predict the new state
        prior = pf.predict(track[-1], time_interval)

        # Update the state
        posterior = pf.update(prior, measurement)

        track.append(posterior)

    measurements = [[m] for m in measurements]

    plot(track, truth, measurements, mapping)
