from datetime import datetime, timedelta

import numpy as np
from scipy.stats import multivariate_t, uniform
from tqdm import tqdm

from ELPF.array_type import StateVector
from ELPF.detection import Clutter, TrueDetection
from ELPF.filter import ExpectedLikelihoodParticleFilter
from ELPF.hypothesise import PDAHypothesiser
from ELPF.measurement import CartesianToRangeBearingMeasurementModel
from ELPF.plotting import AnimatedPlot
from ELPF.state import GroundTruthPath, GroundTruthState, Particle, ParticleState, State, Track
from ELPF.transition import CombinedLinearGaussianTransitionModel, ConstantVelocity

if __name__ == "__main__":
    # Set random seed
    np.random.seed(1999)

    # Define the mapping between the state vector and the measurement space
    mapping = (0, 2)

    # Create the transition model
    process_noise = 0.001
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(process_noise), ConstantVelocity(process_noise)]
    )

    # Create the measurement model
    measurement_noise = np.diag([1, np.deg2rad(0.2)])  # Noise covariance for range and bearing
    translation_offset = np.array([[0], [0]])
    measurement_model = CartesianToRangeBearingMeasurementModel(
        measurement_noise, mapping, translation_offset
    )

    # Number of steps
    num_steps = 200

    # Start time
    start_time = datetime.now().replace(microsecond=0)
    time_interval = timedelta(seconds=1)
    timesteps = [start_time]

    # Generate ground truth
    truth = GroundTruthPath([GroundTruthState([150, -1, 300, -1], timestamp=start_time)])
    for i in range(1, num_steps):
        timesteps.append(start_time + i * time_interval)
        truth.append(
            GroundTruthState(
                transition_model.function(truth[-1], time_interval),
                timestamp=timesteps[-1],
            )
        )

    # Clutter parameters
    clutter_rate = 2
    x_min = min(state.state_vector[0, 0] for state in truth)
    x_max = max(state.state_vector[0, 0] for state in truth)
    y_min = min(state.state_vector[2, 0] for state in truth)
    y_max = max(state.state_vector[2, 0] for state in truth)
    surveillance_area = (x_max - x_min) * (y_max - y_min)
    clutter_spatial_density = clutter_rate / surveillance_area

    prob_detect = 0.95  # 95% chance of detection

    # Generate measurements
    all_measurements = []
    for state in truth:
        measurement_set = set()

        # Generate detection with probability prob_detect
        if np.random.rand() <= prob_detect:
            measurement = StateVector(measurement_model.function(state, noise=True))
            measurement_set.add(
                TrueDetection(
                    state_vector=measurement,
                    measurement_model=measurement_model,
                    timestamp=state.timestamp,
                )
            )

        # Generate clutter with Poisson number of clutter points
        truth_x = state.state_vector[mapping[0]]
        truth_y = state.state_vector[mapping[1]]
        for _ in range(np.random.poisson(clutter_rate)):
            x = uniform.rvs(x_min, x_max - x_min)
            y = uniform.rvs(y_min, y_max - y_min)
            clutter = StateVector(
                measurement_model.function(
                    State(StateVector([x, 0, y, 0]), state.timestamp), noise=False
                )
            )
            measurement_set.add(
                Clutter(
                    clutter,
                    measurement_model=measurement_model,
                    timestamp=state.timestamp,
                )
            )

        all_measurements.append(measurement_set)

    # Define number of particles
    num_particles = 1000

    # Create prior state
    samples = np.random.multivariate_normal(
        mean=truth[0].state_vector.flatten(), cov=np.diag([1.5, 0.5, 1.5, 0.5]), size=num_particles
    )
    weights = np.ones(num_particles) / num_particles
    particles = np.array([Particle(sample, weight) for sample, weight in zip(samples, weights)])
    prior = ParticleState(particles, timestamp=start_time)

    likelihood_func = multivariate_t.pdf
    likelihood_func_kwargs = {"shape": measurement_model.covar, "df": measurement_model.covar.ndim}

    # Create the ELPF
    pf = ExpectedLikelihoodParticleFilter(transition_model, measurement_model)

    # Create the hypothesiser
    hypothesiser = PDAHypothesiser(
        measurement_model,
        prob_detect,
        clutter_spatial_density,
        likelihood_func,
        likelihood_func_kwargs,
        gate_probability=0.99,
        include_all=False,
    )

    # Create a track to store the state estimates
    track = Track([prior])

    # Perform the particle filtering
    for measurements in tqdm(all_measurements, desc="Filtering"):
        # Predict the new state
        prior = pf.predict(track[-1], time_interval)

        # Update the state
        hypotheses = hypothesiser.hypothesise(prior, measurements)
        posterior = pf.update(prior, hypotheses)

        track.append(posterior)

    # Plot the results
    plotter = AnimatedPlot(timesteps, tail_length=1)
    plotter.plot_truths([truth], mapping=mapping)
    plotter.plot_measurements(all_measurements)
    plotter.plot_tracks([track], mapping=mapping, plot_particles=True)
    plotter.show()
