from datetime import datetime, timedelta

import numpy as np
from scipy.stats import multivariate_t, uniform
from tqdm import tqdm

from ELPF.array_type import StateVector
from ELPF.detection import Clutter, MissedDetection, TrueDetection
from ELPF.filter import ExpectedLikelihoodParticleFilter
from ELPF.hypothesise import PDAHypothesiser
from ELPF.initiate_delete import CovarianceBasedDeleter, GaussianParticleInitiator
from ELPF.measurement import CartesianToRangeBearingMeasurementModel
from ELPF.plotting import AnimatedPlot
from ELPF.state import GroundTruthPath, GroundTruthState, State
from ELPF.transition import CombinedLinearGaussianTransitionModel, ConstantVelocity

if __name__ == "__main__":
    # Set random seed
    np.random.seed(1991)

    # Define the mapping between the state vector and the measurement space
    mapping = (0, 2)

    # Create the ground truth transition model
    process_noise = 0.001
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(process_noise), ConstantVelocity(process_noise)]
    )

    # Create the measurement model
    measurement_noise = np.diag([0.1, np.deg2rad(0.02)])  # Noise covariance for range and bearing
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

    # Ground truth path creation
    truths = set()  # Truths across all time
    current_truths = set()  # Truths alive at current time

    # Birth and death probabilities
    death_probability = 0.01
    birth_probability = 0.2

    # Generate ground truth over time
    for k in range(num_steps):
        # Update time
        timesteps.append(start_time + timedelta(seconds=k))

        # Death
        if current_truths and np.random.rand() <= death_probability:  # Death probability
            current_truths.clear()  # Remove the single truth

        # Update truth if it exists
        if current_truths:
            truth = next(iter(current_truths))  # Get the single truth
            truth.append(
                GroundTruthState(
                    transition_model.function(
                        truth[-1], noise=True, time_interval=timedelta(seconds=1)
                    ),
                    timestamp=timesteps[-1],
                )
            )

        # Birth
        if (
            not current_truths and np.random.rand() <= birth_probability
        ):  # Only birth if no current truth
            x, y = np.random.rand(2) * [20, 20]  # Range [0, 20] for x and y
            x_vel, y_vel = (np.random.rand(2)) * 2 - 1  # Range [-1, 1] for x and y velocity
            state = GroundTruthState([x, x_vel, y, y_vel], timestamp=timesteps[-1])

            # Add to truth set for current and for all timestamps
            truth = GroundTruthPath([state])
            current_truths.add(truth)
            truths.add(truth)

    # Clutter parameters
    clutter_rate = 3
    x_min = min(state.state_vector[0, 0] for truth in truths for state in truth)
    x_max = max(state.state_vector[0, 0] for truth in truths for state in truth)
    y_min = min(state.state_vector[2, 0] for truth in truths for state in truth)
    y_max = max(state.state_vector[2, 0] for truth in truths for state in truth)
    surveillance_area = (x_max - x_min) * (y_max - y_min)
    clutter_spatial_density = clutter_rate / surveillance_area

    prob_detect = 0.95  # 95% chance of detection

    # Generate the measurements
    all_measurements = []

    for k in range(num_steps):
        measurement_set = set()
        timestamp = timesteps[k]

        for truth in truths:
            try:
                truth_state = truth[timestamp]
            except IndexError:
                # This truth not alive at this time.
                continue
            # Generate actual detection from the state with a `prob_detect` chance of no detection.
            if np.random.rand() <= prob_detect:
                # Generate actual detection from the state
                measurement = measurement_model.function(truth_state, noise=True)
                measurement_set.add(
                    TrueDetection(
                        state_vector=measurement,
                        timestamp=timestamp,
                        measurement_model=measurement_model,
                    )
                )

        # Generate clutter with a Poisson-distributed number of clutter points
        clutter_count = np.random.poisson(clutter_rate)
        clutter_x = uniform.rvs(x_min, x_max - x_min, size=clutter_count)
        clutter_y = uniform.rvs(y_min, y_max - y_min, size=clutter_count)

        for x, y in zip(clutter_x, clutter_y):
            clutter = StateVector(
                measurement_model.function(
                    State(StateVector([x, 0, y, 0]), timestamp), noise=False
                )
            )
            measurement_set.add(
                Clutter(clutter, measurement_model=measurement_model, timestamp=timestamp)
            )

        all_measurements.append(measurement_set)

    # Create the particle filter transition model
    process_noise = 0.1
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(process_noise), ConstantVelocity(process_noise)]
    )

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
    )

    # Create the deleter
    deleter = CovarianceBasedDeleter(3)

    # Create the initiator
    initiator = GaussianParticleInitiator(1000)

    all_tracks = set()  # All tracks across all time
    current_tracks = set()  # Tracks alive at current time

    confirmation_age = 10  # Confirmation age for tracks

    # Perform the particle filtering
    for n, measurements in tqdm(enumerate(all_measurements), desc="Filtering", total=num_steps):
        associated_measurements = set()
        for track in current_tracks:
            # Predict the new state
            prior = pf.predict(track[-1], time_interval)

            # Update the state
            hypotheses = hypothesiser.hypothesise(prior, measurements)
            posterior = pf.update(prior, hypotheses)

            for hypothesis in hypotheses:
                if not isinstance(hypothesis.measurement, MissedDetection):
                    associated_measurements.add(hypothesis.measurement)

            track.append(posterior)

            track.age += 1

        # Carry out deletion and initiation
        current_tracks -= deleter.delete(current_tracks)
        current_tracks |= initiator.initiate(
            measurements - associated_measurements, start_time + timedelta(seconds=n)
        )
        all_tracks |= current_tracks

    # Remove tracks that are too young
    confirmed_tracks = {track for track in all_tracks if track.age >= confirmation_age}

    # Plot the results
    plotter = AnimatedPlot(timesteps, tail_length=1)
    plotter.plot_truths(truths, mapping=mapping)
    plotter.plot_measurements(all_measurements)
    plotter.plot_tracks(confirmed_tracks, mapping=mapping, plot_particles=True)
    plotter.show()
