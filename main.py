import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

from measurement import CartesianToRangeBearingMeasurementModel
from particle_filter import BootstrapParticleFilter
from state import Particle, ParticleState, State
from transition import ConstantVelocityTransitionModel

sns.set_style("whitegrid")


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

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(k):
        ax.clear()
        # Plot the ground truth up to the current time step
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
