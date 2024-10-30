import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

from ELPF.detection import Clutter, TrueDetection

sns.set_style("whitegrid")


def plot(track, truth, all_measurements, mapping, save=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    num_steps = len(track)

    min_x = min([state.state_vector[0] for state in truth]) - 10
    max_x = max([state.state_vector[0] for state in truth]) + 10
    min_y = min([state.state_vector[2] for state in truth]) - 10
    max_y = max([state.state_vector[2] for state in truth]) + 10

    legend_elements = [
        plt.Line2D([0], [0], color="C3", linestyle="--", label="Ground Truth"),
        plt.Line2D([0], [0], color="C0", marker="o", linestyle="None", label="Measurements"),
        plt.Line2D([0], [0], color="C8", marker="^", linestyle="None", label="Clutter"),
        plt.Line2D([0], [0], color="C2", marker="o", linestyle="None", label="Particles"),
        plt.Line2D([0], [0], color="C2", label="Estimated Track"),
    ]

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

        # Plot the measurements
        measurement_data = []
        for measurement_set in all_measurements[:k]:
            for measurement in measurement_set:
                if measurement.measurement_model is not None:
                    m = measurement.measurement_model.inverse_function(measurement)
                    if isinstance(measurement, TrueDetection):
                        measurement_data.append(m)

        ax.scatter(
            [measurement[0] for measurement in measurement_data],
            [measurement[1] for measurement in measurement_data],
            color="C0",
            label="Measurements",
        )

        # Plot the clutter
        clutter_data = []
        for measurement_set in all_measurements[:k]:
            for measurement in measurement_set:
                if isinstance(measurement, Clutter):
                    clutter_data.append(
                        measurement.measurement_model.inverse_function(measurement)
                    )

        ax.scatter(
            [clutter[0] for clutter in clutter_data],
            [clutter[1] for clutter in clutter_data],
            color="C8",
            marker="^",
            alpha=0.4,
            label="Clutter",
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
            [state.mean[0] for state in track[: k + 1]],
            [state.mean[2] for state in track[: k + 1]],
            color="C2",
            label="Estimated Track",
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        return ax

    ani = FuncAnimation(fig, update, frames=num_steps, repeat=True)
    if save:
        ani.save("pf.mp4", writer="ffmpeg", fps=5)
    else:
        plt.show()
