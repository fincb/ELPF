import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.animation import FuncAnimation
from plotly import colors
from tqdm import tqdm

from ELPF.detection import Clutter, TrueDetection

sns.set_style("darkgrid")


def plot(track, truth, all_measurements, mapping, save=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    num_steps = len(track)

    # Gather all x and y coordinates across truth, measurements, and clutter for dynamic limits
    x_positions = [state.state_vector[0] for state in truth]
    y_positions = [state.state_vector[2] for state in truth]

    # Store measurements and clutter data for all steps
    measurement_data = []
    clutter_data = []

    for k, measurement_set in enumerate(all_measurements):
        measurement_set_data = []
        for measurement in measurement_set:
            m = measurement.measurement_model.inverse_function(measurement)
            if isinstance(measurement.state_vector, TrueDetection):
                measurement_set_data.append(m)
            elif isinstance(measurement, Clutter):
                clutter_data.append(m)

        # Append measurement data for the current step
        measurement_data.append(measurement_set_data)

    # Compute ranges and dynamic buffer
    for m in measurement_data:
        x_positions.extend(m[0] for m in m)

    for c in clutter_data:
        x_positions.append(c[0])
        y_positions.append(c[1])

    # Compute ranges and dynamic buffer
    x_range = max(x_positions) - min(x_positions)
    y_range = max(y_positions) - min(y_positions)
    buffer_ratio = 0.1  # 10% of the range as buffer
    x_buffer = x_range * buffer_ratio
    y_buffer = y_range * buffer_ratio

    # Apply dynamic buffer to the axis limits
    min_x, max_x = min(x_positions) - x_buffer, max(x_positions) + x_buffer
    min_y, max_y = min(y_positions) - y_buffer, max(y_positions) + y_buffer

    # Prepare legend elements for static display
    legend_elements = [
        plt.Line2D([0], [0], color="C3", linestyle="--", label="Ground Truth"),
        plt.Line2D([0], [0], color="C0", marker="o", linestyle="None", label="Measurements"),
        plt.Line2D([0], [0], color="C8", marker="^", linestyle="None", label="Clutter"),
        plt.Line2D([0], [0], color="C2", marker="o", linestyle="None", label="Particles"),
        plt.Line2D([0], [0], color="C2", label="Estimated Track"),
    ]

    pbar = tqdm(total=num_steps, desc="Animating", unit=" frames")

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

        # Plot measurements and clutter
        measurement_data, clutter_data = [], []
        for measurement_set in all_measurements[:k]:
            for measurement in measurement_set:
                m = measurement.measurement_model.inverse_function(measurement)
                if isinstance(measurement, TrueDetection):
                    measurement_data.append(m)
                elif isinstance(measurement, Clutter):
                    clutter_data.append(m)

        if clutter_data:  # Ensure there are clutter points to plot
            ax.scatter(
                [c[0] for c in clutter_data],
                [c[1] for c in clutter_data],
                color="C8",
                marker="^",
                s=10,
                alpha=0.4,
                label="Clutter",
            )

        # Plot the measurements
        if measurement_data:  # Ensure there are measurements to plot
            ax.scatter(
                [m[0] for m in measurement_data],
                [m[1] for m in measurement_data],
                color="C0",
                s=10,
                label="Measurements",
            )

        # Plot particles
        ax.scatter(
            track[k].state_vector[mapping[0], :],
            track[k].state_vector[mapping[1], :],
            color="C2",
            label="Particles",
            s=track[k].weights * 1000,
            alpha=0.5,
        )

        # Plot estimated track
        ax.plot(
            [state.mean[0] for state in track[: k + 1]],
            [state.mean[2] for state in track[: k + 1]],
            color="C2",
            label="Estimated Track",
        )

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        pbar.update(1)

        return ax

    ani = FuncAnimation(fig, update, frames=num_steps, repeat=True)
    if save:
        ani.save("pf.mp4", writer="ffmpeg", fps=5)
    else:
        plt.show()


# class AnimatedPlot:

#     def __init__(self, timesteps, tail_length=1, sim_duration=6):
#         self.fig = go.Figure()
#         self.timesteps = timesteps
#         self.tail_length = tail_length
#         self.sim_duration = sim_duration

#         self.fig.frames = [dict(name=str(time), data=[], traces=[]) for time in timesteps]

#         frame_duration = self.sim_duration * 1000 / len(self.timesteps)

#         start_cutoff = 11
#         end_cutoff = None

#         self.colourway = colors.qualitative.Plotly[1:]

#         self.time_window = (timesteps[-1] - timesteps[0]) * tail_length

#         menus = [
#             {
#                 "type": "buttons",
#                 "buttons": [
#                     {
#                         "args": [
#                             None,
#                             {
#                                 "frame": {"duration": frame_duration, "redraw": True},
#                                 "fromcurrent": True,
#                                 "transition": {"duration": 0},
#                             },
#                         ],
#                         "label": "Play",
#                         "method": "animate",
#                     },
#                     {
#                         "args": [
#                             [None],
#                             {
#                                 "frame": {"duration": 0, "redraw": True},
#                                 "mode": "immediate",
#                                 "transition": {"duration": 0},
#                             },
#                         ],
#                         "label": "Pause",
#                         "method": "animate",
#                     },
#                 ],
#                 "direction": "left",
#                 "pad": {"r": 10, "t": 75},
#                 "showactive": True,
#                 "x": 0.1,
#                 "y": 0,
#                 "xanchor": "right",
#                 "yanchor": "top",
#             }
#         ]

#         sliders = [
#             {
#                 "yanchor": "top",
#                 "xanchor": "left",
#                 "currentvalue": {
#                     "font": {"size": 16},
#                     "prefix": "Time:",
#                     "visible": True,
#                     "xanchor": "right",
#                 },
#                 "transition": {"duration": frame_duration, "easing": "linear"},
#                 "pad": {"b": 10, "t": 50},
#                 "len": 0.9,
#                 "x": 0.1,
#                 "y": 0,
#                 "steps": [
#                     {
#                         "args": [
#                             [frame.name],
#                             {
#                                 "frame": {"duration": 1.0, "easing": "linear", "redraw": True},
#                                 "transition": {"duration": 0, "easing": "linear"},
#                             },
#                         ],
#                         "label": frame.name[start_cutoff:end_cutoff],
#                         "method": "animate",
#                     }
#                     for frame in self.fig.frames
#                 ],
#             }
#         ]

#         self.fig.update_layout(updatemenus=menus, sliders=sliders)
#         self.fig.update_xaxes(title_text="X Position (m)")
#         self.fig.update_yaxes(title_text="Y Position (m)")

#     def plot_truth(self, truths, mapping, resize=True):
#         pass


# ani = AnimatedPlot(timesteps=range(300), tail_length=1, sim_duration=6)
# ani.fig.show()
