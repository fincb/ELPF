from datetime import datetime
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mergedeep import merge
from plotly import colors
from tqdm import tqdm

from ELPF.detection import Clutter, TrueDetection

sns.set_style("darkgrid")


def plot_mpl(track, truth, all_measurements, mapping, save=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    num_steps = len(track)

    # Gather all x and y coordinates across truth, measurements, and clutter for dynamic limits
    x_positions = [state.state_vector[0] for state in truth]
    y_positions = [state.state_vector[2] for state in truth]

    # Store measurements and clutter data for all steps
    measurement_data = []
    clutter_data = []

    for measurement_set in all_measurements:
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
        plt.Line2D([0], [0], color="k", marker="o", linestyle="None", label="Sensor"),
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

        # Plot sensor
        ax.scatter([0], [0], color="k", marker="o", label="Sensor")

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


class AnimatedPlot:
    """A class to create an animated plot for visualizing time-series data, such as ground truth,
    measurements, and tracks over a sequence of time steps. The plot includes interactive controls
    for playing, pausing, and navigating through time using sliders.

    This class is inspired by the Stone Soup library, which is an open-source framework
    for state estimation and tracking.

    References:
    Stone Soup Library: https://stonesoup.readthedocs.io/
    """

    def __init__(self, timesteps, tail_length=1, sim_duration=6):
        """
        Initialises the AnimatedPlot instance.

        Parameters
        ----------
        timesteps : list
            A list of timestamps corresponding to the simulation.
        tail_length : float, optional
            The length of the time window to visualize past states, default is 1.
        sim_duration : float, optional
            The total duration of the simulation in seconds, default is 6.
        """
        self.fig = go.Figure()
        self.timesteps = timesteps
        self.tail_length = tail_length
        self.sim_duration = sim_duration

        # Set up frames for animation
        self.fig.frames = [dict(name=str(time), data=[], traces=[]) for time in timesteps]
        frame_duration = self.sim_duration * 1000 / len(self.timesteps)
        self.colorway = colors.qualitative.Plotly[1:]  # Color palette for traces
        self.time_window = (timesteps[-1] - timesteps[0]) * tail_length
        self.plotting_function_called = False

        self._setup_layout(frame_duration)

    def _setup_layout(self, frame_duration):
        """
        Sets up the layout for the figure, including menus and sliders for animation control.

        Parameters
        ----------
        frame_duration : float
            The duration of each frame in milliseconds for the animation.
        """
        # Define playback buttons
        menus = [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 75},
                "showactive": True,
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
            }
        ]

        # Define slider for time navigation
        sliders = [
            {
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time:",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": frame_duration, "easing": "linear"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 1.0, "easing": "linear", "redraw": True},
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                        "label": frame.name[11:],  # Adjusted to slice directly
                        "method": "animate",
                    }
                    for frame in self.fig.frames
                ],
            }
        ]

        # Update figure layout with menus and sliders
        self.fig.update_layout(updatemenus=menus, sliders=sliders)
        self.fig.update_xaxes(title_text="X Position (m)")
        self.fig.update_yaxes(title_text="Y Position (m)")

    def plot_truths(self, truths, mapping, truths_label="Ground Truth", resize=True):
        """
        Plots ground truth states on the figure.

        Parameters
        ----------
        truths : list
            A list of ground truth state sequences.
        mapping : list
            A list of indices mapping state vector components to x and y coordinates.
        truths_label : str, optional
            The label for the ground truth in the legend, default is "Ground Truth".
        resize : bool, optional
            If True, resizes the axes to fit the data, default is True.
        """
        data = self._prepare_data(truths, mapping, data_type="truth")

        # Set the base trace index for all tracks
        trace_base = len(self.fig.data)

        # Create the base trace for all tracks
        truth_kwargs = {
            "x": [],
            "y": [],
            "mode": "lines",
            "line": {"color": self.colorway[0], "dash": "dash"},
            "legendgroup": truths_label,
            "legendrank": 100,
            "name": truths_label,
            "showlegend": True,
        }
        self.fig.add_trace(go.Scatter(truth_kwargs))

        # Initialise traces for every track
        truth_kwargs["showlegend"] = False

        # Update truth colours
        for i, _ in enumerate(truths):
            truth_kwargs["line"] = {"color": self.colorway[i % len(self.colorway)], "dash": "dash"}
            self.fig.add_trace(go.Scatter(truth_kwargs))

        for frame in self.fig.frames:
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            frame_time = datetime.fromisoformat(frame.name)
            cutoff_time = frame_time - self.time_window

            # Add blank data to ensure track legend stays in place
            data_.append(go.Scatter(x=[np.inf, np.inf], y=[np.inf, np.inf]))
            traces_.append(trace_base)

            for i, truth in enumerate(data):
                indices = np.where((truth["time"] <= frame_time) & (truth["time"] >= cutoff_time))

                data_.append(
                    go.Scatter(
                        x=truth["x"][indices],
                        y=truth["y"][indices],
                        meta=truth["time_str"][indices],
                        hovertemplate="GroundTruthState<br>(%{x}, %{y})<br>Time: %{meta}",
                    )
                )

                traces_.append(trace_base + i + 1)

                frame.data = data_
                frame.traces = traces_

        if resize:
            self._resize(data, plot_type="ground_truth")

        self.plotting_function_called = True

    def plot_measurements(
        self,
        measurements,
        convert_measurements=True,
        measurements_label="Measurements",
        resize=True,
    ):
        """
        Plots measurements on the figure.

        Parameters
        ----------
        measurements : list
            A list of measurement sequences, which may include detections and clutter.
        convert_measurements : bool, optional
            If True, converts raw measurements into state vectors using the inverse measurement
            model, default is True.
        measurements_label : str, optional
            The label for the measurements in the legend, default is "Measurements".
        resize : bool, optional
            If True, resizes the axes to fit the data, default is True.
        """
        # Flatten measurements if they are in sets
        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = measurements

        # Convert measurements to detections and clutter
        detections, clutter = self._conv_measurements(measurements_set, convert_measurements)

        # Prepare data for plotting detections
        detection_data = self._prepare_data(detections, [], data_type="measurement")

        # Prepare data for plotting clutter
        clutter_data = self._prepare_data(clutter, [], data_type="measurement")

        trace_base = len(self.fig.data)

        kwargs = {}

        # Initialise detections
        name = measurements_label + "<br>(Detections)"
        measurement_kwargs = dict(
            x=[],
            y=[],
            mode="markers",
            name=name,
            legendgroup=name,
            legendrank=200,
            showlegend=True,
            marker=dict(color="#636EFA"),
            hoverinfo="none",
        )
        merge(measurement_kwargs, kwargs)

        self.fig.add_trace(go.Scatter(measurement_kwargs))  # trace for legend

        measurement_kwargs.update({"showlegend": False})
        self.fig.add_trace(go.Scatter(measurement_kwargs))  # trace for plotting measurements

        # Initialise clutter trace
        name = measurements_label + "<br>(Clutter)"
        clutter_kwargs = dict(
            x=[],
            y=[],
            mode="markers",
            name=name,
            legendgroup=name,
            legendrank=300,
            showlegend=True,
            marker=dict(symbol="star-triangle-up", color="#FECB52"),
            hoverinfo="none",
        )
        merge(clutter_kwargs, kwargs)

        self.fig.add_trace(go.Scatter(clutter_kwargs))  # trace for plotting clutter

        # Add data to frames
        for frame in self.fig.frames:
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            data_.append(go.Scatter(x=[np.inf], y=[np.inf]))
            traces_.append(trace_base)

            frame_time = datetime.fromisoformat(frame.name)

            cutoff_time = frame_time - self.time_window

            combined_data = {TrueDetection: detection_data, Clutter: clutter_data}

            for i, key in enumerate(combined_data.keys()):
                indices = np.where(
                    (combined_data[key]["time"] <= frame_time)
                    & (combined_data[key]["time"] >= cutoff_time)
                )

                data_.append(
                    go.Scatter(
                        x=combined_data[key]["x"][indices],
                        y=combined_data[key]["y"][indices],
                        meta=combined_data[key]["time_str"][indices],
                        hovertemplate="Measurement<br>(%{x}, %{y})<br>Time: %{meta}",
                    )
                )
                traces_.append(trace_base + i + 1)

            frame["data"] = data_
            frame["traces"] = traces_

        if resize:
            self._resize(combined_data, plot_type="measurements")

        self.plotting_function_called = True

    def _conv_measurements(self, measurements, convert_measurements=True):
        """
        Converts raw measurements into detections and clutter.

        Parameters
        ----------
        measurements : list
            A list of measurement states to convert.
        convert_measurements : bool, optional
            If True, uses the measurement model to convert measurements.

        Returns
        -------
        tuple
            A tuple containing two dictionaries: detections and clutter.
        """
        conv_detections, conv_clutter = {}, {}

        for state in measurements:
            meas_model = state.measurement_model
            state_vec = (
                state.state_vector
                if not convert_measurements
                else meas_model.inverse_function(state).flatten()
            )

            # Separate clutter and detections based on type
            target_dict = conv_clutter if isinstance(state, Clutter) else conv_detections
            target_dict[state] = (*state_vec,)

        return conv_detections, conv_clutter

    def plot_tracks(
        self, tracks, mapping, tracks_label="Tracks", plot_particles=False, resize=True
    ):
        """
        Plots ground truth states on the figure.

        Parameters
        ----------
        tracks : list
            A list of particle state sequences.
        mapping : list
            A list of indices mapping state vector components to x and y coordinates.
        truths_label : str, optional
            The label for the ground truth in the legend, default is "Ground Truth".
        resize : bool, optional
            If True, resizes the axes to fit the data, default is True.
        """
        data = self._prepare_data(tracks, mapping, data_type="track")

        # Set the base trace index for all tracks
        trace_base = len(self.fig.data)

        # Create the base trace for all tracks
        track_kwargs = dict(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=self.colorway[2]),
            legendgroup=tracks_label,
            legendrank=400,
            name=tracks_label,
            showlegend=True,
        )
        self.fig.add_trace(go.Scatter(track_kwargs))

        # Initialise traces for every track
        track_kwargs.update({"showlegend": False})

        # Update track colours
        for i, _ in enumerate(tracks):
            track_kwargs.update({"line": dict(color=self.colorway[(i + 2) % len(self.colorway)])})
            self.fig.add_trace(go.Scatter(track_kwargs))

        for frame in self.fig.frames:
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            frame_time = datetime.fromisoformat(frame.name)
            cutoff_time = frame_time - self.time_window

            # Add blank data to ensure track legend stays in place
            data_.append(go.Scatter(x=[-np.inf, np.inf], y=[-np.inf, np.inf]))
            traces_.append(trace_base)

            for i, track in enumerate(data):
                indices = np.where((track["time"] <= frame_time) & (track["time"] >= cutoff_time))

                data_.append(
                    go.Scatter(
                        x=track["x"][indices],
                        y=track["y"][indices],
                        meta=track["time_str"][indices],
                        hovertemplate="Particle<br>(%{x}, %{y})<br>Time: %{meta}",
                    )
                )
                traces_.append(trace_base + i + 1)

            frame["data"] = data_
            frame["traces"] = traces_

        if resize:
            self._resize(data, plot_type="tracks")

        if plot_particles:
            name = f"{tracks_label}<br>(Particles)"
            particle_kwargs = {
                "mode": "markers",
                "marker": {"size": 2, "color": self.colorway[2]},
                "opacity": 0.4,
                "hoverinfo": "skip",
                "legendgroup": name,
                "name": name,
                "legendrank": 520,
                "showlegend": True,
            }
            self.fig.add_trace(go.Scatter(particle_kwargs))
            particle_kwargs["showlegend"] = False
            for i, track in enumerate(data):
                particle_kwargs["marker"]["color"] = self.colorway[(i + 2) % len(self.colorway)]
                self.fig.add_trace(go.Scatter(particle_kwargs))
            self._plot_particles(tracks, mapping, resize)

        self.plotting_function_called = True

    def _plot_particles(self, tracks, mapping, resize):
        """
        Plots particle states on the figure.

        Parameters
        ----------
        tracks : list
            A list of particle state sequences.
        mapping : list
            A list of indices mapping state vector components to x and y coordinates.
        resize : bool, optional
            If True, resizes the axes to fit the data, default is True.
        """
        # Prepare data for each track
        data = self._prepare_data(tracks, mapping, data_type="particle")

        # Iterate over each frame to plot particles
        for frame in self.fig.frames:
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # Get frame timestamp
            frame_time = datetime.fromisoformat(frame.name)

            # Add placeholder for legend consistency
            data_.append(go.Scatter(x=[np.inf], y=[np.inf]))
            traces_.append(len(self.fig.data) - len(tracks) - 1)

            # Plot data for each track, filtering based on frame time
            for i, track in enumerate(data):
                # Mask for particles at the frame timestamp
                indices = track["time"] == frame_time

                # Append scatter plot for current particles
                data_.append(
                    go.Scatter(
                        x=track["x"][indices].flatten(),
                        y=track["y"][indices].flatten(),
                        meta=track["time_str"][indices].flatten(),
                        hovertemplate="Particle<br>(%{x}, %{y})<br>Time: %{meta}",
                    )
                )
                traces_.append(len(self.fig.data) - len(tracks) + i)

            # Update the frame with the new data and traces
            frame["data"] = data_
            frame["traces"] = traces_

        if resize:
            self._resize(data, plot_type="particle_or_uncertainty")

    def _prepare_data(self, data, mapping, data_type="track"):
        """
        Prepares data for plotting based on the type of data (track, truth, measurements,
        particles).

        Parameters
        ----------
        data : list
            A list of data objects (tracks, truths, measurements, particles).
        mapping : list
            A list of indices mapping state vector components to x and y coordinates.
        data_type : str, optional
            The type of data being processed ('track', 'truth', 'measurement', 'particle'),
            default is 'track'.

        Returns
        -------
        list
            A list of dictionaries with x, y, time, and time_str keys for plotting.
        """
        if data_type == "track":
            # General case for both tracks
            prepared_data = [
                {
                    "x": np.array([state.mean[mapping[0]] for state in datum]),
                    "y": np.array([state.mean[mapping[1]] for state in datum]),
                    "time": np.array([state.timestamp for state in datum], dtype=object),
                    "time_str": np.array(
                        [state.timestamp.strftime("%H:%M:%S") for state in datum], dtype=object
                    ),
                }
                for datum in data
            ]
        elif data_type == "measurement":
            # Case for measurement data (detections + clutter)
            prepared_data = {
                "x": np.array([state_vector[0] for state_vector in data.values()]),
                "y": np.array([state_vector[1] for state_vector in data.values()]),
                "time": np.array([state.timestamp for state in data.keys()], dtype=object),
                "time_str": np.array(
                    [state.timestamp.strftime("%H:%M:%S") for state in data.keys()], dtype=object
                ),
            }
        elif data_type == "particle" or data_type == "truth":
            # Case for particle data
            prepared_data = [
                {
                    "x": np.array([state.state_vector[mapping[0]] for state in datum]),
                    "y": np.array([state.state_vector[mapping[1]] for state in datum]),
                    "time": np.array([state.timestamp for state in datum], dtype=object),
                    "time_str": np.array(
                        [state.timestamp.strftime("%H:%M:%S") for state in datum], dtype=object
                    ),
                }
                for datum in data
            ]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        return prepared_data

    def _resize(self, data, plot_type="track"):
        """
        Resizes the figure axes to ensure all data is visible.

        Parameters
        ----------
        data : list
            A collection of values being added to the figure.
        plot_type : str, optional
            Type of data for resizing (e.g., 'ground_truth', 'measurements'), default is 'track'.
        """
        all_x = []
        all_y = []

        # Collect x and y data based on plot type
        if plot_type == "measurements":
            for key in data.keys():
                all_x.extend(data[key]["x"])
                all_y.extend(data[key]["y"])
        elif plot_type in ("ground_truth", "tracks"):
            for entry in data:
                all_x.extend(entry["x"])
                all_y.extend(entry["y"])
        elif plot_type == "sensor":
            sensor_xy = np.array([sensor.position[[0, 1], 0] for sensor in data])
            all_x.extend(sensor_xy[:, 0])
            all_y.extend(sensor_xy[:, 1])
        elif plot_type == "particle_or_uncertainty":
            for dictionary in data:
                for x_values in dictionary["x"]:
                    all_x.extend([np.nanmax(x_values), np.nanmin(x_values)])
                for y_values in dictionary["y"]:
                    all_y.extend([np.nanmax(y_values), np.nanmin(y_values)])

        # Determine new axis limits
        if all_x and all_y:  # Only resize if there's data
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)

            # Update axes with a small buffer
            buffer_x = (xmax - xmin) / 20
            buffer_y = (ymax - ymin) / 20

            if not self.plotting_function_called:
                # First time plotting data, set axes limits to the current data
                self.fig.update_xaxes(range=[xmin - buffer_x, xmax + buffer_x])
                self.fig.update_yaxes(range=[ymin - buffer_y, ymax + buffer_y])
            else:
                current_x_range = self.fig.layout.xaxis.range
                current_y_range = self.fig.layout.yaxis.range

                # Adjust axes limits if new data goes beyond current limits
                if xmax >= current_x_range[1] or xmin <= current_x_range[0]:
                    self.fig.update_xaxes(
                        range=[
                            min(xmin, current_x_range[0]) - buffer_x,
                            max(xmax, current_x_range[1]) + buffer_x,
                        ]
                    )
                if ymax >= current_y_range[1] or ymin <= current_y_range[0]:
                    self.fig.update_yaxes(
                        range=[
                            min(ymin, current_y_range[0]) - buffer_y,
                            max(ymax, current_y_range[1]) + buffer_y,
                        ]
                    )

    def show(self):
        self.fig.show()

    def save(self, filename="elpf.html"):
        self.fig.write_html(filename)
