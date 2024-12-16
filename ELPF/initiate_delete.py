import numpy as np

from ELPF.detection import MissedDetection
from ELPF.state import Particle, ParticleState, Track


class GaussianParticleInitiator:
    """
    Initialises tracks for new detections by generating Gaussian-distributed particles.

    Attributes:
        num_particles (int): Number of particles to initialise for each track.
    """

    def __init__(self, num_particles):
        """
        Args:
            num_particles (int): Number of particles to initialise per track.
        """
        self.num_particles = num_particles

    def initiate(self, detections, timestamp):
        """
        Initiates new tracks from the provided detections.

        Args:
            detections (iterable): Collection of detections for new track initiation.
            timestamp (datetime): Timestamp to associate with the initial state.

        Returns:
            set: A set of newly initiated tracks.
        """
        tracks = set()

        for detection in detections:
            # Extract measurement model from the detection
            measurement_model = detection.measurement_model

            # Transform detection into state vector (e.g., [x, vx, y, vy])
            state_vector = measurement_model.inverse_function(detection).flatten()
            state_vector = np.array([state_vector[0], 0, state_vector[1], 0])

            # Generate particles by sampling from a Gaussian distribution
            samples = np.random.multivariate_normal(
                mean=state_vector,
                cov=np.diag([1.5, 0.5, 1.5, 0.5]),
                size=self.num_particles,
            )

            # Assign equal weights to all particles
            weights = np.ones(self.num_particles) / self.num_particles

            # Create particle objects
            particles = np.array(
                [Particle(sample, weight) for sample, weight in zip(samples, weights)]
            )

            # Create an initial ParticleState for the track
            prior = ParticleState(particles, timestamp=timestamp)

            # Add the track to the set
            tracks.add(Track([prior]))

        return tracks


class CovarianceBasedDeleter:
    """
    Deletes tracks based on the trace of their covariance matrix.

    Attributes:
        covar_trace_thresh (float): Threshold for the covariance trace above which tracks are
        deleted.
    """

    def __init__(self, covar_trace_thresh):
        """
        Args:
            covar_trace_thresh (float): Threshold for deleting tracks based on covariance.
        """
        self.covar_trace_thresh = covar_trace_thresh

    def delete(self, tracks):
        """
        Identifies tracks to delete based on the covariance trace.

        Args:
            tracks (set): Set of tracks to evaluate for deletion.

        Returns:
            set: Tracks that exceed the covariance trace threshold.
        """
        tracks_to_delete = set()

        for track in tracks:
            # Calculate the trace of the covariance matrix
            covar_trace = np.trace(track[-1].covar)

            # Mark track for deletion if the covariance trace exceeds the threshold
            if covar_trace > self.covar_trace_thresh:
                tracks_to_delete.add(track)

        return tracks_to_delete


class MultiMeasurementInitiator:
    """
    Manages track initiation, confirmation, and deletion for a multi-measurement scenario.

    Attributes:
        min_points (int): Minimum number of updates required to confirm a track.
        particle_filter (object): Particle filter instance for prediction and update.
        hypothesiser (object): Hypothesiser for generating data association hypotheses.
        time_interval (float): Time interval for track prediction in seconds.
        initiator (object): Track initiator to handle unassociated measurements.
        deleter (object): Track deleter to remove invalid tracks.
    """

    def __init__(
        self, min_points, particle_filter, hypothesiser, initiator, deleter, time_interval=1
    ):
        """
        Args:
            min_points (int): Minimum number of updates required to confirm a track.
            particle_filter (object): Particle filter instance for prediction and update.
            hypothesiser (object): Hypothesiser for generating data association hypotheses.
            initiator (object): Track initiator to handle unassociated measurements.
            deleter (object): Track deleter to remove invalid tracks.
            time_interval (float): Time interval for track prediction in seconds.
        """
        self.min_points = min_points
        self.particle_filter = particle_filter
        self.hypothesiser = hypothesiser
        self.time_interval = time_interval
        self.initiator = initiator
        self.deleter = deleter
        self.unconfirmed_tracks = set()

    def initiate(self, detections, timestamp):
        """
        Processes detections to manage track initiation, confirmation, and deletion.

        Args:
            detections (set): Set of detections to process.
            timestamp (datetime): Timestamp to associate with new tracks.

        Returns:
            set: Set of confirmed tracks.
        """
        if not detections:
            return set()

        confirmed_tracks = set()
        associated_detections = set()
        to_remove = set()

        # Update existing unconfirmed tracks
        for track in self.unconfirmed_tracks:
            # Predict the next state
            prior = self.particle_filter.predict(track[-1], self.time_interval)

            # Generate hypotheses and update the track
            hypotheses = self.hypothesiser.hypothesise(prior, detections)
            posterior = self.particle_filter.update(prior, hypotheses)

            # Associate measurements with the track
            for hypothesis in hypotheses:
                if not isinstance(hypothesis.measurement, MissedDetection):
                    associated_detections.add(hypothesis.measurement)

            # Append updated state to the track
            track.append(posterior)

            # Confirm the track if it has enough updates
            if len(track) >= self.min_points:
                confirmed_tracks.add(track)
                to_remove.add(track)

        # Remove confirmed tracks from the unconfirmed set
        self.unconfirmed_tracks -= to_remove

        # Delete tracks that are no longer valid
        self.unconfirmed_tracks -= self.deleter.delete(self.unconfirmed_tracks)

        # Initiate new tracks for unassociated detections
        unassociated_detections = detections - associated_detections
        self.unconfirmed_tracks |= self.initiator.initiate(unassociated_detections, timestamp)

        return confirmed_tracks
