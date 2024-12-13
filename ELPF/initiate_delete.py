import numpy as np

from ELPF.state import Particle, ParticleState, Track


class GaussianParticleInitiator:
    def __init__(self, num_particles):
        self.num_particles = num_particles

    def initiate(self, detections, timestamp):
        tracks = set()

        for detection in detections:
            measurement_model = detection.measurement_model
            state_vector = measurement_model.inverse_function(detection).flatten()
            state_vector = np.array([state_vector[0], 0, state_vector[1], 0])

            # Create prior state
            samples = np.random.multivariate_normal(
                mean=state_vector,
                cov=np.diag([0.01, 0.05, 0.01, 0.05]),
                size=self.num_particles,
            )
            weights = np.ones(self.num_particles) / self.num_particles
            particles = np.array(
                [Particle(sample, weight) for sample, weight in zip(samples, weights)]
            )
            prior = ParticleState(particles, timestamp=timestamp)
            tracks.add(Track([prior]))

        return tracks


class CovarianceBasedDeleter:
    def __init__(self, covar_trace_thresh):
        self.covar_trace_thresh = covar_trace_thresh

    def delete(self, tracks):
        tracks_to_delete = set()
        for track in tracks:
            covar_trace = np.trace(track[-1].covar)

            if covar_trace > self.covar_trace_thresh:
                tracks_to_delete.add(track)

        return tracks_to_delete
