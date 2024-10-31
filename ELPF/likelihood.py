import numpy as np
from scipy.special import gammaln


def gaussian_pdf(
    observed_state: np.ndarray, predicted_states: np.ndarray, covar: np.ndarray
) -> np.ndarray:
    """
    Computes the likelihood of the observed state given multiple predicted states using a
    multivariate Gaussian probability density function (PDF).

    Parameters
    ----------
    observed_state : np.ndarray
        The observed measurement state (shape: (n, 1) or (n,)).
    predicted_states : np.ndarray
        Predicted measurement states from the particles (shape: (n, m) where m is the number of
        particles).
    covar : np.ndarray
        Covariance matrix (shape: (n, n)) for the measurement model. Can also be provided as a
        1D array for diagonal covariance.

    Returns
    -------
    np.ndarray
        An array of likelihoods for each predicted state (shape: (m,)). The likelihoods represent
        the probability of observing the given state for each of the predicted states.

    Notes
    -----
    The function assumes that the predicted states are in the same space as the observed state,
    and that the covariance matrix is valid (positive definite).
    """
    # Ensure covariance is a 2D matrix
    if covar.ndim == 1:
        covar = np.diag(covar)

    covar_inv = np.linalg.inv(covar)
    covar_det = np.linalg.det(covar)

    # Calculate the differences for each predicted state
    diffs = predicted_states - observed_state

    # Calculate the Mahalanobis distance for each difference
    mahalanobis_dists = np.zeros(predicted_states.shape[1])
    for i in range(predicted_states.shape[1]):
        mahalanobis_dists[i] = diffs[:, i].T @ covar_inv @ diffs[:, i]

    # Compute the normalizing constant for the Gaussian distribution
    norm_const = 1 / np.sqrt((2 * np.pi) ** observed_state.size * covar_det)

    # Calculate the PDF values using the vectorized Mahalanobis distances
    likelihoods = norm_const * np.exp(-0.5 * mahalanobis_dists)

    return likelihoods


def t_pdf(
    observed_state: np.ndarray, predicted_states: np.ndarray, covar: np.ndarray
) -> np.ndarray:
    """
    Computes the likelihood of the observed state given multiple predicted states using a
    multivariate Student's t-distribution, fully vectorized.

    Parameters
    ----------
    observed_state : np.ndarray
        The observed measurement state (column vector).
    predicted_states : np.ndarray
        Predicted measurement states from the particles (each column is a predicted state).
    covar : np.ndarray
        Covariance matrix (or diagonal array) for the measurement model.

    Returns
    -------
    np.ndarray
        An array of likelihoods for each predicted state.
    """
    df = observed_state.size
    d = observed_state.shape[0]

    # Ensure covariance is a 2D matrix and compute its determinant and inverse
    if covar.ndim == 1:
        covar = np.diag(covar)
    covar_inv = np.linalg.inv(covar)
    covar_det = np.linalg.det(covar)

    # Calculate the differences for each predicted state
    diffs = predicted_states - observed_state

    # Calculate the Mahalanobis distance for each difference in a vectorized way
    mahalanobis_dists = np.einsum("ij,ji->i", diffs.T @ covar_inv, diffs)

    # Compute the normalizing constants for the t-distribution
    norm_const = np.exp(gammaln((df + d) / 2) - gammaln(df / 2))
    scale_factor = (df * np.pi) ** (d / 2) * np.sqrt(covar_det)

    # Calculate the PDF values using the vectorized Mahalanobis distances
    likelihoods = norm_const / scale_factor / (1 + mahalanobis_dists / df) ** ((df + d) / 2)

    return likelihoods
