import numpy as np
from scipy.stats import multivariate_normal, multivariate_t


def gaussian_pdf(
    observed_state: np.ndarray, predicted_state: np.ndarray, covar: np.ndarray
) -> float:
    """
    Computes the likelihood of the observed state given the predicted state using a Gaussian PDF.

    Parameters
    ----------
    observed_state : np.ndarray
        The observed measurement state.
    predicted_state : np.ndarray
        The predicted measurement state from the particle.
    covar : np.ndarray
        The covariance matrix for the measurement model.

    Returns
    -------
    float
        The likelihood of the observed state given the predicted state.
    """
    diff = observed_state - predicted_state
    likelihood = multivariate_normal.pdf(diff.flatten(), cov=covar)
    return likelihood


def t_pdf(observed_state: np.ndarray, predicted_state: np.ndarray, covar: np.ndarray) -> float:
    """
    Computes the likelihood of the observed state given the predicted state using a multivariate
    Student's t-distribution.

    Parameters
    ----------
    observed_state : np.ndarray
        The observed measurement state.
    predicted_state : np.ndarray
        The predicted measurement state from the particle.
    covar : np.ndarray
        The covariance matrix for the measurement model.

    Returns
    -------
    float
        The likelihood of the observed state given the predicted state.
    """
    df = observed_state.size
    diff = observed_state - predicted_state

    # Ensure the covariance is a 2D matrix
    if covar.ndim == 1:
        covar = np.diag(covar)

    likelihood = multivariate_t.pdf(diff.flatten(), df=df, shape=covar)
    return likelihood
