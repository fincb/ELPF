"""
This module defines matrix wrapper classes for use in state estimation, inspired by the Stone Soup
library.

Stone Soup is an open-source framework for state estimation and tracking.
These classes were adapted from in Stone Soup to facilitate matrix manipulations.

Classes:
    Matrix: A general wrapper for numpy arrays.
    StateVector: A wrapper to enforce an Nx1 shape for state vectors.
    StateVectors: A wrapper for matrices of multiple state vectors.
    CovarianceMatrix: A wrapper for covariance matrices, enforcing NxN shape.

References:
    Stone Soup Library: https://stonesoup.readthedocs.io/
"""

from collections.abc import Sequence

import numpy as np


class Matrix(np.ndarray):
    """General Matrix wrapper for numpy arrays."""

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        return array.view(cls)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert inputs to floats where needed for safe operation
        inputs = [
            np.asarray(input_, dtype=np.float64) if isinstance(input_, Matrix) else input_
            for input_ in inputs
        ]
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        return Matrix(result)


class StateVector(Matrix):
    """State vector wrapper ensuring Nx1 vector shape."""

    def __new__(cls, input_array):
        # Ensure input_array is converted to an array
        array = np.asarray(input_array)
        if array.ndim == 1:
            # If it's a 1D array, reshape it to a column vector
            array = array[:, None]
        elif array.ndim > 2:
            raise ValueError("Input array must be 1D or 2D.")
        return array.view(cls)

    @property
    def mean(self):
        return np.mean(self, axis=0)

    def __getitem__(self, item):
        # Direct access for scalars
        if isinstance(item, int):
            item = (item, 0)
        return super().__getitem__(item)

    def flatten(self, *args, **kwargs):
        return np.ndarray.flatten(self, *args, **kwargs)

    def ravel(self, *args, **kwargs):
        return np.ndarray.ravel(self, *args, **kwargs)


class StateVectors(Matrix):
    """Matrix for multiple state vectors."""

    def __new__(cls, states, *args, **kwargs):
        if isinstance(states, Sequence) and not isinstance(states, np.ndarray):
            if isinstance(states[0], StateVector):
                return np.hstack(states).view(cls)
        array = np.asarray(states, *args, **kwargs)
        return array.view(cls)

    def __iter__(self):
        # Ensure each column is treated as a 2D column vector
        for col in np.array(self).T:
            if col.ndim == 1:  # Ensure it has at least 2 dimensions
                col = col[:, None]  # Make it a column vector
            yield StateVector(col)

    def average(self, axis=None, weights=None):
        return StateVector(np.average(self, axis=axis, weights=weights))

    def cov(self, bias=False):
        return CovarianceMatrix(np.cov(self, rowvar=True, bias=bias))


class CovarianceMatrix(Matrix):
    """Covariance matrix ensuring NxN shape."""

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("Covariance matrix must be square (NxN)")
        return array.view(cls)
