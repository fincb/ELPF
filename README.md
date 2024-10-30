# Expected Likelihood Particle Filter

This repository contains a WIP (Work In Progress) Python implementation of an Expected Likelihood Particle Filter for state estimation.

Currently, the repository includes both the Bootstrap Particle Filter and the Expected Likelihood Particle Filter, which are used for estimating the state of a moving object based on noisy measurements in a range-bearing format. This implementation consists of classes for managing states, particles, and the particle filter algorithms.

## Overview

The Expected Likelihood Particle Filter (ELPF) is a method for tracking targets amidst clutter using a particle filter framework[^1]. This approach is particularly beneficial in scenarios where traditional gating techniques, which rely on a readily available covariance matrix, are unsuitable due to the non-linearity or non-Gaussian characteristics of the system.

The ELPF directly addresses the uncertainty in measurement origins—a common problem in cluttered environments—by incorporating Probabilistic Data Association (PDA). Instead of relying on a single gated measurement, the ELPF computes an expected likelihood, which is essentially a weighted mixture of individual likelihoods from all available measurements. These weights are determined by PDA calculations. By updating the particle weights based on this expected likelihood, the ELPF achieves a more comprehensive and robust update process.

## Features

- **Bootstrap Particle Filter**: Implements a basic particle filtering approach for state estimation.
- **Expected Likelihood Particle Filter**: Enhances the standard particle filter by incorporating Probabilistic Data Association for improved tracking in cluttered environments.

## References

[^1]: Marrs, A., Maskell, S., & Bar-Shalom, Y. (2002, August). Expected likelihood for tracking in clutter with particle filters. In *Signal and Data Processing of Small Targets 2002* (Vol. 4728, pp. 230-239). SPIE. [Link to paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4728/1/Expected-likelihood-for-tracking-in-clutter-with-particle-filters/10.1117/12.478507.full)
