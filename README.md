# Supervised Learning
This repository contains multiple algorithms implemented for a coursework in Supervised Learning for my master's degree. All algorithms are implemented only using
numpy/linear algebra. Where possible numba is used to JIT the code for performance improvements. The code also makes extensive use of caching, which greatly improves runtime
but increases memory requirements.

## Digit Classification
Code for digit classification on MNIST. Run `exercises.py` to run the experiments and hyperparameter tuning. Implemented algorithms include:
1. (Kernelised) Support Vector Machine: Implementation of Sequential Minimal Optimization (SMO). Multi-class implementation with One-versus-All scheme.
2. (Kernelised) Perceptron: Implementation of the perceptron algorithm with Polynomial and Gaussian kernels. Multi-class implementation with One-versus-All scheme and native.
3. Multi-layer perceptron: Implementation of the multi-layer peceptron (feedforward neural network) with L1 and L2 regularisation, SGD and (mini-)batch GD + momentum.

## Sample Complexity
Estimates the sample complexity for different algorithms on a problem with many irrelevant features. Run `exercises.py` to run the experiments. Implemented algorithms include:
1. Ordinary Least Squares (OLS)
2. Nearest Neighbours: Optimised implementation for this specific problem setting.
3. Perceptron
4. Winnow
