# Augmenting Economic Data Using Variational Autoencoder

This repository hosts my codes for the project on simulating economic data using variational autoencoder with the ultimate aim of improving model predictive performance or causal inference.

Here is the abstract: economic data often suffers from being limited or insufficient, making robust inference and prediction challenging. This research explores the usage of variational autoencoders (VAEs) to generate synthetic tabular economic data that closely mirrors the true data distribution. Building on the work of Athey, Imbens, Metzger, and Munro (2021), who use Wasserstein GANs to simulate data from the classic Lalonde datasets, which are widely employed to evaluate causal inference methods, this research applies VAE to the same datasets to demonstrate its capability in simulating tabular data. Beyond cross-sectional data, this study aims to develop VAEs to generate sequential data such as time series. The ultimate goal is to create a systemic way of augmenting economic datasets to enhance the performances of causal inference methods and forecasting models, drawing inspiration from how data augmentation significantly improves model performances in computer vision and NLP machine learning models.

## File structure

The `data` folder contains different versions of the LaLonde dataset.

The `figures` folder contains the output plots comparing marginal distributions of real versus generated data.

[`main.py`](main.py) the master script that runs the complete pipeline of setting up, training the VAE, and creating the analysis plots.

[`plotting.py`](plotting.py) contains functions for plotting training and validation losses, and the marginal distribution comparisons.

[`utils.py`](utils.py) contains the utility functions for data preprocessing, model evaluation, and data generation.

[`VAE.py`](VAE.py) contains the VAE model class.
