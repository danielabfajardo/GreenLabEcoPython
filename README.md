# Green Lab Experiment

The experiment framework is designed to:

Train and evaluate ML models (logistic regression or linear regression) using either scikit-learn or statsmodels.

- Measure CPU time, memory usage, runtime, and energy consumption during training and inference.

- Estimate CO₂ emissions based on energy consumed.


### Files

- RunnerConfig.py: Experiment runner configuration and hooks for measurement.

- ml_utils.py: ML model training, inference, and persistence utilities.

- energy_utils.py: Functions to summarize EnergiBridge logs and compute energy/CO₂.