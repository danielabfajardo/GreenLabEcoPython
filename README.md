# Green Lab Experiment

This repository contains `test_energy_training.py`, this script measures the energy, runtime, CPU, memory, and performance of two types of machine learning models: logistic regression and linear regression, using two libraries: scikit-learn and statsmodels.

## Purpose:
This helps compare different ML models and libraries not only by performance but also by energy efficiency and resource usage.

## Prerequisites

Before installing the requirements, ensure you have the following tools installed:

- **CodeCarbon**  
    [CodeCarbon Documentation](https://mlco2.github.io/codecarbon/)
    ```bash
    pip install codecarbon
    ```

- **PyJoules**  
    [PyJoules Documentation](https://pyjoules.readthedocs.io/)
    ```bash
    pip install pyJoules
    ```

## Installation

1. Clone the repository:
     ```bash
     git clone 
     cd experiment
     ```

2. Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

Run the energy training script:
```bash
python test_energy_training.py
```
Note: You might run the command with sudo 

The results will be available in csv format in `ml_energy_results.csv` and `emissions.csv`