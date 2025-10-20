# GreenLabEcoPython

GreenLabEcoPython is a collection of machine learning energy profiling experiments designed to work with [Experiment Runner](https://github.com/S2-group/experiment-runner) and [EnergiBridge](https://github.com/tdurieux/EnergiBridge). It demonstrates how to measure energy, runtime, and resource usage for ML tasks using real-world datasets.

---

## Requirements

- **Python 3.8+**
- **Experiment Runner**: Framework for orchestrating experiments  
  [GitHub & Installation Guide](https://github.com/S2-group/experiment-runner)
- **EnergiBridge**: Power measurement tool  
  [GitHub & Installation Guide](https://github.com/tdurieux/EnergiBridge)
- **Required Python packages**:  
  Install with:
  ```sh
  pip install -r requirements.txt
  ```

---

## Setup Instructions

### 1. Clone the repositories

```sh
git clone https://github.com/S2-group/experiment-runner.git
cd experiment-runner/
# GreenLabEcoPython is inside this repo (GreenLabEcoPython/)
```

### 2. Install Experiment Runner

Follow the [Experiment Runner installation instructions](https://github.com/S2-group/experiment-runner#requirements):

```sh
pip install -r requirements.txt
```

### 3. Install EnergiBridge

Follow the [EnergiBridge installation guide](https://github.com/tdurieux/EnergiBridge#installation).

Example for Linux:
```sh
git clone https://github.com/tdurieux/EnergiBridge.git
cd EnergiBridge
make
sudo cp energibridge /usr/local/bin/
```

### 4. Download the Medical Insurance Dataset

- Go to [Kaggle: Medical Insurance Cost Prediction](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction?select=medical_insurance.csv)
- Download `medical_insurance.csv`
- Place the file in:
  ```
  GreenLabEcoPython/datasets/medical_insurance.csv
  ```

### 5. Preprocess the Dataset

This step splits the dataset into small, medium, and large versions with different feature counts and sizes.

```sh
python -m GreenLabEcoPython.ml_utils
```

You should see output indicating creation of train/test splits for each dataset size.

### 6. Run the Experiment

From the root of the repository, execute:

```sh
python experiment-runner/ GreenLabEcoPython/RunnerConfig.py
```

This will run all experiment variations, measure energy and resource usage, and save results in:

```
GreenLabEcoPython/experiments/ml_energy_experiment2/
```

---

## Output

- **Processed datasets**:  
  `GreenLabEcoPython/datasets/data_small_train.csv`, `data_small_test.csv`, etc.
- **Experiment results**:  
  `GreenLabEcoPython/experiments/ml_energy_experiment2/run_table.csv`  
  (Contains accuracy, F1, R2, runtime, energy, CO2, CPU/mem stats for each run)

  The run_table for the experiment will be found in the GreenLabEcoPython/experiments/ folder along side subsidiary folders containing information about each run of the experiment.

---

## Troubleshooting

- If you get errors about missing columns, ensure your dataset matches the expected format.
- If EnergiBridge fails, check installation and permissions (may require `sudo`).
- For more details, see the [Experiment Runner Wiki](https://github.com/S2-group/experiment-runner/wiki).

---

## References

- [Experiment Runner](https://github.com/S2-group/experiment-runner)
- [EnergiBridge](https://github.com/tdurieux/EnergiBridge)
- [Medical Insurance Dataset on Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction?select=medical_insurance.csv)

---

## License

See [LICENSE](LICENSE).