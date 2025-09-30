# test_energy_training_split.py
import time
import numpy as np
import pandas as pd
import psutil
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
import statsmodels.api as sm
from pyJoules.energy_meter import measure_energy
from codecarbon import EmissionsTracker
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Algorithms and libraries
# -------------------------------
algorithms = ["logistic", "linear"]
libraries = ["sklearn", "statsmodels"]

results_list = []

# -------------------------------
# Helper functions
# -------------------------------
@measure_energy()
def train_model(algo, lib, X_train, y_train):
    if algo == "logistic":
        if lib == "sklearn":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            return model
        else:  # statsmodels logistic
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_sm = sm.add_constant(X_train_scaled)
            model = sm.GLM(y_train, X_sm, family=sm.families.Binomial()).fit()
            return (model, scaler)  # return scaler too
    else:  # linear regression
        if lib == "sklearn":
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        else:  # statsmodels linear
            X_sm = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_sm).fit()
            return model

@measure_energy()
def inference_model(algo, lib, model, X_test, y_test):
    if algo == "logistic":
        if lib == "sklearn":
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            return acc, f1
        else:  # statsmodels logistic
            model, scaler = model
            X_test_scaled = scaler.transform(X_test)
            X_test_sm = sm.add_constant(X_test_scaled)
            y_pred = (model.predict(X_test_sm) >= 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            return acc, f1
    else:  # linear regression
        if lib == "sklearn":
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
        else:  # statsmodels linear
            X_test_sm = sm.add_constant(X_test)
            y_pred = model.predict(X_test_sm)
            r2 = r2_score(y_test, y_pred)
            return r2

# -------------------------------
# Energy & carbon tracking
# -------------------------------
tracker = EmissionsTracker(measure_power_secs=1)
tracker.start()

for algo in algorithms:
    print(f"\nRunning {algo} regression...")

    # Load dataset
    if algo == "logistic":
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        data = load_diabetes()
        X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for lib in libraries:
        print(f"Library: {lib}")

        # -------------------------------
        # Training phase
        # -------------------------------
        process = psutil.Process()
        cpu_before = process.cpu_times()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        model = train_model(algo, lib, X_train, y_train)
        end_time = time.time()

        cpu_after = process.cpu_times()
        mem_after = process.memory_info().rss / (1024 * 1024)

        runtime = end_time - start_time
        emissions = tracker.stop()

        cpu_user = cpu_after.user - cpu_before.user
        cpu_system = cpu_after.system - cpu_before.system
        mem_usage = mem_after - mem_before

        results_list.append({
            "phase": "training",
            "algorithm": algo,
            "library": lib,
            "accuracy": None,
            "f1_score": None,
            "r2": None,
            "runtime_s": runtime,
            "cpu_user_s": cpu_user,
            "cpu_system_s": cpu_system,
            "memory_delta_mb": mem_usage,
            "estimated_CO2_kg": emissions
        })

        # -------------------------------
        # Inference phase
        # -------------------------------
        process = psutil.Process()
        cpu_before = process.cpu_times()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        metrics = inference_model(algo, lib, model, X_test, y_test)
        end_time = time.time()

        cpu_after = process.cpu_times()
        mem_after = process.memory_info().rss / (1024 * 1024)

        runtime = end_time - start_time
        emissions = tracker.stop()

        cpu_user = cpu_after.user - cpu_before.user
        cpu_system = cpu_after.system - cpu_before.system
        mem_usage = mem_after - mem_before

        if algo == "logistic":
            acc, f1 = metrics
            results_list.append({
                "phase": "inference",
                "algorithm": algo,
                "library": lib,
                "accuracy": acc,
                "f1_score": f1,
                "r2": None,
                "runtime_s": runtime,
                "cpu_user_s": cpu_user,
                "cpu_system_s": cpu_system,
                "memory_delta_mb": mem_usage,
                "estimated_CO2_kg": emissions
            })
        else:
            r2 = metrics
            results_list.append({
                "phase": "inference",
                "algorithm": algo,
                "library": lib,
                "accuracy": None,
                "f1_score": None,
                "r2": r2,
                "runtime_s": runtime,
                "cpu_user_s": cpu_user,
                "cpu_system_s": cpu_system,
                "memory_delta_mb": mem_usage,
                "estimated_CO2_kg": emissions
            })

# -------------------------------
# Save combined CSV
# -------------------------------
df = pd.DataFrame(results_list)
df.to_csv("ml_energy_results.csv", index=False)
print("\nSaved results to ml_energy_split.csv")
print(df)
