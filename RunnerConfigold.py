from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge

from typing import Dict, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import psutil
import time
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from ml_utils import train_model, inference_model, save_model, load_model, MODEL_SAVE_PATH, preprocess_datasets
from energy_utils import compute_energy_from_log, summarize_energy_profile


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))
    
    name: str = "ml_energy_experiment"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000

    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
        ])
        self.run_table_model = None
        self.profiler = None
        self.current_phase = None
        self.current_metrics = {}

        output.console_log("Custom ML Energy Runner loaded")

    def create_run_table_model(self) -> RunTableModel:
        factor_algo = FactorModel("algorithm", ["logistic", "linear"])
        factor_lib = FactorModel("library", ["sklearn", "statsmodels"])
        factor_phase = FactorModel("process", ["training", "inference"])
        return RunTableModel(
            factors=[factor_algo, factor_lib, factor_phase],
            repetitions=1,
            data_columns=[
                "accuracy", "f1_score", "r2",
                "runtime_s", "cpu_user_s", "cpu_system_s",
                "memory_delta_mb", "energy_j", "estimated_CO2_kg"
            ]
        )

    def start_measurement(self, context: RunnerContext):
        self.current_phase = context.execute_run["process"]
        algo = context.execute_run["algorithm"]

        # load dataset
        data = load_breast_cancer() if algo == "logistic" else load_diabetes()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.phase_Xy = (X_train, y_train) if self.current_phase == "training" else (X_test, y_test)

        # Start profiler
        self.profiler = EnergiBridge(
            target_program=f"python3 -c 'print(\"noop\")'",
            out_file=context.run_dir / "energibridge.csv"
        )
        self.profiler.start()

        # System resource snapshot
        proc = psutil.Process()
        self.cpu_before = proc.cpu_times()
        self.mem_before = proc.memory_info().rss / (1024 * 1024)
        self.proc = proc

    def stop_measurement(self, context: RunnerContext):
        algo = context.execute_run["algorithm"]
        lib = context.execute_run["library"]

        # Stop the energy profiler
        self.profiler.stop(wait=True)

        result = self.profiler.parse_log(
        self.profiler.logfile, self.profiler.summary_logfile)

        # EnergiBridge can return either (df, dict) or (dict)
        if isinstance(result, tuple) and len(result) == 2:
            eb_log, eb_summary = result
        else:
            eb_log, eb_summary = result, {}

        # Convert dict to DataFrame if needed
        import pandas as pd
        if isinstance(eb_log, dict):
            try:
                eb_log = pd.DataFrame(eb_log)
                print("[INFO] Converted eb_log dict to DataFrame.")
            except Exception as e:
                print(f"[ERROR] Failed to convert eb_log to DataFrame: {e}")
                eb_log = pd.DataFrame()

        energy_summary = summarize_energy_profile(eb_log)

        self.energy_j = energy_summary["total_energy_j"]
        self.runtime_s = energy_summary["runtime_s"]
        self.avg_power_w = energy_summary["avg_power_w"]
        # Estimate CO2 emissions from energy consumption (kg)
        #1 kWh=3.6×10^6 J and 1 kWh≈0.475 kg CO₂
        # CO₂ (kg)=energy (J)× 0.475/3.6×10^6​

        CO2_PER_J = 0.475 / 3_600_000  # kg CO2 per Joule
        self.estimated_CO2_kg = self.energy_j * CO2_PER_J

        # Record CPU/memory deltas
        cpu_after = self.proc.cpu_times()
        mem_after = self.proc.memory_info().rss / (1024 * 1024)
        self.cpu_user = cpu_after.user - self.cpu_before.user
        self.cpu_system = cpu_after.system - self.cpu_before.system
        self.memory_delta = mem_after - self.mem_before

        # Handle experiment phase
        if self.current_phase == "training":
            X_train, y_train = self.phase_Xy
            trained_model = train_model(algo, lib, X_train, y_train)
            save_model(trained_model, MODEL_SAVE_PATH)
            self.current_metrics = {"accuracy": None, "f1_score": None, "r2": None}

        elif self.current_phase == "inference":
            X_test, y_test = self.phase_Xy
            model = load_model(MODEL_SAVE_PATH)
            metrics = inference_model(algo, lib, model, X_test, y_test)

            if algo == "logistic":
                acc, f1 = metrics
                self.current_metrics = {"accuracy": acc, "f1_score": f1, "r2": None}
            else:
                self.current_metrics = {"accuracy": None, "f1_score": None, "r2": metrics}

    def populate_run_data(self, context: RunnerContext):
        import json

        runtime_s = None

        # Try to read runtime info from EnergiBridge summary file
        if self.profiler and self.profiler.summary_logfile:
            summary_path = self.profiler.summary_logfile
            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        summary_data = json.load(f)
                        runtime_s = summary_data.get("runtime_seconds", None)
                except Exception as e:
                    print(f"[WARNING] Could not parse EnergiBridge summary file: {e}")

        return {
            "accuracy": self.current_metrics.get("accuracy"),
            "f1_score": self.current_metrics.get("f1_score"),
            "r2": self.current_metrics.get("r2"),
            "runtime_s": runtime_s or getattr(self, "runtime_s", None),
            "cpu_user_s": getattr(self, "cpu_user", None),
            "cpu_system_s": getattr(self, "cpu_system", None),
            "memory_delta_mb": getattr(self, "memory_delta", None),
            "energy_j": getattr(self, "energy_j", None),
            "estimated_CO2_kg": getattr(self, "estimated_CO2_kg", None),
        }
