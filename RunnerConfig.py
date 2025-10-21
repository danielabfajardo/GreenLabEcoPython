from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge
from ProgressManager.RunTable.Models.RunProgress import RunProgress
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import psutil
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from ml_utils import train_model, inference_model, save_model, load_model, MODEL_SAVE_PATH, preprocess_datasets
from energy_utils import compute_energy_from_log, summarize_energy_profile
from ml_utils import preprocess_datasets
import json
import csv

# ================ USER SPECIFIC CONFIGURATIONS ================================
# This class is used to Orchestrate the experiment runner's configuration.
# The Experiment Runner provides methods to hook into various stages of the experiment lifecycle,
# allowing for customizing the experiment behavior as needed.
class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__))) 
    name: str = "ml_energy_experiment2"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 120_000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None

        output.console_log("Custom ML Energy Runner loaded")

    # The run table is customized to have specific combinations in a specific order
    # rather than using the built-in combination generator which randomizes all factor treatments
    # What we require is to have all combinations of (algorithm, library, dataset)
    # and for each combination, have two runs: one for training and one for inference
    # This partial randomization ensures that similar runs are grouped together in training and inference sequences
    def create_run_table_model(self) -> RunTableModel:
        factor_algo    = FactorModel("algorithm", ["logistic", "linear"])
        factor_lib     = FactorModel("library", ["sklearn", "statsmodels"])
        factor_dataset = FactorModel("dataset", ["small", "medium", "large"])
        factor_process = FactorModel("process", ["training", "inference"])

        # choose how many randomized passes you want
        repetitions = 20

        def build_rows_once(rep_idx: int):
            # all (algo, lib, dataset), shuffled for this repetition
            combos = [
                (a, l, d)
                for a in factor_algo.treatments
                for l in factor_lib.treatments
                for d in factor_dataset.treatments
            ]
            random.shuffle(combos)

            rows = []
            for a, l, d in combos:
                for proc in factor_process.treatments:
                    rows.append({
                        "__run_id": f"{a}_{l}_{d}_{proc}_rep{rep_idx}",
                        "__done": RunProgress.TODO,
                        "algorithm": a,
                        "library": l,
                        "dataset": d,
                        "process": proc,
                        "accuracy": " ",
                        "f1_score": " ",
                        "r2": " ",
                        "runtime_s": " ",
                        "cpu_user_s": " ",
                        "cpu_system_s": " ",
                        "memory_delta_mb": " ",
                        "energy_j": " ",
                        "estimated_CO2_kg": " "
                    })
            return rows

        # accumulate randomized blocks per repetition
        all_rows = []
        for rep in range(repetitions):
            all_rows.extend(build_rows_once(rep))

        # Create model just for API compatibility and monkey‑patch generator
        rt = RunTableModel(
            factors=[factor_algo, factor_lib, factor_dataset, factor_process],
            repetitions=repetitions,
            data_columns=[
                "accuracy", "f1_score", "r2",
                "runtime_s", "cpu_user_s", "cpu_system_s",
                "memory_delta_mb", "energy_j", "estimated_CO2_kg"
            ]
        )
        rt.generate_experiment_run_table = lambda: all_rows
        self.run_table_model = rt
        return rt

    # This method was used to preprocess datasets before starting the experiment
    # It is invoked only once during the lifetime of the program
    # It splits the dataset into small, medium and large each with their training and testing sets
    def before_experiment(self) -> None:
        print("Starting ML Energy Experiment...")
        print("Processing the datasets and preparing for runs...")
        preprocess_datasets()

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        pass

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        pass       

    # This function initializes and starts the energy and resource measurement for an experiment run.
    # It sets up the EnergiBridge profiler, configures output paths, and begins logging energy, CPU, and memory usage.
    # The measurement continues until stop_measurement is called, ensuring all resource usage during the run is captured.
    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        algorithm = context.execute_run["algorithm"]
        library = context.execute_run["library"]
        dataset = context.execute_run["dataset"]
        process = context.execute_run["process"]

        # derive a run identifier to pass into the target script so it can populate the central CSV
        run_id = None
        try:
            # preferred: the run entry contains __run_id
            run_id = context.execute_run.get("__run_id")
        except Exception:
            # fallback to the run dir name
            run_id = context.run_dir.name

        cmd = (
            f"python3 -m GreenLabEcoPython.ml_scripts.{library}_{algorithm}_{process}"
            f" --dataset {dataset} --run_id \"{run_id}\""
        )

        self.profiler = EnergiBridge(
            target_program=cmd,
            out_file=context.run_dir / "energibridge.csv"
        )
        # If energibridge does not require sudo on this machine, avoid sudo prompt:
        self.profiler.requires_admin = False
        self.profiler.start()

        # System resource snapshot
        proc = psutil.Process()
        self.cpu_before = proc.cpu_times()
        self.mem_before = proc.memory_info().rss / (1024 * 1024)
        self.proc = proc

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        pass

    # This function stops the energy and resource measurement for an experiment run.
    # It finalizes the EnergiBridge profiler, collects and saves the logged energy, CPU, and memory usage data.
    # This ensures all resource usage during the run is properly recorded and available for analysis.
    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
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

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        pass

    # This function collects and aggregates all relevant metrics and results from the experiment run.
    # It extracts accuracy, F1, R2, runtime, energy, CO2, CPU, and memory usage, and organizes them into a structured format.
    # The data is then saved to the run table for later analysis and comparison across different experiment configurations.
    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """
        Populate the run table with metrics from inference_results.csv and calculated energy/system metrics.
        """
        import pandas as pd

        # Get run_id for this run
        run_id = context.execute_run.get("__run_id")

        # Read inference results
        inference_csv = Path(self.ROOT_DIR) / "inference_results.csv"
        if inference_csv.exists():
            df = pd.read_csv(inference_csv)
            row = df[df["__run_id"] == run_id]
            if not row.empty:
                accuracy = row.iloc[0].get("accuracy", "")
                f1_score = row.iloc[0].get("f1_score", "")
                r2 = row.iloc[0].get("r2", "")
            else:
                accuracy = ""
                f1_score = ""
                r2 = ""
        else:
            accuracy = ""
            f1_score = ""
            r2 = ""

        # Calculate energy/system metrics
        energy_j = getattr(self, "energy_j", None)
        runtime_s = getattr(self, "runtime_s", None)
        avg_power_w = getattr(self, "avg_power_w", None)
        estimated_co2_kg = getattr(self, "estimated_co2_kg", None)
        cpu_user_s = getattr(self, "cpu_user", None)
        cpu_system_s = getattr(self, "cpu_system", None)
        memory_delta_mb = getattr(self, "memory_delta", None)

        return {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "r2": r2,
            "runtime_s": runtime_s,
            "energy_j": energy_j,
            "avg_power_w": avg_power_w,
            "estimated_CO2_kg": estimated_co2_kg,
            "cpu_user_s": cpu_user_s,
            "cpu_system_s": cpu_system_s,
            "memory_delta_mb": memory_delta_mb,
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
