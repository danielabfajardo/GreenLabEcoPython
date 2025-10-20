"""
energy_utils.py

Utility functions for parsing and computing energy usage from EnergiBridge logs.
Designed to handle variable column names across hardware platforms.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compute_energy_from_log(eb_log: pd.DataFrame) -> float:
    """
    Compute total energy used during a run based on available columns.
    
    Parameters
    ----------
    eb_log : pd.DataFrame
        DataFrame produced by EnergiBridge.parse_log()
    
    Returns
    -------
    float
        Total energy in Joules.
    """
    # First preference: direct CPU energy measurement
    if "CPU_ENERGY (J)" in eb_log.columns:
        energy_j = eb_log["CPU_ENERGY (J)"].iloc[-1] - eb_log["CPU_ENERGY (J)"].iloc[0]
        print("[INFO] Energy computed from 'CPU_ENERGY (J)' column.")
        return energy_j

    # Fallback: sum energy from per-core readings
    core_cols = [c for c in eb_log.columns if "CORE" in c and "ENERGY" in c]
    if core_cols:
        energy_j = sum(eb_log[c].iloc[-1] - eb_log[c].iloc[0] for c in core_cols)
        print(f"[INFO] Energy computed as sum of core energies: {', '.join(core_cols)}")
        return energy_j

    # Check for PACKAGE_ENERGY or other direct energy columns
    energy_cols = [c for c in eb_log.columns if "ENERGY" in c and "(J)" in c]
    if energy_cols:
        energy_j = sum(eb_log[c].iloc[-1] - eb_log[c].iloc[0] for c in energy_cols)
        print(f"[INFO] Energy computed using energy columns: {', '.join(energy_cols)}")
        return energy_j

    # Last resort: integrate power over time (for macOS/systems without direct energy readings)
    if "SYSTEM_POWER (Watts)" in eb_log.columns and "Delta" in eb_log.columns:
        # Delta is in milliseconds, convert to seconds
        time_deltas_s = eb_log["Delta"] / 1000.0
        power_w = eb_log["SYSTEM_POWER (Watts)"]
        
        # Energy = Power × Time (trapezoidal integration)
        energy_j = np.trapz(power_w, dx=time_deltas_s.mean())
        print(f"[INFO] Energy computed by integrating SYSTEM_POWER over time: {energy_j:.2f} J")
        return energy_j

    print("[WARNING] No energy-related columns found in EnergiBridge log.")
    return 0.0

def summarize_energy_profile(eb_log: pd.DataFrame) -> dict:
    """
    Produce a quick summary of the energy and system profile.
    """
    total_energy = compute_energy_from_log(eb_log)
    
    # Calculate runtime from Delta column (more accurate than Time stamps)
    if "Delta" in eb_log.columns:
        total_time = eb_log["Delta"].sum() / 1000.0  # Convert ms to seconds
    elif "Time" in eb_log.columns:
        total_time = (eb_log["Time"].iloc[-1] - eb_log["Time"].iloc[0]) / 1000.0
    else:
        total_time = None

    summary = {
        "total_energy_j": float(total_energy),
        "runtime_s": float(total_time) if total_time else 0.0,
        "avg_power_w": float(total_energy / total_time) if total_time and total_time > 0 else 0.0,
    }

    print(f"[INFO] Energy summary: {summary}")
    return summary