"""
energy_utils.py

Utility functions for parsing and computing energy usage from EnergiBridge logs.
Designed to handle variable column names across hardware platforms.
"""

import pandas as pd

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

    # As a last resort: look for any ENERGY-like columns
    energy_cols = [c for c in eb_log.columns if "ENERGY" in c]
    if energy_cols:
        energy_j = sum(eb_log[c].iloc[-1] - eb_log[c].iloc[0] for c in energy_cols)
        print(f"[INFO] Energy computed using fallback columns: {', '.join(energy_cols)}")
        return energy_j

    print("[WARNING] No energy-related columns found in EnergiBridge log.")
    return 0.0


def summarize_energy_profile(eb_log: pd.DataFrame) -> dict:
    """
    Produce a quick summary of the energy and system profile.
    """
    total_energy = compute_energy_from_log(eb_log)
    total_time = (
        eb_log["Time"].iloc[-1] - eb_log["Time"].iloc[0]
        if "Time" in eb_log.columns
        else None
    )

    summary = {
        "total_energy_j": total_energy,
        "runtime_s": total_time,
        "avg_power_w": total_energy / total_time if total_time else None,
    }

    print(f"[INFO] Energy summary: {summary}")
    return summary
