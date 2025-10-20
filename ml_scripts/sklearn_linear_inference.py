import argparse
import csv
import json
from pathlib import Path
import pandas as pd

from GreenLabEcoPython.ml_utils import inference_model, load_model

ALGO = "linear"
LIB = "sklearn"

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["count"].values
    X = df.drop(columns=["count"]).values
    print(f"Loaded {csv_path}: X shape {X.shape}, y shape {y.shape}")
    return X, y

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="small|medium|large")
    parser.add_argument("--run_id", required=False, help="experiment run id")
    args = parser.parse_args()

    test_path = Path(f"GreenLabEcoPython/datasets/data_{args.dataset}_test.csv")
    model_path = Path(f"trained_{ALGO}_{LIB}_{args.dataset}.pkl")

    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    X_test, y_test = load_xy(test_path)
    model = load_model(model_path)
    r2 = inference_model(ALGO, LIB, model, X_test, y_test)

    out_csv = Path("GreenLabEcoPython/inference_results.csv")
    fieldnames = ["__run_id", "accuracy", "f1_score", "r2", "dataset"]
    row = {
        "__run_id": args.run_id or f"{ALGO}_{LIB}_{args.dataset}_inference",
        "accuracy": "",
        "f1_score": "",
        "r2": r2,
        "dataset": X_test.shape
    }

    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    run()
