import argparse
import json
from pathlib import Path
import pandas as pd

from GreenLabEcoPython.ml_utils import train_model, save_model

ALGO = "logistic"
LIB = "statsmodels"

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["breast_cancer_history"].values
    X = df.drop(columns=["breast_cancer_history"]).values
    print(f"Loaded {csv_path}: X shape {X.shape}, y shape {y.shape}")
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="small|medium|large")
    parser.add_argument('--run_id', required=False, help='Unique run identifier')
    args = parser.parse_args()

    train_path = Path(f"GreenLabEcoPython/datasets/data_{args.dataset}_train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found")

    X_train, y_train = load_xy(train_path)
    model = train_model(ALGO, LIB, X_train, y_train)

    model_path = Path(f"trained_{ALGO}_{LIB}_{args.dataset}.pkl")
    save_model(model, model_path)

if __name__ == "__main__":
    main()
