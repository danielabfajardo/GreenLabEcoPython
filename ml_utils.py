# ml_utils.py
import joblib
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

MODEL_SAVE_PATH = "trained_model.pkl"

# This function prepares the medical insurance dataset for machine learning experiments.
# It creates three dataset sizes (small, medium, large) with different numbers of features and rows, using sampling with or without replacement.
# For each size, it encodes categorical variables, handles missing values, selects the appropriate features, and splits the data into train and test sets.
# The processed datasets are saved for use in downstream training and evaluation.
def preprocess_datasets(
    source_csv="GreenLabEcoPython/datasets/bcsc_risk_factors_summarized1_092020.csv",
    out_dir="GreenLabEcoPython/datasets"
):
    """
    Preprocess the medical insurance dataset into small, medium, and large sizes.
    Each size has training and testing splits.
    - small: 1x full dataset, ~15 features
    - medium: 2.5x full dataset, ~30 features
    - large: 5x full dataset, all features
    """
    data_path = Path("GreenLabEcoPython/datasets/medical_insurance.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Please ensure the medical_insurance.csv file is in the datasets folder.")
    
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Create target variables
    # For logistic regression: use existing cancer_history column, rename to breast_cancer_history
    if 'cancer_history' in df.columns:
        df['breast_cancer_history'] = df['cancer_history']
    else:
        raise ValueError("Dataset must contain 'cancer_history' column")
    
    # For linear regression: use existing claims_count, rename to count
    if 'claims_count' in df.columns:
        df['count'] = df['claims_count']
    else:
        raise ValueError("Dataset must contain 'claims_count' column")
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['breast_cancer_history', 'count']]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Remove unnecessary columns that shouldn't be features
    columns_to_drop = ['person_id', 'cancer_history', 'claims_count']  # Keep derived targets
    for col in columns_to_drop:
        if col in df.columns and col not in ['breast_cancer_history', 'count']:
            df = df.drop(columns=[col])
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    total_rows = len(df)
    sizes = {
        'small': total_rows,
        'medium': int(total_rows * 2.5),
        'large': int(total_rows * 5)
    }
    # Feature selection for each size
    all_features = [col for col in df.columns if col not in ['breast_cancer_history', 'count']]
    small_features = all_features[:13] + ['breast_cancer_history', 'count']  # 13 + 2 targets = 15 columns
    medium_features = all_features[:28] + ['breast_cancer_history', 'count'] # 28 + 2 targets = 30 columns
    large_features = df.columns.tolist()  # all columns
    
    feature_sets = {
        'small': small_features,
        'medium': medium_features,
        'large': large_features
    }
    
    output_dir = Path("GreenLabEcoPython/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for size_name, num_rows in sizes.items():
        features = feature_sets[size_name]
        df_selected = df[features].copy()
        if num_rows > total_rows:
            df_sampled = df_selected.sample(n=num_rows, replace=True, random_state=42).reset_index(drop=True)
        else:
            df_sampled = df_selected.sample(n=num_rows, replace=False, random_state=42).reset_index(drop=True)
        
        # Split into train and test (50/50) with stratification on breast_cancer_history if possible
        try:
            train_df, test_df = train_test_split(
                df_sampled, 
                test_size=0.5, 
                random_state=42,
                stratify=df_sampled['breast_cancer_history']
            )
        except ValueError:
            train_df, test_df = train_test_split(
                df_sampled, 
                test_size=0.5, 
                random_state=42
            )
        
        # Save the datasets
        train_path = output_dir / f"data_{size_name}_train.csv"
        test_path = output_dir / f"data_{size_name}_test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"[INFO] Created {size_name} dataset:")
        print(f"  Features: {len(features)}")
        print(f"  Train: {train_path} ({len(train_df)} rows)")
        print(f"  Test: {test_path} ({len(test_df)} rows)")
        print(f"  Breast cancer history distribution: {train_df['breast_cancer_history'].value_counts().to_dict()}")
        print(f"  Claims count (mean): {train_df['count'].mean():.2f}")

    print("\n[SUCCESS] Dataset preprocessing complete!")
    print(f"[INFO] Columns in small: {feature_sets['small']}")
    print(f"[INFO] Columns in medium: {feature_sets['medium']}")
    print(f"[INFO] Columns in large: {feature_sets['large']}")

# This function trains a machine learning model using the provided training dataset.
# It selects the appropriate algorithm and hyperparameters, fits the model to the data, and evaluates performance metrics.
# The trained model is then saved for later inference and analysis.
def train_model(algo, lib, X_train, y_train):
    """Train and return model, possibly with scaler."""
    if algo == "logistic":
        if lib == "sklearn":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)
            return (model, scaler)
        else:  # statsmodels logistic
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_sm = sm.add_constant(X_train_scaled)
            model = sm.GLM(y_train, X_sm, family=sm.families.Binomial()).fit()
            return (model, scaler)
    else:  # linear regression
        if lib == "sklearn":
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        else:  # statsmodels linear
            X_sm = sm.add_constant(X_train, has_constant='add')
            model = sm.OLS(y_train, X_sm).fit()
            return model

# This function loads a trained machine learning model and applies it to the test dataset for inference.
# It generates predictions, computes relevant evaluation metrics, and returns the results for analysis.
# This step is essential for assessing model performance on unseen data.
def inference_model(algo, lib, model, X_test, y_test):
    """Perform inference and return metrics."""
    if algo == "logistic":
        if lib == "sklearn":
            model, scaler = model
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')
        else:
            model, scaler = model
            X_test_scaled = scaler.transform(X_test)
            X_test_sm = sm.add_constant(X_test_scaled, has_constant='add')
            y_pred = (model.predict(X_test_sm) >= 0.5).astype(int)
            return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')
    else:
        if lib == "sklearn":
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)
        else:
            X_test_sm = sm.add_constant(X_test, has_constant='add')
            y_pred = model.predict(X_test_sm)
            return r2_score(y_test, y_pred)


def save_model(model, path):
    """Persist trained model (and scaler, if any)."""
    joblib.dump(model, path)


def load_model(path):
    """Load model from disk."""
    return joblib.load(path)

if __name__ == "__main__":
    preprocess_datasets()
