# ml_utils.py
import joblib
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
import numpy as np

MODEL_SAVE_PATH = "trained_model.pkl"

def train_model(algo, lib, X_train, y_train):
    """Train and return model, possibly with scaler."""
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
            return (model, scaler)
    else:  # linear regression
        if lib == "sklearn":
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        else:  # statsmodels linear
            X_sm = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_sm).fit()
            return model


def inference_model(algo, lib, model, X_test, y_test):
    """Perform inference and return metrics."""
    if algo == "logistic":
        if lib == "sklearn":
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
        else:
            model, scaler = model
            X_test_scaled = scaler.transform(X_test)
            X_test_sm = sm.add_constant(X_test_scaled)
            y_pred = (model.predict(X_test_sm) >= 0.5).astype(int)
            return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
    else:
        if lib == "sklearn":
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)
        else:
            X_test_sm = sm.add_constant(X_test)
            y_pred = model.predict(X_test_sm)
            return r2_score(y_test, y_pred)


def save_model(model, path):
    """Persist trained model (and scaler, if any)."""
    joblib.dump(model, path)


def load_model(path):
    """Load model from disk."""
    return joblib.load(path)
