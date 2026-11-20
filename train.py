from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
ARTIFACTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

def load_dataset():
    """Load the local UCI CSV; otherwise retrieve dataset 601 from UCI."""
    if DATA_PATH.exists():
        print(f"Loading local dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Local CSV not found. Downloading dataset 601 from UCI...")
        dataset = fetch_ucirepo(id=601)
        df = pd.concat(
            [dataset.data.features, dataset.data.targets],
            axis=1
        )
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to: {DATA_PATH}")
    return df

def make_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

def main():
    df = load_dataset()

    print("\n========== DATA AUDIT ==========")
    print("Shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nTarget distribution:\n", df[TARGET].value_counts(dropna=False))
    print("\nTarget percentages:\n", df[TARGET].value_counts(normalize=True) * 100)

    # Keep only legitimate prediction-time features and the target.
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # EDA figures
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Machine Failure Class Distribution")
    plt.xlabel("Machine failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.heatmap(df[NUMERIC_FEATURES + [TARGET]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Split before fitting any preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Train failure rate:", round(y_train.mean() * 100, 2), "%")
    print("Test failure rate:", round(y_test.mean() * 100, 2), "%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "average_precision": "average_precision",
    }

    models = {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
    }

    print("\n========== STRATIFIED 5-FOLD VALIDATION ==========")
    rows = []
    pipelines = {}
    for name, estimator in models.items():
        pipe = make_pipeline(estimator)
        pipelines[name] = pipe
        result = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False
        )
        row = {"Model": name}
        for metric in scoring:
            row[metric] = result[f"test_{metric}"].mean()
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in row.items() if k != "Model"})

    comparison = pd.DataFrame(rows).sort_values(
        "average_precision", ascending=False
    )
    comparison.to_csv(OUTPUTS / "cv_model_comparison.csv", index=False)
    print("\nCV comparison:\n", comparison.to_string(index=False))

    # Small, justified tuning search on training data only.
    rf_pipeline = pipelines["Random Forest"]
    param_distributions = {
        "classifier__n_estimators": [150, 250, 350],
        "classifier__max_depth": [None, 8, 12, 18],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", 0.8],
    }
    search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    print("\n========== RANDOM FOREST TUNING ==========")
    search.fit(X_train, y_train)
    tuned_rf = search.best_estimator_
    print("Best CV PR-AUC:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

    # Compare tuned RF CV score with Logistic Regression CV score.
    logistic_cv_ap = comparison.loc[
        comparison["Model"] == "Logistic Regression", "average_precision"
    ].iloc[0]
    if search.best_score_ >= logistic_cv_ap:
        selected_name = "Tuned Random Forest"
        selected_pipeline = tuned_rf
    else:
        selected_name = "Logistic Regression"
        selected_pipeline = pipelines["Logistic Regression"]

    # Select a probability threshold using out-of-fold training predictions.
    print("\n========== THRESHOLD SELECTION ==========")
    oof_probabilities = cross_val_predict(
        selected_pipeline, X_train, y_train, cv=3,
        method="predict_proba", n_jobs=-1
    )[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(
        y_train, oof_probabilities
    )
    if len(thresholds):
        f1_values = (
            2 * precisions[:-1] * recalls[:-1] /
            (precisions[:-1] + recalls[:-1] + 1e-12)
        )
        best_index = int(np.argmax(f1_values))
        chosen_threshold = float(thresholds[best_index])
    else:
        chosen_threshold = 0.5

    print("Selected model:", selected_name)
    print("Selected threshold:", round(chosen_threshold, 4))

    # Fit final pipeline on training data. Holdout remains untouched until now.
    selected_pipeline.fit(X_train, y_train)
    test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= chosen_threshold).astype(int)

    print("\n========== FINAL UNTOUCHED HOLDOUT RESULTS ==========")
    metrics = {
        "selected_model": selected_name,
        "threshold": chosen_threshold,
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, test_probabilities)),
    }
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(
        y_test, test_predictions, zero_division=0
    ))

    with open(OUTPUTS / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"])
    plt.title("Holdout Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)
    plt.close()

    # Permutation importance on training data; not used to tune or select the model.
    print("\n========== PERMUTATION IMPORTANCE ==========")
    importance = permutation_importance(
        selected_pipeline, X_train, y_train,
        n_repeats=5, random_state=42,
        scoring="average_precision", n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance.importances_mean
    }).sort_values("Importance", ascending=False)
    importance_df.to_csv(OUTPUTS / "permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Permutation Feature Importance (Training Data)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150)
    plt.close()

    # Save complete preprocessing + classifier pipeline and threshold metadata.
    joblib.dump(selected_pipeline, ARTIFACTS / "predictops_pipeline.pkl")
    metadata = {
        "model_name": selected_name,
        "threshold": chosen_threshold,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "random_state": 42,
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI ID 601)"
    }
    with open(ARTIFACTS / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining completed successfully.")
    print("Saved model:", ARTIFACTS / "predictops_pipeline.pkl")
    print("Saved metadata:", ARTIFACTS / "model_metadata.json")
    print("Saved charts and reports in:", OUTPUTS)

if __name__ == "__main__":
    main()
