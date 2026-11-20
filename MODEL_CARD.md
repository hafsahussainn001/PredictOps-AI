# Model Card — PredictOps AI

## Model details
- Project: PredictOps AI — Predictive Maintenance & Explainable ML
- Model type: Binary classification pipeline
- Final estimator: Fill in the selected model printed by `train.py`
- Version/date: Fill in after final training

## Intended use
Educational predictive-maintenance demonstration using machine sensor readings. It can be used to demonstrate how a model estimates machine-failure risk and how predictions can be exposed through an inference interface.

## Out-of-scope use
Do not use this prototype as the sole basis for safety-critical decisions, automatic machine shutdown, or maintenance guarantees.

## Dataset
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, ID 601. It is a synthetic dataset of 10,000 records. DOI: https://doi.org/10.24432/C5HS5C. Licence: CC BY 4.0.

## Inputs
- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Target: `Machine failure` (0 = no failure, 1 = failure).

## Data and leakage decisions
Identifiers `UDI` and `Product ID` are excluded. Failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they encode component failure modes used to form the overall target. Preprocessing is fitted inside a scikit-learn pipeline after the train/test split.

## Evaluation
Copy the actual holdout accuracy, precision, recall, F1, PR-AUC, selected threshold, and confusion matrix from the generated files. Do not invent or estimate metrics.

## Limitations and risks
- The dataset is synthetic and may not represent real industrial operating conditions.
- Failure cases are a minority class; accuracy alone is insufficient.
- False negatives may mean a real failure risk is not flagged. False positives may cause unnecessary inspections.
- Probability estimates and threshold decisions are model-dependent and are not certainty.
- Feature importance describes model reliance, not causation.
- Distribution shift, sensor faults, and missing or invalid inputs can reduce reliability.

## Human oversight
A qualified maintenance professional must review alerts and make operational decisions. The system is a learning prototype, not a certified industrial safety system.

## Reproducibility
Random seed: 42. Record the Python and dependency versions used. The saved artifact includes preprocessing and the fitted estimator.
