# PredictOps AI — Predictive Maintenance & Explainable ML

## Project objective
Predict the risk of machine failure from operational sensor readings using a reproducible tabular machine-learning pipeline. The application is a decision-support prototype, not a guaranteed failure diagnosis.

## Dataset
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, dataset ID 601. The dataset contains 10,000 records and is synthetic. Dataset citation:

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Licence: CC BY 4.0. Cite the dataset when redistributing or adapting it.

## Features and leakage prevention
The model uses Type, Air temperature, Process temperature, Rotational speed, Torque, and Tool wear. UDI and Product ID are identifiers and are excluded. TWF, HDF, PWF, OSF, and RNF are excluded because they describe component failure modes used to construct the overall Machine failure target. The target is `Machine failure`.

## Models and evaluation
- DummyClassifier prior baseline
- Class-weighted Logistic Regression
- Class-weighted Random Forest
- Stratified 5-fold cross-validation
- RandomizedSearchCV for a small Random Forest search
- Precision, recall, F1 and average precision (PR-AUC)
- Confusion matrix on the untouched holdout set
- Permutation feature importance

The final model is selected using training-set cross-validation average precision. A decision threshold is selected using out-of-fold training predictions. The test set is held out from preprocessing fitting, tuning, and threshold selection.

## Setup
```bash
python -m pip install -r requirements.txt
```

Place `ai4i2020.csv` in `data/`. If it is not present, `train.py` attempts to retrieve UCI dataset 601 using `ucimlrepo`.

## Train
```bash
python train.py
```

The script saves the complete preprocessing/model pipeline and metadata in `artifacts/`, and charts/reports in `outputs/`.

## Run the web application
```bash
streamlit run app.py
```

## Repository contents
- `train.py`: data audit, training, cross-validation, tuning, evaluation and explainability
- `app.py`: Streamlit inference interface
- `artifacts/`: saved pipeline and metadata (generated after training)
- `outputs/`: charts and evaluation reports (generated after training)
- `MODEL_CARD.md`: model documentation

## Limitations
- The AI4I dataset is synthetic and may not represent every real factory, machine, sensor, or operating environment.
- Failure events are rare, so accuracy alone can be misleading.
- A predicted probability is not certainty and should not be treated as a confirmed diagnosis.
- The model is not validated for safety-critical or autonomous shutdown decisions.
- Performance can change when deployed on different machines or sensor distributions.

## Reproducibility
A fixed random seed of 42 is used for splitting and model procedures. Preprocessing is inside the pipeline and is fitted on training folds only. Record the Python and package versions used for the final run.
