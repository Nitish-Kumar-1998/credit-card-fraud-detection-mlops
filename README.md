# Fraud Detection MLOps Pipeline

A production-grade ML pipeline for credit card fraud detection with full MLOps tooling.

## Tech Stack
- **Model**: XGBoost, RandomForest
- **Experiment Tracking**: MLflow
- **Explainability**: SHAP
- **Drift Monitoring**: Evidently
- **API**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## Project Structure
- `src/data/` — data loading and feature engineering
- `src/models/` — training, evaluation, prediction
- `src/monitoring/` — drift detection and retraining
- `src/explainability/` — SHAP visualizations
- `api/` — FastAPI serving layer
- `tests/` — unit tests

## Key Design Decisions
- **Primary metric**: PR-AUC (not accuracy or ROC-AUC)
- **Class imbalance**: SMOTE applied after train/test split
- **Every experiment tracked**: MLflow, no exceptions

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Results
*To be updated after training*