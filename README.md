# Fraud Detection MLOps Pipeline

End-to-end ML pipeline for credit card fraud detection with full MLOps tooling.
Covers: data versioning, experiment tracking, explainability, drift monitoring, API serving, CI/CD, and Docker deployment.

## Problem

Detect fraudulent credit card transactions in a highly imbalanced dataset.
- 284,807 transactions
- Only 492 fraud cases (0.17%)
- Accuracy is meaningless — predicting everything as normal gives 99.83% accuracy but catches zero fraud
- Solution: use PR-AUC as primary metric which directly measures performance on the minority class

## Architecture

```
creditcard.csv → DVC (data versioning) → Feature Engineering → Train/Test Split → SMOTE
                                                                                      ↓
                                                                              Model Training
                                                                         (LR, RF, XGBoost)
                                                                                      ↓
                                                                          MLflow Experiment
                                                                             Tracking
                                                                                      ↓
                                                                        MLflow Model Registry
                                                                                      ↓
                                                                    FastAPI (/predict, /health)
                                                                                      ↓
                                                                    Evidently Drift Monitoring
```

## Results

| Model | PR-AUC | Fraud Recall | Fraud Precision |
|---|---|---|---|
| XGBoost | 0.8698 | 0.87 | 0.70 |
| Random Forest | 0.8578 | 0.81 | 0.85 |
| Logistic Regression | 0.7201 | 0.92 | 0.06 |

XGBoost selected as production model — highest PR-AUC across all experiments.

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| XGBoost | Best performing model |
| scikit-learn | LR, RF, preprocessing |
| imbalanced-learn | SMOTE for class imbalance |
| MLflow | Experiment tracking + Model Registry |
| DVC | Data versioning |
| SHAP | Model explainability |
| Evidently | Drift monitoring |
| FastAPI | Model serving |
| Docker | Containerization |
| GitHub Actions | CI/CD pipeline |
| pytest | Unit testing |

## Project Structure

```
fraud-detection-mlops/
├── data/
│   ├── raw/                          # Original data (tracked by DVC)
│   └── processed/                    # Feature engineered splits (tracked by DVC)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering + SMOTE
│   └── 03_model_experiments.ipynb    # Model training + SHAP
├── src/
│   ├── monitoring/
│   │   └── drift.py                  # Evidently drift report
│   └── data/
├── api/
│   └── main.py                       # FastAPI app
├── tests/
│   └── test_features.py              # pytest unit tests
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## How to Run

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/yourusername/fraud-detection-mlops
cd fraud-detection-mlops
docker-compose up
```

- API: http://localhost:8000
- MLflow UI: http://localhost:5000

### Option 2 — Local Setup

```bash
# Clone repo
git clone https://github.com/yourusername/fraud-detection-mlops
cd fraud-detection-mlops

# Install dependencies
pip install -r requirements.txt

# Pull data from DVC
dvc pull

# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict fraud
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount_log": 4.5,
    "Time_norm": 0.3
  }'
```

### Run Tests

```bash
pytest tests/ -v
```

### Generate Drift Report

```bash
python src/monitoring/drift.py
# Report saved to reports/drift_report.html
```

## Key Design Decisions

**Why PR-AUC over accuracy?**
With 0.17% fraud, a model predicting everything as normal gets 99.83% accuracy but catches zero fraud. PR-AUC measures performance directly on the minority class. A model predicting all normal gets PR-AUC of 0.0017 — immediately exposed.

**Why SMOTE after train/test split?**
Applying SMOTE before splitting causes data leakage — synthetic samples created from test-set neighbors leak information into training data, inflating metrics. Always split first, SMOTE second.

**Why log1p on Amount?**
Amount has mean $88 but max $25,519 — classic right skew. log1p compresses the range from 0-25,519 to 0-10. Used log1p not log because some transactions have $0 amount and log(0) is undefined.

**Why XGBoost?**
Highest PR-AUC (0.8698) across all three models. Also catches more fraud (recall 0.87) than Random Forest (0.81) which matters more than precision in fraud detection — missing real fraud is worse than a false alarm.

**Why SHAP?**
XGBoost is a black box. SHAP explains why each prediction was made. V14 confirmed as top fraud predictor — low V14 values push strongly toward fraud prediction, consistent with EDA findings.

**Why Evidently?**
Fraud patterns change over time as fraudsters adapt. Evidently monitors whether incoming data distribution has shifted from training data. If more than 30% of features drift, model retraining should be triggered.

## Interview Q&A

**Why PR-AUC not ROC-AUC?**
ROC-AUC looks at TPR vs FPR. With 99.8% negatives, even a bad model gets high ROC-AUC because FPR stays low. PR-AUC looks at precision-recall tradeoff — directly sensitive to minority class performance.

**What is SMOTE?**
Synthetic Minority Over-sampling Technique. For each fraud case, finds K nearest neighbors in feature space and creates synthetic samples along line segments between them. Risk: if applied before split, synthetic samples leak test information into training.

**What does SHAP tell you?**
V14 is the most important feature. Low V14 values push the model strongly toward fraud prediction. This is consistent with EDA where fraud transactions showed V14 shifted to -5 to -7 range.

**How would you handle model drift in production?**
Monitor feature distributions using KS test comparing training vs production data. If drift share exceeds 30%, trigger retraining pipeline. Also monitor prediction distribution and actual performance metrics when labels become available via chargeback reports.

