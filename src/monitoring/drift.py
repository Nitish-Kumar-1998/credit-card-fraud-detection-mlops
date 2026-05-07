import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import os

def generate_drift_report():
    # Load reference data (training) and current data (test)
    reference = pd.read_csv('data/processed/X_train.csv').sample(1000, random_state=42)
    current   = pd.read_csv('data/processed/X_test.csv').sample(500, random_state=42)

    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    # Run report
    report.run(reference_data=reference, current_data=current)

    # Save report
    os.makedirs('reports', exist_ok=True)
    report.save_html('reports/drift_report.html')
    print("Drift report saved to reports/drift_report.html")

if __name__ == "__main__":
    generate_drift_report()


    