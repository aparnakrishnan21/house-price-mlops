import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os

# Reference dataset (training/processed data)
reference_data = pd.read_csv(
    "data/processed/processed_data.csv"
)

current_data = reference_data.sample(100)
#print(reference_data.head())
#print(current_data.head())
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result=report.run(
    reference_data=reference_data,
    current_data=current_data
)
os.makedirs("reports", exist_ok=True)
result.save_html(
    "reports/drift_report.html"
)
print("Drift report generated successfully!")