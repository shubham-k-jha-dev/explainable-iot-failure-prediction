import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

files = {
    "MLP": "mlp_metrics.csv",
    "1D-CNN": "cnn_metrics.csv",
    "LSTM": "lstm_metrics.csv"
}

rows = []

for model_name, filename in files.items():
    path = os.path.join(REPORTS_DIR, filename)
    df = pd.read_csv(path)

    metrics = dict(zip(df["Metric"], df["Value"]))

    rows.append({
        "Model": model_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1 Score": metrics["F1 Score"],
        "ROC-AUC": metrics["ROC-AUC"]
    })

comparison = pd.DataFrame(rows)

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

output_path = os.path.join(
    REPORTS_DIR,
    "model_comparison.csv"
)

comparison.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("MODEL RANKING BY F1 SCORE")
print("=" * 75)

for index, row in comparison.iterrows():
    print(
        f"{index + 1}. {row['Model']} "
        f"| F1: {row['F1 Score']:.4f} "
        f"| Recall: {row['Recall']:.4f} "
        f"| ROC-AUC: {row['ROC-AUC']:.4f}"
    )

print("\nSaved:")
print("- reports/model_comparison.csv")