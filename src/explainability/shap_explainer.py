import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "models", "processed_data.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

device = torch.device("cpu")

data = joblib.load(DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

X_train = data["X_train"].astype(np.float32)
X_test = data["X_test"].astype(np.float32)

feature_names = list(
    preprocessor.get_feature_names_out()
)

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = MLP(X_train.shape[1]).to(device)

checkpoint = torch.load(
    os.path.join(BASE_DIR, "models", "mlp_model.pth"),
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

background_size = min(100, len(X_train))
explanation_size = min(100, len(X_test))

rng = np.random.RandomState(42)

background = X_train[
    rng.choice(
        len(X_train),
        background_size,
        replace=False
    )
]

samples = X_test[:explanation_size]

def predict(X):
    X_tensor = torch.tensor(
        X.astype(np.float32)
    )

    with torch.no_grad():
        output = model(X_tensor)
        return torch.sigmoid(output).numpy()

explainer = shap.Explainer(
    predict,
    background,
    feature_names=feature_names
)

shap_values = explainer(samples)

values = shap_values.values

if values.ndim == 3:
    values = values[:, :, 0]

mean_absolute_shap = np.mean(
    np.abs(values),
    axis=0
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean Absolute SHAP": mean_absolute_shap
})

importance = importance.sort_values(
    "Mean Absolute SHAP",
    ascending=False
)

importance.to_csv(
    os.path.join(
        REPORTS_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)

plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"][::-1],
    importance["Mean Absolute SHAP"][::-1]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Global SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "shap_global_importance.png"
    ),
    dpi=300
)

plt.close()

sample_index = 0

sample = samples[
    sample_index:sample_index + 1
]

sample_shap = values[
    sample_index
]

prediction_probability = float(
    predict(sample)[0][0]
)

local_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": sample_shap,
    "Feature Value": sample[0]
})

local_explanation["Absolute SHAP"] = (
    local_explanation["SHAP Value"].abs()
)

local_explanation = local_explanation.sort_values(
    "Absolute SHAP",
    ascending=False
)

local_explanation.to_csv(
    os.path.join(
        REPORTS_DIR,
        "shap_local_explanation.csv"
    ),
    index=False
)

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

print(f"\nProcessed features: {len(feature_names)}")
print(f"Explained samples: {explanation_size}")
print(
    f"Sample prediction probability: "
    f"{prediction_probability:.4f}"
)

print("\nProcessed Feature Names:")

for feature in feature_names:
    print(f"- {feature}")

print("\nGlobal Feature Importance:")

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nLocal Explanation:")

print(
    local_explanation.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved:")
print("- reports/shap_feature_importance.csv")
print("- reports/shap_global_importance.png")
print("- reports/shap_local_explanation.csv")