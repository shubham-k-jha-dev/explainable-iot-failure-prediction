import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import roc_curve
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    ExplanationItem,
    HealthResponse,
    SweepRequest,
    SweepResponse
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_model.pth")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
DATA_PATH = os.path.join(BASE_DIR, "models", "processed_data.joblib")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

device = torch.device("cpu")

data = joblib.load(DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

X_train = data["X_train"].astype(np.float32)

feature_names = list(
    preprocessor.get_feature_names_out()
)

display_names = {
    "num__Air temperature [K]": "Air Temperature",
    "num__Process temperature [K]": "Process Temperature",
    "num__Rotational speed [rpm]": "Rotational Speed",
    "num__Torque [Nm]": "Torque",
    "num__Tool wear [min]": "Tool Wear",
    "cat__Type_H": "Machine Type H",
    "cat__Type_L": "Machine Type L",
    "cat__Type_M": "Machine Type M"
}


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
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

background_size = min(100, len(X_train))

rng = np.random.RandomState(42)

background = X_train[
    rng.choice(
        len(X_train),
        background_size,
        replace=False
    )
]


def predict_probability(X):
    X = np.asarray(
        X,
        dtype=np.float32
    )

    tensor = torch.tensor(X)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    return probabilities.numpy()


explainer = shap.Explainer(
    predict_probability,
    background,
    feature_names=feature_names
)


app = FastAPI(
    title="Explainable IoT Sensor Failure Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/static/reports",
    StaticFiles(directory=REPORTS_DIR),
    name="reports"
)


@app.get(
    "/health",
    response_model=HealthResponse
)
def health():
    return {
        "status": "healthy",
        "model": "MLP",
        "device": str(device)
    }


def preprocess_input(request):
    df = pd.DataFrame([{
        "Type": request.type,
        "Air temperature [K]": request.air_temperature,
        "Process temperature [K]": request.process_temperature,
        "Rotational speed [rpm]": request.rotational_speed,
        "Torque [Nm]": request.torque,
        "Tool wear [min]": request.tool_wear
    }])

    return preprocessor.transform(df).astype(
        np.float32
    )


def make_prediction(processed_input):
    probability = float(
        predict_probability(processed_input)[0][0]
    )

    prediction = int(
        probability >= 0.5
    )

    result = (
        "Likely Failure"
        if prediction == 1
        else "No Failure"
    )

    confidence = (
        probability
        if prediction == 1
        else 1 - probability
    )

    return (
        prediction,
        result,
        probability,
        confidence
    )


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    processed_input = preprocess_input(request)

    prediction, result, probability, confidence = (
        make_prediction(processed_input)
    )

    return {
        "prediction": prediction,
        "result": result,
        "failure_probability": round(
            probability,
            6
        ),
        "confidence": round(
            confidence,
            6
        )
    }


@app.post(
    "/explain",
    response_model=ExplanationResponse
)
def explain(request: PredictionRequest):
    processed_input = preprocess_input(request)

    prediction, result, probability, confidence = (
        make_prediction(processed_input)
    )

    shap_values = explainer(
        processed_input
    )

    values = shap_values.values

    if values.ndim == 3:
        values = values[:, :, 0]

    values = values[0]

    explanations = []

    for feature, shap_value in zip(
        feature_names,
        values
    ):
        explanations.append(
            ExplanationItem(
                feature=display_names.get(
                    feature,
                    feature
                ),
                shap_value=round(
                    float(shap_value),
                    6
                ),
                direction=(
                    "increases failure risk"
                    if shap_value > 0
                    else "decreases failure risk"
                )
            )
        )

    explanations.sort(
        key=lambda item: abs(item.shap_value),
        reverse=True
    )

    return {
        "prediction": prediction,
        "result": result,
        "failure_probability": round(
            probability,
            6
        ),
        "explanations": explanations
    }


# ---------------------------------------------------------------------------
# Dashboard endpoints (read-only views over the saved artifacts)
# ---------------------------------------------------------------------------

raw_df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")

numeric_columns = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]"
}

failure_modes = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure"
}

model_files = {
    "MLP": "mlp",
    "LSTM": "lstm",
    "1D-CNN": "cnn"
}

y_test = np.asarray(data["y_test"]).ravel().astype(int)


@app.get("/metrics")
def metrics():
    comparison = pd.read_csv(
        os.path.join(REPORTS_DIR, "model_comparison.csv")
    )

    models = []

    for _, row in comparison.iterrows():
        name = row["Model"]

        probabilities = np.asarray(
            joblib.load(
                os.path.join(
                    MODELS_DIR,
                    f"{model_files[name]}_test_probabilities.joblib"
                )
            )
        ).ravel()

        fpr, tpr, _ = roc_curve(y_test, probabilities)

        # Downsample the ROC curve so the payload stays small
        step = max(1, len(fpr) // 120)
        roc = [
            {"fpr": round(float(x), 4), "tpr": round(float(y), 4)}
            for x, y in zip(fpr[::step], tpr[::step])
        ]
        roc.append({"fpr": 1.0, "tpr": 1.0})

        models.append({
            "name": name,
            "key": model_files[name],
            "accuracy": float(row["Accuracy"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1 Score"]),
            "roc_auc": float(row["ROC-AUC"]),
            "probabilities": [
                round(float(p), 5) for p in probabilities
            ],
            "roc": roc
        })

    return {
        "labels": y_test.tolist(),
        "models": models
    }


@app.get("/shap/global")
def shap_global():
    importance = pd.read_csv(
        os.path.join(REPORTS_DIR, "shap_feature_importance.csv")
    )

    local = pd.read_csv(
        os.path.join(REPORTS_DIR, "shap_local_explanation.csv")
    )

    return {
        "global": [
            {
                "feature": display_names.get(row["Feature"], row["Feature"]),
                "importance": float(row["Mean Absolute SHAP"])
            }
            for _, row in importance.iterrows()
        ],
        "local": [
            {
                "feature": display_names.get(row["Feature"], row["Feature"]),
                "shap_value": float(row["SHAP Value"]),
                "feature_value": float(row["Feature Value"])
            }
            for _, row in local.iterrows()
        ]
    }


@app.get("/dataset/summary")
def dataset_summary():
    failed = raw_df[raw_df["Machine failure"] == 1]
    healthy = raw_df[raw_df["Machine failure"] == 0]

    features = []

    for key, column in numeric_columns.items():
        features.append({
            "key": key,
            "column": column,
            "min": float(raw_df[column].min()),
            "max": float(raw_df[column].max()),
            "mean": round(float(raw_df[column].mean()), 3),
            "std": round(float(raw_df[column].std()), 3),
            "mean_failure": round(float(failed[column].mean()), 3),
            "mean_healthy": round(float(healthy[column].mean()), 3)
        })

    types = []

    for machine_type in ["L", "M", "H"]:
        subset = raw_df[raw_df["Type"] == machine_type]
        types.append({
            "type": machine_type,
            "count": int(len(subset)),
            "failures": int(subset["Machine failure"].sum()),
            "failure_rate": round(
                float(subset["Machine failure"].mean()) * 100,
                2
            )
        })

    return {
        "rows": int(len(raw_df)),
        "columns": int(raw_df.shape[1]),
        "missing_values": int(raw_df.isna().sum().sum()),
        "failures": int(len(failed)),
        "healthy": int(len(healthy)),
        "failure_rate": round(len(failed) / len(raw_df) * 100, 2),
        "features": features,
        "types": types,
        "failure_modes": [
            {
                "code": code,
                "name": name,
                "count": int(raw_df[code].sum())
            }
            for code, name in failure_modes.items()
        ]
    }


@app.get("/dataset/sample")
def dataset_sample(
    failure: int | None = Query(default=None, ge=0, le=1)
):
    subset = raw_df

    if failure is not None:
        subset = raw_df[raw_df["Machine failure"] == failure]

    row = subset.sample(1).iloc[0]

    return {
        "udi": int(row["UDI"]),
        "product_id": row["Product ID"],
        "actual_failure": int(row["Machine failure"]),
        "failure_modes": [
            code for code in failure_modes if int(row[code]) == 1
        ],
        "input": {
            "type": row["Type"],
            **{
                key: float(row[column])
                for key, column in numeric_columns.items()
            }
        }
    }


@app.post(
    "/sweep",
    response_model=SweepResponse
)
def sweep(request: SweepRequest):
    if request.feature not in numeric_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature '{request.feature}'"
        )

    column = numeric_columns[request.feature]
    values = np.linspace(
        raw_df[column].min(),
        raw_df[column].max(),
        request.steps
    )

    df = pd.DataFrame([{
        "Type": request.input.type,
        "Air temperature [K]": request.input.air_temperature,
        "Process temperature [K]": request.input.process_temperature,
        "Rotational speed [rpm]": request.input.rotational_speed,
        "Torque [Nm]": request.input.torque,
        "Tool wear [min]": request.input.tool_wear
    }] * request.steps)
    df[column] = values

    probabilities = predict_probability(
        preprocessor.transform(df).astype(np.float32)
    ).ravel()

    return {
        "feature": request.feature,
        "points": [
            {
                "value": round(float(value), 3),
                "failure_probability": round(float(probability), 6)
            }
            for value, probability in zip(values, probabilities)
        ]
    }