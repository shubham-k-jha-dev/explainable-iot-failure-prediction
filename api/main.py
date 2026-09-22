import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from fastapi import FastAPI
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    ExplanationItem,
    HealthResponse
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_model.pth")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
DATA_PATH = os.path.join(BASE_DIR, "models", "processed_data.joblib")

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