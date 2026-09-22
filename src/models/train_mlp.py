import os
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "models", "processed_data.joblib")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = joblib.load(DATA_PATH)

X_train = data["X_train"].astype(np.float32)
X_val = data["X_val"].astype(np.float32)
X_test = data["X_test"].astype(np.float32)

y_train = data["y_train"].astype(np.float32)
y_val = data["y_val"].astype(np.float32)
y_test = data["y_test"].astype(np.float32)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)

X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val).unsqueeze(1)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

input_size = X_train.shape[1]

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

model = MLP(input_size).to(device)

positive_count = y_train.sum()
negative_count = len(y_train) - positive_count

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)

epochs = 100
patience = 10

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_state = None
patience_counter = 0

for epoch in range(epochs):

    model.train()

    running_train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item() * X_batch.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            running_val_loss += loss.item() * X_batch.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

    else:

        patience_counter += 1

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():

    logits = model(X_test_tensor.to(device))

    probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)

cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 60)
print("MLP TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Failure", "Failure"],
        zero_division=0
    )
)

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, len(val_losses) + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(REPORTS_DIR, "mlp_loss_curve.png"),
    dpi=300
)

plt.close()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"]
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title("MLP Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORTS_DIR, "mlp_confusion_matrix.png"),
    dpi=300
)

plt.close()

fpr, tpr, _ = roc_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"MLP ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("MLP ROC Curve")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(REPORTS_DIR, "mlp_roc_curve.png"),
    dpi=300
)

plt.close()

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    os.path.join(REPORTS_DIR, "mlp_metrics.csv"),
    index=False
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "architecture": "MLP",
        "threshold": 0.5
    },
    os.path.join(MODEL_DIR, "mlp_model.pth")
)

joblib.dump(
    probabilities,
    os.path.join(MODEL_DIR, "mlp_test_probabilities.joblib")
)

print("\nSaved:")
print("- models/mlp_model.pth")
print("- models/mlp_test_probabilities.joblib")
print("- reports/mlp_metrics.csv")
print("- reports/mlp_loss_curve.png")
print("- reports/mlp_confusion_matrix.png")
print("- reports/mlp_roc_curve.png")

print(f"\nDevice: {device}")