import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

target = "Machine failure"

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features].copy()
y = df[target].copy()

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

joblib.dump(
    preprocessor,
    os.path.join(MODEL_DIR, "preprocessor.joblib")
)

joblib.dump(
    features,
    os.path.join(MODEL_DIR, "feature_names.joblib")
)

joblib.dump(
    {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    },
    os.path.join(MODEL_DIR, "processed_data.joblib")
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"\nOriginal features: {X.shape[1]}")
print(f"Processed features: {X_train_processed.shape[1]}")

print("\nTrain set:")
print(f"X: {X_train_processed.shape}")
print(f"y: {y_train.shape}")
print(f"Failures: {y_train.sum()}")

print("\nValidation set:")
print(f"X: {X_val_processed.shape}")
print(f"y: {y_val.shape}")
print(f"Failures: {y_val.sum()}")

print("\nTest set:")
print(f"X: {X_test_processed.shape}")
print(f"y: {y_test.shape}")
print(f"Failures: {y_test.sum()}")

print("\nClass distribution:")

for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    failures = int(labels.sum())
    total = len(labels)
    percentage = failures / total * 100
    print(
        f"{name}: {failures}/{total} failures "
        f"({percentage:.2f}%)"
    )

print("\nSaved:")
print("- models/preprocessor.joblib")
print("- models/feature_names.joblib")
print("- models/processed_data.joblib")