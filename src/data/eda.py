import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)


# LOAD DATASET

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# COLUMN INFORMATION

print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

print(df.dtypes)


# MISSING VALUES

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print(f"\nTotal missing values: {missing_values.sum()}")


# DUPLICATES

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# TARGET DISTRIBUTION


TARGET = "Machine failure"

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df[TARGET].value_counts().sort_index()

print(target_counts)

failure_percentage = (
    df[TARGET].mean() * 100
)

print(f"\nFailure percentage: {failure_percentage:.2f}%")
print(f"No-failure percentage: {100 - failure_percentage:.2f}%")


# NUMERICAL SUMMARY


print("\n" + "=" * 60)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 60)

print(df.describe())


# SENSOR FEATURES

sensor_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("\n" + "=" * 60)
print("SENSOR FEATURES")
print("=" * 60)

for feature in sensor_features:
    print(f"\n{feature}")
    print(f"  Min:    {df[feature].min():.2f}")
    print(f"  Mean:   {df[feature].mean():.2f}")
    print(f"  Median: {df[feature].median():.2f}")
    print(f"  Max:    {df[feature].max():.2f}")

# PLOT 1 — TARGET DISTRIBUTION

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=TARGET
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Samples")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORTS_DIR, "target_distribution.png"),
    dpi=300
)

plt.close()


# PLOT 2 — SENSOR DISTRIBUTIONS


for feature in sensor_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        bins=30
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()

    filename = (
        feature.lower()
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
    )

    plt.savefig(
        os.path.join(
            REPORTS_DIR,
            f"{filename}_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# PLOT 3 — CORRELATION MATRIX

numeric_columns = sensor_features + [TARGET]

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Sensor Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORTS_DIR, "correlation_matrix.png"),
    dpi=300
)

plt.close()

# PLOT 4 — SENSOR FEATURES VS FAILURE
for feature in sensor_features:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=TARGET,
        y=feature
    )

    plt.title(f"{feature} vs Machine Failure")
    plt.xlabel("Machine Failure")
    plt.ylabel(feature)
    plt.tight_layout()

    filename = (
        feature.lower()
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
    )

    plt.savefig(
        os.path.join(
            REPORTS_DIR,
            f"{filename}_vs_failure.png"
        ),
        dpi=300
    )

    plt.close()

# SAVE CLEAN DATA SUMMARY
summary = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "failure_samples": int(df[TARGET].sum()),
    "no_failure_samples": int((df[TARGET] == 0).sum()),
    "failure_percentage": round(failure_percentage, 2)
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    os.path.join(REPORTS_DIR, "eda_summary.csv"),
    index=False
)


# COMPLETION MESSAGE

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nReports saved to:")
print(REPORTS_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(REPORTS_DIR)):
    print(f"  - {file}")