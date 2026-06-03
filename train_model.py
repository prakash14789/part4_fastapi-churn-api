import pandas as pd
import numpy as np
import joblib
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# LOAD DATA


df = pd.read_csv("data/rfm_modeling_snapshot.csv")


# TRAIN / VAL / TEST


train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

target = "churn_next_60d"

drop_cols = [
    "customer_id",
    "snapshot_date",
    "split",
    target
]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]

X_val = val_df.drop(columns=drop_cols)
y_val = val_df[target]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]


# FEATURE TYPES


categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

# PREPROCESSING


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# MODEL


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# TRAIN


pipeline.fit(X_train, y_train)

# EVALUATE


preds = pipeline.predict(X_test)

probs = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)

precision = precision_score(y_test, preds)

recall = recall_score(y_test, preds)

f1 = f1_score(y_test, preds)

roc_auc = roc_auc_score(y_test, probs)

cm = confusion_matrix(y_test, preds)

print("\n===== TEST METRICS =====\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)


# SAVE MODEL


joblib.dump(pipeline, "model.pkl")

# SAVE METRICS


metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nmodel.pkl saved")
print("metrics.json saved")

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

probs = pipeline.predict_proba(X_test)[:,1]

for threshold in np.arange(0.30, 0.65, 0.05):
    preds = (probs >= threshold).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )