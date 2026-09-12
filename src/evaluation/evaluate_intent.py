from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.intent.ml_classifier import build_vectorizer, build_classifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "intent_review_grouped.csv"
)

ERROR_PATH = (
    BASE_DIR
    / "data"
    / "golden"
    / "ml_intent_errors.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INTENT CLASSIFICATION EVALUATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

TEXT_COLUMN = "customer_message"
LABEL_COLUMN = "reviewed_intent"

df = df.dropna(
    subset=[
        TEXT_COLUMN,
        LABEL_COLUMN
    ]
).copy()

print(f"\nEvaluation examples: {len(df)}")
print(f"Number of intents: {df[LABEL_COLUMN].nunique()}")

print("\nIntent distribution:")
print(df[LABEL_COLUMN].value_counts().sort_index())


# ============================================================
# PREPARE DATA
# ============================================================

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN].astype(str)


# ============================================================
# CROSS-VALIDATION
# ============================================================

print("\nRunning 5-fold stratified cross-validation...")

vectorizer = build_vectorizer()

X_tfidf = vectorizer.fit_transform(X)

classifier = build_classifier()

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    classifier,
    X_tfidf,
    y,
    cv=cv
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("PER-INTENT RESULTS")
print("=" * 70)

report = classification_report(
    y,
    predictions,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm_df)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = df[
    y != predictions
].copy()

errors["predicted_intent"] = predictions[
    y != predictions
]

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(f"\nIncorrect predictions: {len(errors)}")
print(f"Correct predictions  : {len(df) - len(errors)}")

if len(errors) > 0:

    print("\nTop confusion pairs:")

    confusion_pairs = (
        errors
        .groupby(
            [
                LABEL_COLUMN,
                "predicted_intent"
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        confusion_pairs.head(15).to_string(
            index=False
        )
    )


# ============================================================
# SAVE ERRORS
# ============================================================

errors[
    [
        "review_id",
        TEXT_COLUMN,
        LABEL_COLUMN,
        "predicted_intent"
    ]
].to_csv(
    ERROR_PATH,
    index=False
)

print(
    f"\nSaved error analysis to:"
    f"\n{ERROR_PATH}"
)


# ============================================================
# SAMPLE INCORRECT PREDICTIONS
# ============================================================

if len(errors) > 0:

    print("\n" + "=" * 70)
    print("SAMPLE INCORRECT PREDICTIONS")
    print("=" * 70)

    for _, row in errors.head(15).iterrows():

        print("\nCustomer:")
        print(row[TEXT_COLUMN])

        print(
            "Actual    :",
            row[LABEL_COLUMN]
        )

        print(
            "Predicted :",
            row["predicted_intent"]
        )

        print("-" * 70)


print("\n" + "=" * 70)
print("INTENT EVALUATION COMPLETE")
print("=" * 70)