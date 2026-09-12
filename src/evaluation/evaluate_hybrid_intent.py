# src/evaluation/evaluate_hybrid_intent.py

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.intent.ml_classifier import (
    create_pipeline,
    normalize_text,
    apply_domain_rules,
)


DATA_PATH = "data/golden/intent_review_grouped.csv"

INTENTS = [
    "BILLING_ISSUE",
    "INTERNET_OUTAGE",
    "TECHNICAL_ISSUE",
    "SLOW_INTERNET",
    "ACCOUNT_ISSUE",
    "PLAN_CHANGE",
    "CANCELLATION",
    "PAYMENT_ISSUE",
    "SERVICE_AVAILABILITY",
    "GENERAL_INQUIRY",
]


def hybrid_predict(text, model):
    """
    Predict using ML + domain rules.
    """

    text = normalize_text(text)

    probabilities = model.predict_proba([text])[0]

    best_index = int(np.argmax(probabilities))

    ml_intent = model.classes_[best_index]

    ml_confidence = float(
        probabilities[best_index]
    )

    result = apply_domain_rules(
        text=text,
        ml_intent=ml_intent,
        ml_confidence=ml_confidence,
    )

    return result["intent"]


def main():

    print("=" * 70)
    print("PROPER HYBRID INTENT CROSS-VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=[
            "customer_message",
            "reviewed_intent",
        ]
    ).copy()

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["reviewed_intent"] = (
        df["reviewed_intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_message"] != ""
    ]

    texts = df["customer_message"].tolist()

    labels = df["reviewed_intent"].tolist()

    print()
    print(f"Total examples: {len(df)}")

    # --------------------------------------------------------
    # 5-fold CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    predictions = []

    actual_labels = []

    fold_number = 1

    for train_index, test_index in cv.split(
        texts,
        labels
    ):

        print()
        print(
            f"Running fold {fold_number}/5..."
        )

        X_train = [
            texts[i]
            for i in train_index
        ]

        y_train = [
            labels[i]
            for i in train_index
        ]

        X_test = [
            texts[i]
            for i in test_index
        ]

        y_test = [
            labels[i]
            for i in test_index
        ]

        # Train ONLY on training fold.
        model = create_pipeline()

        normalized_train = [
            normalize_text(text)
            for text in X_train
        ]

        model.fit(
            normalized_train,
            y_train,
        )

        # Predict validation fold.
        for text, actual in zip(
            X_test,
            y_test
        ):

            prediction = hybrid_predict(
                text,
                model
            )

            predictions.append(
                prediction
            )

            actual_labels.append(
                actual
            )

        fold_number += 1

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        actual_labels,
        predictions,
    )

    precision = precision_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL HYBRID CROSS-VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HYBRID CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            actual_labels,
            predictions,
            labels=INTENTS,
            target_names=INTENTS,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HYBRID CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        actual_labels,
        predictions,
        labels=INTENTS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=INTENTS,
        columns=INTENTS,
    )

    print(cm_df)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        {
            "customer_message": texts,
            "actual_intent": actual_labels,
            "predicted_intent": predictions,
        }
    )

    results_df["correct"] = (
        results_df["actual_intent"]
        ==
        results_df["predicted_intent"]
    )

    output_path = (
        "data/golden/"
        "hybrid_intent_predictions.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Error analysis
    # --------------------------------------------------------

    errors = results_df[
        results_df["correct"] == False
    ].copy()

    error_path = (
        "data/golden/"
        "hybrid_intent_errors.csv"
    )

    errors.to_csv(
        error_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Correct   : "
        f"{results_df['correct'].sum()}"
    )

    print(
        f"Incorrect : "
        f"{len(errors)}"
    )

    print(
        f"Total     : "
        f"{len(results_df)}"
    )

    print()
    print(
        f"Predictions saved to: "
        f"{output_path}"
    )

    print(
        f"Errors saved to: "
        f"{error_path}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()