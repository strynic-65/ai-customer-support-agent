import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.escalation.escalation_detector import detect_escalation


DATA_PATH = "data/golden/escalation_test_cases.csv"


def evaluate_escalation():
    print("=" * 70)
    print("ESCALATION SYSTEM EVALUATION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    predictions = []
    actuals = []
    details = []

    for _, row in df.iterrows():

        customer_message = str(row["customer_message"])
        expected_decision = str(row["expected_escalation"]).strip().upper()

        # The escalation detector only needs the message.
        # We use a routine intent and high confidence because
        # this benchmark is testing escalation-risk rules themselves.
        result = detect_escalation(
            customer_message=customer_message,
            intent="GENERAL_INQUIRY",
            confidence=0.90,
        )

        predicted_decision = result["decision"]

        predictions.append(predicted_decision)
        actuals.append(expected_decision)

        details.append(
            {
                "customer_message": customer_message,
                "expected_escalation": expected_decision,
                "predicted_escalation": predicted_decision,
                "correct": expected_decision == predicted_decision,
                "reasons": " | ".join(result["reasons"]),
            }
        )

    accuracy = accuracy_score(actuals, predictions)

    precision = precision_score(
        actuals,
        predictions,
        pos_label="ESCALATE",
        zero_division=0,
    )

    recall = recall_score(
        actuals,
        predictions,
        pos_label="ESCALATE",
        zero_division=0,
    )

    f1 = f1_score(
        actuals,
        predictions,
        pos_label="ESCALATE",
        zero_division=0,
    )

    print()
    print(f"Total Cases : {len(df)}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            actuals,
            predictions,
            labels=["AUTO_HANDLE", "ESCALATE"],
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print("-" * 70)

    cm = confusion_matrix(
        actuals,
        predictions,
        labels=["AUTO_HANDLE", "ESCALATE"],
    )

    print("                 Predicted")
    print("               AUTO   ESCALATE")
    print(f"Actual AUTO    {cm[0][0]:4d}   {cm[0][1]:8d}")
    print(f"Actual ESC     {cm[1][0]:4d}   {cm[1][1]:8d}")

    results_df = pd.DataFrame(details)

    output_path = "data/golden/escalation_predictions.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print(f"Detailed results saved to: {output_path}")

    incorrect = results_df[results_df["correct"] == False]

    print()
    print(f"Incorrect Cases: {len(incorrect)}")

    if len(incorrect) > 0:
        print()
        print("Error Analysis")
        print("-" * 70)

        for _, row in incorrect.iterrows():
            print(f"Message    : {row['customer_message']}")
            print(f"Expected   : {row['expected_escalation']}")
            print(f"Predicted  : {row['predicted_escalation']}")
            print(f"Reason     : {row['reasons']}")
            print("-" * 70)
    else:
        print("All escalation test cases passed!")

    print()
    print("=" * 70)
    print("ESCALATION EVALUATION COMPLETED")
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    evaluate_escalation()