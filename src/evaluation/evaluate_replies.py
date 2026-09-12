"""
Evaluate generated Comcast support replies using LLM-as-a-Judge.

The evaluator expects the same scoring schema used by JUDGE_PROMPT:

    relevance
    helpfulness
    accuracy
    empathy
    conciseness
    safety
    overall_score

If Gemini quota is unavailable, the evaluation is marked as
unavailable instead of generating misleading scores.
"""

import os
import time

import pandas as pd

from src.pipeline import run_pipeline
from src.evaluation.llm_judge import judge_reply


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/golden/golden_evaluation_set.csv"

OUTPUT_PATH = "reports/results/reply_evaluation.csv"

# Number of golden examples to evaluate.
SAMPLE_SIZE = 5

# Delay between Gemini requests.
REQUEST_DELAY = 15


# ============================================================
# EXPECTED METRICS
# ============================================================

METRICS = [
    "relevance",
    "helpfulness",
    "accuracy",
    "empathy",
    "conciseness",
    "safety",
    "overall_score",
]


# ============================================================
# VALID JUDGE RESPONSE CHECK
# ============================================================

def has_valid_judge_scores(judge_result):
    """
    Check whether the LLM judge returned all expected scores.

    Every score must be between 1 and 5.
    """

    if not isinstance(judge_result, dict):
        return False

    for metric in METRICS:

        value = judge_result.get(metric)

        if value is None:
            return False

        try:
            score = float(value)
        except (TypeError, ValueError):
            return False

        if not 1 <= score <= 5:
            return False

    return True


# ============================================================
# DETECT QUOTA FAILURE
# ============================================================

def is_quota_error(judge_result):
    """
    Detect Gemini quota/rate-limit errors.

    Supports the explicit status returned by llm_judge.py
    as well as fallback text detection.
    """

    if not isinstance(judge_result, dict):
        return False

    # --------------------------------------------------------
    # Preferred: explicit error type
    # --------------------------------------------------------

    if judge_result.get("error_type") == "QUOTA_EXHAUSTED":
        return True

    # --------------------------------------------------------
    # Preferred: explicit API status
    # --------------------------------------------------------

    if judge_result.get("api_status") == "QUOTA_EXHAUSTED":
        return True

    # --------------------------------------------------------
    # Fallback: inspect error/raw response
    # --------------------------------------------------------

    error_text = str(
        judge_result.get("error", "")
    )

    raw_response = str(
        judge_result.get("raw_response", "")
    )

    api_error = str(
        judge_result.get("api_error", "")
    )

    combined_text = (
        error_text
        + " "
        + raw_response
        + " "
        + api_error
    ).lower()

    quota_terms = [
        "quota",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
    ]

    return any(
        term in combined_text
        for term in quota_terms
    )


# ============================================================
# CREATE EMPTY RESULT
# ============================================================

def create_result_row(
    customer_message,
    detected_intent=None,
    generated_response="",
    judge_status="ERROR",
    judge_feedback="",
    judge_result=None,
):
    """
    Create a standardized evaluation result row.
    """

    if not isinstance(judge_result, dict):
        judge_result = {}

    row = {
        "customer_message": customer_message,
        "detected_intent": detected_intent,
        "generated_response": generated_response,

        "relevance": judge_result.get("relevance"),
        "helpfulness": judge_result.get("helpfulness"),
        "accuracy": judge_result.get("accuracy"),
        "empathy": judge_result.get("empathy"),
        "conciseness": judge_result.get("conciseness"),
        "safety": judge_result.get("safety"),
        "overall_score": judge_result.get("overall_score"),

        "judge_status": judge_status,
        "judge_feedback": judge_feedback,
    }

    return row


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_replies():

    # --------------------------------------------------------
    # 1. LOAD GOLDEN DATASET
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        print(
            f"ERROR: Golden evaluation file not found: "
            f"{INPUT_PATH}"
        )

        return

    df = pd.read_csv(INPUT_PATH)

    if df.empty:

        print(
            "ERROR: Golden evaluation set is empty."
        )

        return

    if "customer_message" not in df.columns:

        print(
            "ERROR: Dataset must contain "
            "'customer_message' column."
        )

        return

    sample_size = min(
        SAMPLE_SIZE,
        len(df)
    )

    sample = df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    print(
        "Golden examples selected:",
        len(sample)
    )

    results = []

    quota_exhausted = False

    # --------------------------------------------------------
    # 2. EVALUATE EACH EXAMPLE
    # --------------------------------------------------------

    for position, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        customer_message = str(
            row["customer_message"]
        )

        print("\n" + "=" * 60)

        print(
            f"Evaluating example "
            f"{position}/{sample_size}"
        )

        print(
            "Customer:",
            customer_message
        )

        try:

            # ------------------------------------------------
            # RUN SUPPORT PIPELINE
            # ------------------------------------------------

            pipeline_result = run_pipeline(
                customer_message,
                top_k=3
            )

            generated_response = (
                pipeline_result.get(
                    "reply",
                    ""
                )
            )

            retrieved_cases = (
                pipeline_result.get(
                    "retrieved_cases",
                    []
                )
            )

            detected_intent = (
                pipeline_result.get(
                    "intent",
                    "GENERAL_INQUIRY"
                )
            )

            # ------------------------------------------------
            # LLM JUDGE
            # ------------------------------------------------

            if quota_exhausted:

                judge_result = {
                    "error": (
                        "LLM judge unavailable: "
                        "Gemini quota exhausted."
                    ),
                    "error_type": "QUOTA_EXHAUSTED",
                    "api_status": "QUOTA_EXHAUSTED",
                }

            else:

                judge_result = judge_reply(
                    customer_message=customer_message,
                    generated_response=generated_response,
                    intent=detected_intent,
                    historical_cases=retrieved_cases,
                )

                # --------------------------------------------
                # CHECK QUOTA
                # --------------------------------------------

                if is_quota_error(judge_result):

                    quota_exhausted = True

            # ------------------------------------------------
            # DETERMINE STATUS
            # ------------------------------------------------

            if has_valid_judge_scores(
                judge_result
            ):

                judge_status = "SUCCESS"

            elif is_quota_error(
                judge_result
            ):

                judge_status = (
                    "QUOTA_EXHAUSTED"
                )

            elif isinstance(judge_result, dict):

                judge_status = (
                    judge_result.get(
                        "error_type",
                        "INVALID_JUDGE_RESPONSE"
                    )
                )

                if judge_status not in {
                    "INVALID_JUDGE_RESPONSE",
                    "API_ERROR",
                    "UNEXPECTED_ERROR",
                    "INVALID_INPUT",
                }:

                    judge_status = (
                        "INVALID_JUDGE_RESPONSE"
                    )

            else:

                judge_status = (
                    "INVALID_JUDGE_RESPONSE"
                )

            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            if isinstance(judge_result, dict):

                explanation = (
                    judge_result.get(
                        "explanation",
                        judge_result.get(
                            "error",
                            ""
                        )
                    )
                )

            else:

                explanation = str(
                    judge_result
                )

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            results.append(
                create_result_row(
                    customer_message=customer_message,
                    detected_intent=detected_intent,
                    generated_response=generated_response,
                    judge_status=judge_status,
                    judge_feedback=explanation,
                    judge_result=judge_result,
                )
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            print(
                "\nGenerated Response:"
            )

            print(
                generated_response
            )

            print(
                "\nDetected Intent:"
            )

            print(
                detected_intent
            )

            print(
                "\nJudge Status:"
            )

            print(
                judge_status
            )

            if judge_status == "SUCCESS":

                print(
                    "\nJudge Scores:"
                )

                for metric in METRICS:

                    print(
                        f"{metric}: "
                        f"{judge_result.get(metric)}"
                    )

            else:

                print(
                    "\nJudge:"
                )

                print(
                    judge_result
                )

        except Exception as error:

            print(
                "\nERROR:",
                error
            )

            results.append(
                create_result_row(
                    customer_message=customer_message,
                    detected_intent=None,
                    generated_response="",
                    judge_status="ERROR",
                    judge_feedback=str(error),
                )
            )

        # ----------------------------------------------------
        # WAIT BETWEEN GEMINI REQUESTS
        # ----------------------------------------------------

        if (
            position < sample_size
            and not quota_exhausted
        ):

            print(
                f"\nWaiting {REQUEST_DELAY} seconds "
                "before the next Gemini request..."
            )

            time.sleep(
                REQUEST_DELAY
            )

    # ========================================================
    # 3. RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 4. PRINT METRICS
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "REPLY QUALITY RESULTS"
    )

    print("=" * 60)

    print(
        "Evaluated examples:",
        len(results_df)
    )

    if not results_df.empty:

        successful_count = (
            results_df[
                "judge_status"
            ] == "SUCCESS"
        ).sum()

        print(
            "Successful LLM evaluations:",
            successful_count
        )

        # ----------------------------------------------------
        # Status counts
        # ----------------------------------------------------

        print(
            "\nJudge Status Counts:"
        )

        print(
            results_df[
                "judge_status"
            ].value_counts()
        )

        # ----------------------------------------------------
        # Numeric metrics
        # ----------------------------------------------------

        for column in METRICS:

            average = results_df[
                column
            ].mean()

            if pd.isna(average):

                print(
                    f"{column.upper():15}: N/A"
                )

            else:

                print(
                    f"{column.upper():15}: "
                    f"{average:.2f}/5"
                )

    else:

        print(
            "No evaluation results were generated."
        )

    # ========================================================
    # 5. SAVE RESULTS
    # ========================================================

    os.makedirs(
        "reports/results",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    evaluate_replies()