"""
Evaluate generated Comcast support replies using LLM-as-a-Judge.

If Gemini quota is unavailable, the script records the evaluation
as unavailable instead of producing misleading scores.
"""

import os
import re
import time

import pandas as pd

from src.pipeline import run_pipeline
from src.evaluation.llm_judge import judge_reply


INPUT_PATH = "data/golden/golden_evaluation_set.csv"
OUTPUT_PATH = "reports/results/reply_evaluation.csv"

# Number of golden examples to evaluate.
SAMPLE_SIZE = 5

# Delay between successful Gemini requests.
REQUEST_DELAY = 15


# ============================================================
# SCORE EXTRACTION
# ============================================================

def extract_score(text, metric):
    """
    Extract a score such as:

        RELEVANCE: 5

    Returns None if the score cannot be found.
    """

    if not isinstance(text, str):
        return None

    pattern = rf"{metric}\s*:\s*(\d+(?:\.\d+)?)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None


# ============================================================
# CHECK WHETHER JUDGE RETURNED REAL SCORES
# ============================================================

def has_valid_judge_scores(judge_result):
    """
    Check whether the LLM judge returned the expected
    six evaluation metrics.
    """

    metrics = [
        "RELEVANCE",
        "HELPFULNESS",
        "TONE",
        "ACCURACY",
        "BRAND_STYLE",
        "OVERALL",
    ]

    if not isinstance(judge_result, str):
        return False

    found_scores = 0

    for metric in metrics:
        if extract_score(
            judge_result,
            metric
        ) is not None:
            found_scores += 1

    return found_scores == len(metrics)


# ============================================================
# DETECT GEMINI QUOTA FAILURE
# ============================================================

def is_quota_error(text):
    """
    Detect Gemini quota/rate-limit messages.
    """

    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    quota_terms = [
        "quota",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "429",
    ]

    return any(
        term in text_lower
        for term in quota_terms
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_replies():

    # ---------------------------------------------------------
    # 1. Load Golden Evaluation Set
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_PATH
    )

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

    # ---------------------------------------------------------
    # 2. Evaluate each example
    # ---------------------------------------------------------

    for position, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        customer_message = row[
            "customer_message"
        ]

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

            # -------------------------------------------------
            # Run complete support pipeline
            # -------------------------------------------------

            pipeline_result = run_pipeline(
                customer_message,
                top_k=3
            )

            generated_response = (
                pipeline_result["reply"]
            )

            retrieved_cases = (
                pipeline_result["retrieved_cases"]
            )

            # -------------------------------------------------
            # If quota was already detected, do not make
            # additional judge requests.
            # -------------------------------------------------

            if quota_exhausted:

                judge_result = (
                    "LLM_JUDGE_UNAVAILABLE: "
                    "Gemini quota exhausted."
                )

            else:

                # ---------------------------------------------
                # Ask LLM judge
                # ---------------------------------------------

                judge_result = judge_reply(
                    customer_message=customer_message,
                    generated_response=generated_response,
                    historical_cases=retrieved_cases
                )

                # ---------------------------------------------
                # Check whether the judge returned valid scores
                # ---------------------------------------------

                if not has_valid_judge_scores(
                    judge_result
                ):

                    if is_quota_error(
                        judge_result
                    ):

                        quota_exhausted = True

                        judge_result = (
                            "LLM_JUDGE_UNAVAILABLE: "
                            "Gemini quota exhausted."
                        )

                    else:

                        judge_result = (
                            "LLM_JUDGE_INVALID_RESPONSE: "
                            + str(judge_result)
                        )

            # -------------------------------------------------
            # Extract scores
            # -------------------------------------------------

            relevance = extract_score(
                judge_result,
                "RELEVANCE"
            )

            helpfulness = extract_score(
                judge_result,
                "HELPFULNESS"
            )

            tone = extract_score(
                judge_result,
                "TONE"
            )

            accuracy = extract_score(
                judge_result,
                "ACCURACY"
            )

            brand_style = extract_score(
                judge_result,
                "BRAND_STYLE"
            )

            overall = extract_score(
                judge_result,
                "OVERALL"
            )

            # -------------------------------------------------
            # Determine evaluation status
            # -------------------------------------------------

            if has_valid_judge_scores(
                judge_result
            ):

                judge_status = "SUCCESS"

            elif "QUOTA" in str(
                judge_result
            ).upper():

                judge_status = "QUOTA_EXHAUSTED"

            else:

                judge_status = "INVALID_JUDGE_RESPONSE"

            # -------------------------------------------------
            # Store result
            # -------------------------------------------------

            results.append(
                {
                    "customer_message": customer_message,
                    "generated_response": generated_response,
                    "relevance": relevance,
                    "helpfulness": helpfulness,
                    "tone": tone,
                    "accuracy": accuracy,
                    "brand_style": brand_style,
                    "overall": overall,
                    "judge_status": judge_status,
                    "judge_feedback": judge_result,
                }
            )

            print("\nGenerated Response:")
            print(
                generated_response
            )

            print("\nJudge Status:")
            print(
                judge_status
            )

            print("\nJudge:")
            print(
                judge_result
            )

        except Exception as error:

            print(
                "\nERROR:",
                error
            )

            results.append(
                {
                    "customer_message": customer_message,
                    "generated_response": "",
                    "relevance": None,
                    "helpfulness": None,
                    "tone": None,
                    "accuracy": None,
                    "brand_style": None,
                    "overall": None,
                    "judge_status": "ERROR",
                    "judge_feedback": str(error),
                }
            )

        # -----------------------------------------------------
        # Wait only if Gemini is still available.
        # -----------------------------------------------------

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

    # ---------------------------------------------------------
    # 3. Create results DataFrame
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------
    # 4. Calculate averages
    # ---------------------------------------------------------

    numeric_columns = [
        "relevance",
        "helpfulness",
        "tone",
        "accuracy",
        "brand_style",
        "overall",
    ]

    print("\n" + "=" * 60)
    print("REPLY QUALITY RESULTS")
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

        for column in numeric_columns:

            average = results_df[
                column
            ].mean()

            if pd.isna(average):

                print(
                    f"{column.upper():15}: "
                    "N/A"
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

    # ---------------------------------------------------------
    # 5. Save results
    # ---------------------------------------------------------

    os.makedirs(
        "reports/results",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nResults saved to:")
    print(
        OUTPUT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate_replies()