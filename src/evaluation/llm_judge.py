"""
LLM-as-a-Judge for evaluating AI-generated Comcast support replies.

The judge evaluates:
- Relevance
- Helpfulness
- Accuracy
- Empathy
- Conciseness
- Safety
- Overall score

It also explicitly handles Gemini quota exhaustion.
"""

import json

from src.generation.llm_client import (
    generate_response,
    get_last_api_status,
    get_last_api_error,
)

from src.generation.prompts import format_judge_prompt


# ============================================================
# REQUIRED JUDGE FIELDS
# ============================================================

REQUIRED_SCORE_FIELDS = [
    "relevance",
    "helpfulness",
    "accuracy",
    "empathy",
    "conciseness",
    "safety",
    "overall_score",
]


# ============================================================
# QUOTA DETECTION
# ============================================================

def _is_quota_message(text):
    """
    Detect quota/rate-limit information from an error message.
    """

    if not isinstance(text, str):
        return False

    text = text.lower()

    indicators = [
        "quota",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
    ]

    return any(
        indicator in text
        for indicator in indicators
    )


# ============================================================
# JUDGE RESPONSE PARSER
# ============================================================

def _parse_judge_response(response_text):
    """
    Parse the JSON response returned by the LLM judge.

    Returns:
        dict containing the parsed evaluation
        OR an error dictionary.
    """

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if response_text is None:
        return {
            "error": "Judge returned no response.",
            "raw_response": None,
        }

    if not isinstance(response_text, str):
        return {
            "error": "Judge response is not a string.",
            "raw_response": str(response_text),
        }

    # --------------------------------------------------------
    # Remove markdown code fences if present
    # --------------------------------------------------------

    cleaned_response = response_text.strip()

    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]

    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    cleaned_response = cleaned_response.strip()

    # --------------------------------------------------------
    # Check for quota message
    # --------------------------------------------------------

    if _is_quota_message(cleaned_response):
        return {
            "error": "Gemini quota exhausted.",
            "error_type": "QUOTA_EXHAUSTED",
            "raw_response": response_text,
        }

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        result = json.loads(cleaned_response)

    except json.JSONDecodeError:
        return {
            "error": "Could not parse judge response as JSON.",
            "error_type": "INVALID_JUDGE_RESPONSE",
            "raw_response": response_text,
        }

    # --------------------------------------------------------
    # Ensure JSON object
    # --------------------------------------------------------

    if not isinstance(result, dict):
        return {
            "error": "Judge response JSON is not an object.",
            "error_type": "INVALID_JUDGE_RESPONSE",
            "raw_response": response_text,
        }

    return result


# ============================================================
# SCORE VALIDATION
# ============================================================

def _validate_scores(result):
    """
    Validate the structure and values of the judge scores.

    Every required score must be an integer/float between 1 and 5.
    """

    missing_fields = [
        field
        for field in REQUIRED_SCORE_FIELDS
        if field not in result
    ]

    if missing_fields:
        return {
            "error": (
                "Judge response is missing required fields: "
                + ", ".join(missing_fields)
            ),
            "error_type": "INVALID_JUDGE_RESPONSE",
            "raw_response": result,
        }

    # --------------------------------------------------------
    # Validate each score
    # --------------------------------------------------------

    for field in REQUIRED_SCORE_FIELDS:

        value = result[field]

        # bool is technically an int in Python, so explicitly reject it.
        if isinstance(value, bool):
            return {
                "error": (
                    f"Judge field '{field}' must be a number "
                    "between 1 and 5."
                ),
                "error_type": "INVALID_JUDGE_RESPONSE",
                "raw_response": result,
            }

        if not isinstance(value, (int, float)):
            return {
                "error": (
                    f"Judge field '{field}' must be a number "
                    "between 1 and 5."
                ),
                "error_type": "INVALID_JUDGE_RESPONSE",
                "raw_response": result,
            }

        if not 1 <= value <= 5:
            return {
                "error": (
                    f"Judge field '{field}' must be between 1 and 5."
                ),
                "error_type": "INVALID_JUDGE_RESPONSE",
                "raw_response": result,
            }

    return None


# ============================================================
# MAIN JUDGE FUNCTION
# ============================================================

def judge_reply(
    customer_message,
    generated_response,
    intent="GENERAL_INQUIRY",
    historical_cases=None,
):
    """
    Evaluate a generated Comcast support response using an LLM.

    Parameters:
        customer_message:
            Original customer message.

        generated_response:
            AI-generated support response.

        intent:
            Detected customer intent.

        historical_cases:
            Retrieved historical cases used by the support agent.

    Returns:
        Valid judge JSON dictionary, or an error dictionary.
    """

    # --------------------------------------------------------
    # Validate customer message
    # --------------------------------------------------------

    if (
        not isinstance(customer_message, str)
        or not customer_message.strip()
    ):
        return {
            "error": "Customer message must be a non-empty string.",
            "error_type": "INVALID_INPUT",
        }

    # --------------------------------------------------------
    # Validate generated response
    # --------------------------------------------------------

    if (
        not isinstance(generated_response, str)
        or not generated_response.strip()
    ):
        return {
            "error": "Generated response must be a non-empty string.",
            "error_type": "INVALID_INPUT",
        }

    # --------------------------------------------------------
    # Validate intent
    # --------------------------------------------------------

    if not isinstance(intent, str) or not intent.strip():
        intent = "GENERAL_INQUIRY"

    # --------------------------------------------------------
    # Format judge prompt
    # --------------------------------------------------------

    prompt = format_judge_prompt(
        customer_message=customer_message,
        intent=intent,
        generated_response=generated_response,
    )

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    result = generate_response(
        prompt,
        intent="GENERAL_INQUIRY",
    )

    # --------------------------------------------------------
    # Check API status FIRST
    # --------------------------------------------------------

    api_status = get_last_api_status()
    api_error = get_last_api_error()

    if api_status == "QUOTA_EXHAUSTED":

        return {
            "error": "Gemini quota exhausted.",
            "error_type": "QUOTA_EXHAUSTED",
            "api_status": api_status,
            "api_error": api_error,
            "raw_response": result,
        }

    # --------------------------------------------------------
    # Other API errors
    # --------------------------------------------------------

    if api_status == "API_ERROR":

        return {
            "error": "Gemini API request failed.",
            "error_type": "API_ERROR",
            "api_status": api_status,
            "api_error": api_error,
            "raw_response": result,
        }

    if api_status == "UNEXPECTED_ERROR":

        return {
            "error": "Unexpected Gemini error.",
            "error_type": "UNEXPECTED_ERROR",
            "api_status": api_status,
            "api_error": api_error,
            "raw_response": result,
        }

    # --------------------------------------------------------
    # Parse judge response
    # --------------------------------------------------------

    parsed_result = _parse_judge_response(result)

    # --------------------------------------------------------
    # Return parser errors
    # --------------------------------------------------------

    if "error" in parsed_result:
        return parsed_result

    # --------------------------------------------------------
    # Validate scores
    # --------------------------------------------------------

    validation_error = _validate_scores(parsed_result)

    if validation_error is not None:
        return validation_error

    # --------------------------------------------------------
    # Successful evaluation
    # --------------------------------------------------------

    parsed_result["api_status"] = api_status

    return parsed_result