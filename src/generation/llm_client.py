"""
Gemini LLM client for the AI Customer Support Agent.

Handles:
- Gemini API configuration
- Response generation
- API quota/rate-limit detection
- Safe fallback responses
- Explicit status tracking for evaluation modules
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# LAST API STATUS
# ============================================================
#
# Other modules such as the LLM-as-a-Judge can inspect this
# value to distinguish:
#
# - successful Gemini response
# - quota exhaustion
# - API error
# - unexpected error
#
# ============================================================

LAST_API_STATUS = "NOT_CALLED"
LAST_API_ERROR = None


# ============================================================
# FALLBACK RESPONSES
# ============================================================

FALLBACK_RESPONSES = {
    "INTERNET_OUTAGE": (
        "We're sorry you're experiencing an internet outage. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help troubleshoot the service issue."
    ),

    "SLOW_INTERNET": (
        "We're sorry your internet connection is running slowly. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help troubleshoot the connection."
    ),

    "TECHNICAL_ISSUE": (
        "We're sorry you're experiencing a technical issue. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help troubleshoot the problem."
    ),

    "BILLING_ISSUE": (
        "We're sorry you're having trouble with your bill. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help review the billing concern."
    ),

    "PAYMENT_ISSUE": (
        "We're sorry you're having trouble with your payment. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help with the payment issue."
    ),

    "ACCOUNT_ISSUE": (
        "We're sorry you're having trouble with your account. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help troubleshoot the account issue."
    ),

    "PLAN_CHANGE": (
        "We'd be happy to help with your service plan. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help with your plan request."
    ),

    "CANCELLATION": (
        "We're sorry to hear you'd like to cancel your service. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help with your request."
    ),

    "SERVICE_AVAILABILITY": (
        "We'd be happy to help you check service availability. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel so we can help with your request."
    ),

    "GENERAL_INQUIRY": (
        "Thank you for contacting Comcast. "
        "Please send us a Direct Message through a secure Comcast "
        "support channel and we'll be happy to help."
    ),
}


# ============================================================
# FALLBACK RESPONSE
# ============================================================

def fallback_response(prompt, intent="GENERAL_INQUIRY"):
    """
    Return a safe development fallback response.

    The prompt parameter is kept for compatibility with the
    existing project interface.
    """

    if not isinstance(intent, str):
        intent = "GENERAL_INQUIRY"

    intent = intent.strip().upper()

    return FALLBACK_RESPONSES.get(
        intent,
        FALLBACK_RESPONSES["GENERAL_INQUIRY"]
    )


# ============================================================
# ERROR DETECTION
# ============================================================

def _is_quota_error(error):
    """
    Detect whether a Gemini API error indicates quota,
    rate-limit, or resource exhaustion.
    """

    error_text = str(error).lower()

    quota_indicators = [
        "quota",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
    ]

    return any(
        indicator in error_text
        for indicator in quota_indicators
    )


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(prompt, intent="GENERAL_INQUIRY"):
    """
    Generate a response using Gemini.

    If Gemini is unavailable, return a safe fallback response
    instead of crashing the application.

    LAST_API_STATUS will contain one of:

        SUCCESS
        QUOTA_EXHAUSTED
        API_ERROR
        UNEXPECTED_ERROR
    """

    global LAST_API_STATUS
    global LAST_API_ERROR

    # Reset status for this request.
    LAST_API_STATUS = "NOT_CALLED"
    LAST_API_ERROR = None

    # --------------------------------------------------------
    # Validate prompt
    # --------------------------------------------------------

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        # ----------------------------------------------------
        # Empty response
        # ----------------------------------------------------

        if (
            response is None
            or not getattr(response, "text", None)
            or not response.text.strip()
        ):
            LAST_API_STATUS = "API_ERROR"
            LAST_API_ERROR = "Gemini returned an empty response."

            print(
                "WARNING: Gemini returned an empty response."
            )

            return fallback_response(
                prompt,
                intent
            )

        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        LAST_API_STATUS = "SUCCESS"
        LAST_API_ERROR = None

        return response.text.strip()

    # --------------------------------------------------------
    # Gemini Client Error
    # --------------------------------------------------------

    except errors.ClientError as error:

        LAST_API_ERROR = str(error)

        if _is_quota_error(error):

            LAST_API_STATUS = "QUOTA_EXHAUSTED"

            print(
                "\nWARNING: Gemini API quota is currently exhausted."
            )

            print(
                "Using development fallback response."
            )

            return fallback_response(
                prompt,
                intent
            )

        # ----------------------------------------------------
        # Other Gemini API error
        # ----------------------------------------------------

        LAST_API_STATUS = "API_ERROR"

        print(
            "\nWARNING: Gemini API request failed."
        )

        print(error)

        return fallback_response(
            prompt,
            intent
        )

    # --------------------------------------------------------
    # Unexpected Error
    # --------------------------------------------------------

    except Exception as error:

        LAST_API_STATUS = "UNEXPECTED_ERROR"
        LAST_API_ERROR = str(error)

        print(
            "\nWARNING: Unexpected LLM error."
        )

        print(error)

        return fallback_response(
            prompt,
            intent
        )


# ============================================================
# STATUS HELPERS
# ============================================================

def get_last_api_status():
    """
    Return the status of the most recent Gemini request.
    """

    return LAST_API_STATUS


def get_last_api_error():
    """
    Return the error message from the most recent failed
    Gemini request, if available.
    """

    return LAST_API_ERROR