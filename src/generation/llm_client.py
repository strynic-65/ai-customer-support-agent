import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors


# ============================================================
# LOAD ENVIRONMENT
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
# FALLBACK RESPONSE
# ============================================================

def fallback_response(prompt):
    """
    Development fallback used when Gemini is unavailable.

    This prevents the complete support pipeline from crashing
    during development or API quota exhaustion.
    """

    return (
        "Thank you for contacting Comcast. "
        "We understand that you are experiencing an issue with your service. "
        "We are sorry for the inconvenience. "
        "Please send us a Direct Message with your account details "
        "so that our support team can investigate the issue securely."
    )


# ============================================================
# LLM RESPONSE GENERATION
# ============================================================

def generate_response(prompt):
    """
    Generate a response using Gemini.

    If Gemini is unavailable because of quota/API errors,
    return a safe fallback response instead of crashing
    the complete pipeline.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response is None or not response.text:
            print("Warning: Gemini returned an empty response.")
            return fallback_response(prompt)

        return response.text.strip()

    except errors.ClientError as error:

        # Gemini quota / rate-limit error
        if getattr(error, "code", None) == 429:

            print(
                "\nWARNING: Gemini API quota is currently exhausted."
            )

            print(
                "Using development fallback response."
            )

            return fallback_response(prompt)

        print(
            "\nWARNING: Gemini API request failed."
        )

        print(error)

        return fallback_response(prompt)

    except Exception as error:

        print(
            "\nWARNING: Unexpected LLM error."
        )

        print(error)

        return fallback_response(prompt)