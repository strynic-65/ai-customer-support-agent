"""
LLM-as-a-Judge for evaluating AI-generated Comcast support replies.
"""

from src.generation.llm_client import generate_response
from src.generation.prompts import format_judge_prompt


def judge_reply(
    customer_message,
    generated_response,
    historical_cases=None
):
    """
    Evaluate a generated support response using Gemini.

    Returns:
        Dictionary containing:
        - relevance
        - helpfulness
        - brand_style
        - grounding
        - hallucination
        - overall_score
        - explanation
    """

    if not isinstance(customer_message, str) or not customer_message.strip():
        raise ValueError(
            "Customer message must be a non-empty string."
        )

    if not isinstance(generated_response, str) or not generated_response.strip():
        raise ValueError(
            "Generated response must be a non-empty string."
        )

    if historical_cases is None:
        historical_cases = []

    # Build evaluation prompt
    prompt = format_judge_prompt(
        customer_message=customer_message,
        generated_response=generated_response,
        historical_cases=historical_cases
    )

    # Ask Gemini to evaluate the response
    result = generate_response(prompt)

    return result.strip()


if __name__ == "__main__":

    customer_message = (
        "My internet has been very slow since yesterday."
    )

    generated_response = (
        "I'm sorry you're experiencing slow internet. "
        "Please restart your modem and router and check "
        "whether the issue continues."
    )

    historical_cases = [
        {
            "document": (
                "Customer reported slow internet. "
                "Support asked them to restart their equipment."
            ),
            "score": 0.82
        }
    ]

    result = judge_reply(
        customer_message=customer_message,
        generated_response=generated_response,
        historical_cases=historical_cases
    )

    print("\n" + "=" * 60)
    print("LLM-AS-A-JUDGE RESULT")
    print("=" * 60)
    print(result)