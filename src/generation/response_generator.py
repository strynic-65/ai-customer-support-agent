from src.generation.llm_client import generate_response
from src.generation.prompts import format_reply_prompt


def generate_support_reply(customer_message, intent, retrieved_cases):
    """
    Generate a customer support reply using Gemini
    and historically similar Comcast cases.
    """

    if not isinstance(customer_message, str) or not customer_message.strip():
        raise ValueError("Customer message must be a non-empty string.")

    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("Intent must be a non-empty string.")

    if retrieved_cases is None:
        retrieved_cases = []

    prompt = format_reply_prompt(
        customer_message=customer_message,
        intent=intent,
        historical_cases=retrieved_cases
    )

    reply = generate_response(prompt)

    return reply.strip()