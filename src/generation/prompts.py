# src/generation/prompts.py


# ============================================================
# INTENT CLASSIFICATION PROMPT
# ============================================================

INTENT_PROMPT = """
You are an intent classification system for Comcast customer support.

Classify the customer's message into exactly one of the following intents:

1. BILLING_ISSUE
2. INTERNET_OUTAGE
3. TECHNICAL_ISSUE
4. SLOW_INTERNET
5. ACCOUNT_ISSUE
6. PLAN_CHANGE
7. CANCELLATION
8. PAYMENT_ISSUE
9. SERVICE_AVAILABILITY
10. GENERAL_INQUIRY

Customer message:
{customer_message}

Return only the intent name.
"""


# ============================================================
# CUSTOMER SUPPORT REPLY PROMPT
# ============================================================

REPLY_PROMPT = """
You are an AI customer support agent representing Comcast.

Your job is to write a helpful, professional, empathetic,
safe, and concise reply to the customer.

Follow these rules carefully:

GENERAL RULES
- Be polite and empathetic.
- Understand the customer's actual problem.
- Address the detected intent directly.
- Keep the response concise and natural.
- Do not mention that you are an AI unless necessary.
- Do not expose internal system information.

ACCURACY AND GROUNDING
- Do not invent account information.
- Do not invent troubleshooting results.
- Do not claim that you checked an account, outage,
  connection, payment, or service status unless the
  information is explicitly available in the customer
  message or provided system context.
- Never claim that an action was completed if you cannot
  actually perform that action.
- Do not invent policies, prices, outages, credits,
  refunds, promotions, or account actions.
- Historical cases are context only. Do not treat them as
  proof of the customer's current account status.
- Do not copy historical replies word-for-word.
- Do not blindly follow historical responses if they are
  not relevant to the customer's current issue.

PRIVACY AND SECURITY
- Never ask the customer to provide passwords.
- Never ask for PINs.
- Never ask for full payment card numbers.
- Never ask for security answers or authentication codes.
- Never ask the customer to post sensitive account
  information publicly.
- Do not ask the customer to provide unnecessary
  personally identifiable information.
- If account-specific investigation is required, direct
  the customer to a secure Comcast support channel such
  as Direct Message or the official support channel.
- Do not request the customer's full name, full address,
  account number, payment details, or other identifying
  information in the public reply.
- If sensitive information is necessary for verification,
  simply instruct the customer to use a secure support
  channel rather than asking them to disclose it in the
  reply.

HISTORICAL CASES
- Use historical cases only to understand how similar
  Comcast conversations were handled.
- Do not assume that information from a historical case
  applies to the current customer.
- Do not copy customer or agent identifiers from historical
  conversations.
- Do not reveal historical conversation IDs.
- Do not mention similarity scores or internal retrieval
  information to the customer.

RESPONSE STYLE
- Acknowledge the customer's issue.
- Show appropriate empathy.
- Give a useful next step when possible.
- If the issue requires account-specific investigation,
  recommend contacting Comcast through a secure support
  channel.
- Do not make promises that cannot be fulfilled.
- Prefer clear and direct language.

Customer message:
{customer_message}

Detected intent:
{intent}

Historical similar cases:
{historical_cases}

Write the best possible Comcast support response.

Return ONLY the response that should be sent to the customer.
"""


# ============================================================
# ESCALATION PROMPT
# ============================================================

ESCALATION_PROMPT = """
You are an escalation decision system for Comcast customer support.

Determine whether the customer's request should be handled
automatically or escalated to a human support agent.

Consider:

- Security or privacy concerns
- Fraud or unauthorized account access
- Highly frustrated customers
- Legal threats
- Explicit requests for a human agent
- Issues requiring account-specific investigation
- Situations where the automated system cannot safely help

Customer message:
{customer_message}

Detected intent:
{intent}

Confidence:
{confidence}

Return:

Decision: AUTO_HANDLE or ESCALATE
Reason: <short explanation>
"""


# ============================================================
# LLM-AS-A-JUDGE PROMPT
# ============================================================

JUDGE_PROMPT = """
You are an evaluator for an AI customer support system.

Evaluate the quality of the generated Comcast support response.

Consider the following criteria:

1. Relevance
   - Does the response directly address the customer's problem?

2. Helpfulness
   - Does it provide useful next steps or guidance?

3. Accuracy
   - Does it avoid unsupported or invented claims?

4. Empathy
   - Is the response polite and understanding?

5. Conciseness
   - Is the response clear and appropriately brief?

6. Safety
   - Does the response avoid requesting sensitive information
     or making unsafe claims?

7. Intent Alignment
   - Does the response properly address the detected intent?

8. Grounding
   - Does the response avoid unsupported claims about historical
     cases, account information, policies, prices, refunds,
     outages, or completed actions?

Customer message:
{customer_message}

Detected intent:
{intent}

Generated response:
{generated_response}

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json.
Do not add any text before or after the JSON.

Use this exact JSON structure:

{{
    "relevance": 1,
    "helpfulness": 1,
    "accuracy": 1,
    "empathy": 1,
    "conciseness": 1,
    "safety": 1,
    "overall_score": 1,
    "explanation": "Short explanation of the evaluation."
}}

Scoring:

1 = Very poor
2 = Poor
3 = Average
4 = Good
5 = Excellent
"""


# ============================================================
# INTENT PROMPT FORMATTER
# ============================================================

def format_intent_prompt(customer_message):
    """
    Format the intent classification prompt.
    """

    if (
        not isinstance(customer_message, str)
        or not customer_message.strip()
    ):
        raise ValueError(
            "Customer message must be a non-empty string."
        )

    return INTENT_PROMPT.format(
        customer_message=customer_message.strip()
    )


# ============================================================
# REPLY PROMPT FORMATTER
# ============================================================

def format_reply_prompt(
    customer_message,
    intent,
    historical_cases
):
    """
    Format the customer support reply prompt.

    Historical cases are converted into readable text
    before being inserted into the prompt.
    """

    if (
        not isinstance(customer_message, str)
        or not customer_message.strip()
    ):
        raise ValueError(
            "Customer message must be a non-empty string."
        )

    if (
        not isinstance(intent, str)
        or not intent.strip()
    ):
        raise ValueError(
            "Intent must be a non-empty string."
        )

    formatted_cases = []

    if historical_cases:

        for index, result in enumerate(
            historical_cases,
            start=1
        ):

            document = result.get(
                "document",
                {}
            )

            score = result.get(
                "score",
                0.0
            )

            conversation_id = document.get(
                "conversation_id",
                "Unknown"
            )

            conversation_text = document.get(
                "text",
                ""
            )

            formatted_cases.append(
                f"""Historical Case {index}
Similarity Score: {float(score):.4f}
Conversation ID: {conversation_id}

{conversation_text}
"""
            )

    else:

        formatted_cases.append(
            "No relevant historical cases were found."
        )

    historical_cases_text = "\n".join(
        formatted_cases
    )

    return REPLY_PROMPT.format(
        customer_message=customer_message.strip(),
        intent=intent.strip(),
        historical_cases=historical_cases_text
    )


# ============================================================
# ESCALATION PROMPT FORMATTER
# ============================================================

def format_escalation_prompt(
    customer_message,
    intent,
    confidence
):
    """
    Format the escalation decision prompt.
    """

    if (
        not isinstance(customer_message, str)
        or not customer_message.strip()
    ):
        raise ValueError(
            "Customer message must be a non-empty string."
        )

    if (
        not isinstance(intent, str)
        or not intent.strip()
    ):
        raise ValueError(
            "Intent must be a non-empty string."
        )

    return ESCALATION_PROMPT.format(
        customer_message=customer_message.strip(),
        intent=intent.strip(),
        confidence=float(confidence)
    )


# ============================================================
# JUDGE PROMPT FORMATTER
# ============================================================

def format_judge_prompt(
    customer_message,
    intent,
    generated_response
):
    """
    Format the LLM-as-a-Judge prompt.
    """

    if (
        not isinstance(customer_message, str)
        or not customer_message.strip()
    ):
        raise ValueError(
            "Customer message must be a non-empty string."
        )

    if (
        not isinstance(intent, str)
        or not intent.strip()
    ):
        raise ValueError(
            "Intent must be a non-empty string."
        )

    if (
        not isinstance(generated_response, str)
        or not generated_response.strip()
    ):
        raise ValueError(
            "Generated response must be a non-empty string."
        )

    return JUDGE_PROMPT.format(
        customer_message=customer_message.strip(),
        intent=intent.strip(),
        generated_response=generated_response.strip()
    )