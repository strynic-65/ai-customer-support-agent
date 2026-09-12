"""
Risk rules for deciding whether a customer-support
request should be escalated to a human agent.
"""


# Keywords indicating possible security/privacy concerns
SECURITY_KEYWORDS = [
    "hacked",
    "fraud",
    "stolen",
    "identity theft",
    "security",
    "unauthorized access",
    "someone accessed my account",
]


# Keywords indicating strong frustration or threats
FRUSTRATION_KEYWORDS = [
    "angry",
    "furious",
    "frustrated",
    "ridiculous",
    "worst service",
    "terrible service",
    "lawsuit",
    "sue",
    "legal action",
]


# Customer explicitly requesting a human
HUMAN_REQUEST_KEYWORDS = [
    "human",
    "agent",
    "representative",
    "real person",
    "speak to someone",
    "talk to someone",
    "customer service representative",
]


def contains_keyword(text, keywords):
    """
    Check whether any keyword appears in the text.
    """

    if not isinstance(text, str):
        return False

    text = text.lower()

    return any(keyword in text for keyword in keywords)


def check_security_risk(text):
    """
    Detect possible security or privacy concerns.
    """

    return contains_keyword(text, SECURITY_KEYWORDS)


def check_frustration(text):
    """
    Detect strong customer frustration.
    """

    return contains_keyword(text, FRUSTRATION_KEYWORDS)


def requests_human(text):
    """
    Detect whether the customer explicitly wants
    a human support representative.
    """

    return contains_keyword(text, HUMAN_REQUEST_KEYWORDS)


def detect_risks(text):
    """
    Return all detected risk categories.
    """

    return {
        "security_risk": check_security_risk(text),
        "frustration": check_frustration(text),
        "human_requested": requests_human(text),
    }