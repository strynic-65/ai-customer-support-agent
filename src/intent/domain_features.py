"""
Domain-specific feature extraction and rule-based intent scoring.

Supported intents:
- BILLING_ISSUE
- INTERNET_OUTAGE
- TECHNICAL_ISSUE
- SLOW_INTERNET
- ACCOUNT_ISSUE
- PLAN_CHANGE
- CANCELLATION
- PAYMENT_ISSUE
- SERVICE_AVAILABILITY
- GENERAL_INQUIRY
"""

import re


# ============================================================
# INTENT NAMES
# ============================================================

INTENTS = [
    "PLAN_CHANGE",
    "CANCELLATION",
    "SERVICE_AVAILABILITY",
    "INTERNET_OUTAGE",
    "SLOW_INTERNET",
    "TECHNICAL_ISSUE",
    "ACCOUNT_ISSUE",
    "PAYMENT_ISSUE",
    "BILLING_ISSUE",
    "GENERAL_INQUIRY",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize customer message for rule matching.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    contractions = {
        "can't": "cannot",
        "cant": "cannot",
        "won't": "will not",
        "wont": "will not",
        "wouldn't": "would not",
        "wouldnt": "would not",
        "don't": "do not",
        "dont": "do not",
        "doesn't": "does not",
        "doesnt": "does not",
        "didn't": "did not",
        "didnt": "did not",
        "isn't": "is not",
        "isnt": "is not",
        "aren't": "are not",
        "arent": "are not",
        "wasn't": "was not",
        "wasnt": "was not",
        "weren't": "were not",
        "werent": "were not",
        "couldn't": "could not",
        "couldnt": "could not",
        "shouldn't": "should not",
        "shouldnt": "should not",
    }

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _phrase_to_regex(phrase):
    """
    Convert a multi-word phrase into a flexible regex.

    Example:
        "why is my bill so high"

    Can match:
        "why is my bill so high"
        "why is my comcast bill so high"
        "why is my xfinity bill so high"
    """

    phrase = normalize_text(phrase)

    if not phrase:
        return None

    words = phrase.split()

    if len(words) == 1:
        return rf"\b{re.escape(words[0])}\b"

    return r"\b" + r"\s+".join(
        re.escape(word)
        for word in words
    ) + r"\b"


def _phrase_matches(text, phrase):
    """
    Return True when a phrase matches the text.

    First tries exact substring matching.

    For multi-word phrases, also allows a small number of
    words to appear between the important words.

    This prevents rules such as:
        "my bill is high"

    from failing on:
        "my Comcast bill is high"
    """

    if not isinstance(text, str):
        return False

    if not isinstance(phrase, str):
        return False

    text = normalize_text(text)
    phrase = normalize_text(phrase)

    if not text or not phrase:
        return False

    # Preserve existing exact matching behavior.
    if phrase in text:
        return True

    words = phrase.split()

    # Single-word phrases should remain exact.
    if len(words) == 1:
        return False

    # Allow up to 3 words between important phrase words.
    flexible_pattern = r"\b" + r"(?:\s+\w+){0,3}\s+".join(
        re.escape(word)
        for word in words
    ) + r"\b"

    return bool(re.search(flexible_pattern, text))


def _has_any(text, phrases):
    """
    Return True when at least one phrase occurs in text.
    """

    return any(
        _phrase_matches(text, phrase)
        for phrase in phrases
    )


def _count_matches(text, phrases):
    """
    Count the number of matching phrases.
    """

    return sum(
        1
        for phrase in phrases
        if _phrase_matches(text, phrase)
    )


# ============================================================
# PLAN CHANGE
# ============================================================

PLAN_CHANGE_PATTERNS = [
    "change my plan",
    "change current plan",
    "change my current plan",
    "change plan",
    "change my package",
    "change package",
    "upgrade my plan",
    "upgrade plan",
    "upgrade my internet",
    "upgrade internet",
    "upgrade service",
    "downgrade my plan",
    "downgrade plan",
    "downgrade my service",
    "downgrade service",
    "switch my plan",
    "switch plan",
    "new plan",
    "different plan",
    "add internet",
    "add a service",
    "add service",
    "remove a service",
    "change service",
]

PLAN_CHANGE_STRONG = [
    "i want to change my plan",
    "i want to upgrade my plan",
    "i want to upgrade my internet plan",
    "i want to downgrade my plan",
    "i want to downgrade my service",
    "i want a different plan",
    "can i change my plan",
    "can i upgrade my plan",
    "can i downgrade my plan",
]


# ============================================================
# CANCELLATION
# ============================================================

CANCELLATION_PATTERNS = [
    "cancel my service",
    "cancel service",
    "cancel my account",
    "cancel account",
    "cancel my internet",
    "cancel internet",
    "cancel my plan",
    "cancel plan",
    "terminate my service",
    "terminate service",
    "terminate my account",
    "close my account",
    "close account",
    "disconnect my service",
    "disconnect service",
    "stop my service",
    "stop service",
    "get rid of my service",
    "get rid of service",
    "end my service",
    "end service",
]

CANCELLATION_STRONG = [
    "i want to cancel my service",
    "i want to cancel my account",
    "please cancel my account",
    "please cancel my service",
    "i want to terminate my service",
    "i want to close my account",
    "i want to disconnect my service",
]


# ============================================================
# SERVICE AVAILABILITY
# ============================================================

SERVICE_AVAILABILITY_PATTERNS = [
    "is comcast available",
    "comcast available",
    "is internet available",
    "internet available in my area",
    "available in my area",
    "available in my neighborhood",
    "available at my address",
    "available at my new address",
    "service available",
    "service availability",
    "internet availability",
    "can i get comcast",
    "can i get internet",
    "can i get service",
    "new address",
    "new home",
    "new apartment",
    "new house",
    "moving to",
    "move to",
    "coverage in my area",
    "coverage area",
]

SERVICE_AVAILABILITY_STRONG = [
    "can i get comcast at my new address",
    "can i get comcast at my address",
    "is comcast available in my area",
    "is comcast available at my address",
    "is internet available in my area",
    "is your service available in my area",
    "do you provide service in my area",
    "do you provide internet at my address",
    "can i get internet at my new address",
]


# ============================================================
# INTERNET OUTAGE
# ============================================================

INTERNET_OUTAGE_PATTERNS = [
    "internet is down",
    "internet down",
    "my internet is down",
    "our internet is down",
    "internet stopped working",
    "internet has stopped working",
    "internet not working",
    "no internet",
    "no internet connection",
    "lost internet",
    "internet outage",
    "service outage",
    "outage in my area",
    "connection is down",
    "connection down",
    "wifi is down",
    "wifi down",
    "wifi stopped working",
    "wifi has stopped working",
]

INTERNET_OUTAGE_STRONG = [
    "internet is completely down",
    "internet has been completely down",
    "my internet has been completely down",
    "internet is down",
    "my internet is down",
    "our internet is down",
    "we have no internet",
    "no internet for",
    "internet stopped working",
    "internet has stopped working",
    "internet outage",
    "service outage",
    "wifi is completely down",
]


# ============================================================
# SLOW INTERNET
# ============================================================

SLOW_INTERNET_PATTERNS = [
    "internet is slow",
    "internet is very slow",
    "internet is extremely slow",
    "internet too slow",
    "slow internet",
    "very slow internet",
    "extremely slow internet",
    "slow connection",
    "slow speed",
    "slow speeds",
    "slow wifi",
    "wifi is slow",
    "wifi is very slow",
    "wifi is extremely slow",
    "wi fi is slow",
    "wi fi is very slow",
    "internet speed",
    "poor internet speed",
    "slow download",
    "slow upload",
    "download is slow",
    "upload is slow",
    "high latency",
    "high ping",
    "lagging internet",
    "internet lagging",
]

SLOW_INTERNET_STRONG = [
    "internet is slow",
    "internet is very slow",
    "internet is extremely slow",
    "very slow internet",
    "extremely slow internet",
    "slow internet",
    "slow connection",
    "slow speeds",
    "slow wifi",
    "wifi is slow",
    "wifi is very slow",
    "wifi is extremely slow",
    "wi fi is slow",
    "wi fi is very slow",
    "slow download",
    "slow upload",
    "download is slow",
    "upload is slow",
]


# ============================================================
# TECHNICAL ISSUE
# ============================================================

TECHNICAL_ISSUE_PATTERNS = [
    "router",
    "modem",
    "gateway",
    "equipment",
    "device setup",
    "installation",
    "activate my modem",
    "activate modem",
    "activate router",
    "activation",
    "error code",
    "error message",
    "error",
    "troubleshoot",
    "troubleshooting",
    "technical problem",
    "technical issue",
    "technical support",
    "reboot",
    "restart router",
    "restart modem",
    "reset router",
    "reset modem",
    "wifi router",
    "network equipment",
    "cannot connect device",
    "cannot connect my device",
    "device cannot connect",
]

TECHNICAL_ISSUE_STRONG = [
    "my router is not working",
    "my modem is not working",
    "my router stopped working",
    "my modem stopped working",
    "router is not working",
    "modem is not working",
    "my modem has an error",
    "my router has an error",
    "error code",
    "error message",
    "need technical support",
    "technical issue",
    "technical problem",
    "troubleshoot my router",
    "troubleshoot my modem",
    "activate my modem",
    "activate my router",
    "install my modem",
    "install my router",
]


# ============================================================
# ACCOUNT ISSUE
# ============================================================

ACCOUNT_ISSUE_PATTERNS = [
    "my account",
    "account issue",
    "account problem",
    "account access",
    "account information",
    "account details",
    "account settings",
    "account password",
    "forgot password",
    "forgot my password",
    "reset password",
    "change password",
    "cannot log in",
    "cannot login",
    "cannot access my account",
    "unable to log in",
    "unable to login",
    "login problem",
    "login issue",
    "sign in problem",
    "sign in issue",
    "account locked",
    "locked out of my account",
    "someone accessed my account",
]

ACCOUNT_ISSUE_STRONG = [
    "i cannot log into my account",
    "i cannot login to my account",
    "i cannot access my account",
    "unable to access my account",
    "unable to log into my account",
    "forgot my password",
    "forgot password",
    "reset my password",
    "change my password",
    "my account is locked",
    "locked out of my account",
    "someone accessed my account",
]


# ============================================================
# PAYMENT ISSUE
# ============================================================

PAYMENT_ISSUE_PATTERNS = [
    "payment failed",
    "payment was declined",
    "payment declined",
    "card payment failed",
    "card was declined",
    "credit card declined",
    "debit card declined",
    "payment error",
    "payment problem",
    "payment issue",
    "cannot make payment",
    "cannot pay",
    "unable to make payment",
    "payment did not go through",
    "payment will not go through",
    "autopay failed",
    "autopay problem",
    "autopay issue",
]

PAYMENT_ISSUE_STRONG = [
    "my payment was declined",
    "my payment failed",
    "card payment failed",
    "my card payment failed",
    "credit card was declined",
    "debit card was declined",
    "cannot make a payment",
    "cannot make payment",
    "unable to make payment",
    "payment did not go through",
    "autopay failed",
]


# ============================================================
# BILLING ISSUE
# ============================================================

BILLING_ISSUE_PATTERNS = [
    "my bill",
    "bill is high",
    "bill too high",
    "bill is wrong",
    "wrong bill",
    "billing issue",
    "billing problem",
    "billing error",
    "billing question",
    "billing charge",
    "unexpected charge",
    "extra charge",
    "charged twice",
    "double charged",
    "overcharged",
    "incorrect charge",
    "incorrect bill",
    "higher bill",
    "high bill",
    "price on my bill",
    "charge on my bill",
    "monthly bill",
    "invoice",
]

BILLING_ISSUE_STRONG = [
    "why is my bill so high",
    "my bill is too high",
    "my bill is wrong",
    "i was charged twice",
    "charged twice",
    "double charged",
    "incorrect charge",
    "unexpected charge",
    "billing error",
    "billing issue",
    "wrong bill",
]


# ============================================================
# GENERAL INQUIRY
# ============================================================

GENERAL_INQUIRY_PATTERNS = [
    "tell me about your services",
    "tell me about your service",
    "what services do you offer",
    "what do you offer",
    "general question",
    "general inquiry",
    "i have a question",
    "just a question",
    "more information",
    "information about your services",
    "information about your service",
]


# ============================================================
# GET INTENT MATCHES
# ============================================================

def get_intent_matches(text):
    """
    Return matched phrases for each intent.
    """

    text = normalize_text(text)

    patterns = {
        "PLAN_CHANGE": PLAN_CHANGE_PATTERNS,
        "CANCELLATION": CANCELLATION_PATTERNS,
        "SERVICE_AVAILABILITY": SERVICE_AVAILABILITY_PATTERNS,
        "INTERNET_OUTAGE": INTERNET_OUTAGE_PATTERNS,
        "SLOW_INTERNET": SLOW_INTERNET_PATTERNS,
        "TECHNICAL_ISSUE": TECHNICAL_ISSUE_PATTERNS,
        "ACCOUNT_ISSUE": ACCOUNT_ISSUE_PATTERNS,
        "PAYMENT_ISSUE": PAYMENT_ISSUE_PATTERNS,
        "BILLING_ISSUE": BILLING_ISSUE_PATTERNS,
        "GENERAL_INQUIRY": GENERAL_INQUIRY_PATTERNS,
    }

    matches = {}

    for intent, intent_patterns in patterns.items():
        matches[intent] = [
            phrase
            for phrase in intent_patterns
            if _phrase_matches(text, phrase)
        ]

    return matches


# ============================================================
# DOMAIN SCORES
# ============================================================

def get_domain_scores(text):
    """
    Calculate domain-specific intent scores.
    """

    text = normalize_text(text)

    scores = {
        intent: 0.0
        for intent in INTENTS
    }

    # --------------------------------------------------------
    # PLAN CHANGE
    # --------------------------------------------------------

    plan_matches = _count_matches(
        text,
        PLAN_CHANGE_PATTERNS
    )

    scores["PLAN_CHANGE"] += min(
        plan_matches * 2.0,
        6.0
    )

    if _has_any(text, PLAN_CHANGE_STRONG):
        scores["PLAN_CHANGE"] += 4.0

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    cancellation_matches = _count_matches(
        text,
        CANCELLATION_PATTERNS
    )

    scores["CANCELLATION"] += min(
        cancellation_matches * 2.0,
        6.0
    )

    if _has_any(text, CANCELLATION_STRONG):
        scores["CANCELLATION"] += 4.0

    # --------------------------------------------------------
    # SERVICE AVAILABILITY
    # --------------------------------------------------------

    availability_matches = _count_matches(
        text,
        SERVICE_AVAILABILITY_PATTERNS
    )

    scores["SERVICE_AVAILABILITY"] += min(
        availability_matches * 2.0,
        6.0
    )

    if _has_any(text, SERVICE_AVAILABILITY_STRONG):
        scores["SERVICE_AVAILABILITY"] += 4.0

    # --------------------------------------------------------
    # INTERNET OUTAGE
    # --------------------------------------------------------

    outage_matches = _count_matches(
        text,
        INTERNET_OUTAGE_PATTERNS
    )

    scores["INTERNET_OUTAGE"] += min(
        outage_matches * 2.0,
        6.0
    )

    if _has_any(text, INTERNET_OUTAGE_STRONG):
        scores["INTERNET_OUTAGE"] += 3.0

    # --------------------------------------------------------
    # SLOW INTERNET
    # --------------------------------------------------------

    slow_matches = _count_matches(
        text,
        SLOW_INTERNET_PATTERNS
    )

    scores["SLOW_INTERNET"] += min(
        slow_matches * 2.0,
        6.0
    )

    if _has_any(text, SLOW_INTERNET_STRONG):
        scores["SLOW_INTERNET"] += 3.0

    # --------------------------------------------------------
    # TECHNICAL ISSUE
    # --------------------------------------------------------

    technical_matches = _count_matches(
        text,
        TECHNICAL_ISSUE_PATTERNS
    )

    scores["TECHNICAL_ISSUE"] += min(
        technical_matches * 1.0,
        4.0
    )

    if _has_any(text, TECHNICAL_ISSUE_STRONG):
        scores["TECHNICAL_ISSUE"] += 3.0

    # --------------------------------------------------------
    # ACCOUNT ISSUE
    # --------------------------------------------------------

    account_matches = _count_matches(
        text,
        ACCOUNT_ISSUE_PATTERNS
    )

    scores["ACCOUNT_ISSUE"] += min(
        account_matches * 1.0,
        4.0
    )

    if _has_any(text, ACCOUNT_ISSUE_STRONG):
        scores["ACCOUNT_ISSUE"] += 3.0

    # --------------------------------------------------------
    # PAYMENT ISSUE
    # --------------------------------------------------------

    payment_matches = _count_matches(
        text,
        PAYMENT_ISSUE_PATTERNS
    )

    scores["PAYMENT_ISSUE"] += min(
        payment_matches * 2.0,
        6.0
    )

    if _has_any(text, PAYMENT_ISSUE_STRONG):
        scores["PAYMENT_ISSUE"] += 4.0

    # --------------------------------------------------------
    # BILLING ISSUE
    # --------------------------------------------------------

    billing_matches = _count_matches(
        text,
        BILLING_ISSUE_PATTERNS
    )

    scores["BILLING_ISSUE"] += min(
        billing_matches * 1.5,
        6.0
    )

    if _has_any(text, BILLING_ISSUE_STRONG):
        scores["BILLING_ISSUE"] += 3.5

    # --------------------------------------------------------
    # GENERAL INQUIRY
    # --------------------------------------------------------

    general_matches = _count_matches(
        text,
        GENERAL_INQUIRY_PATTERNS
    )

    scores["GENERAL_INQUIRY"] += min(
        general_matches * 2.0,
        4.0
    )

    # ========================================================
    # CONFLICT RESOLUTION
    # ========================================================

    # --------------------------------------------------------
    # PLAN CHANGE
    # --------------------------------------------------------

    if _has_any(text, PLAN_CHANGE_STRONG):
        scores["PLAN_CHANGE"] += 2.0

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    if _has_any(text, CANCELLATION_STRONG):
        scores["CANCELLATION"] += 2.0

        scores["ACCOUNT_ISSUE"] = min(
            scores["ACCOUNT_ISSUE"],
            1.0
        )

    # --------------------------------------------------------
    # SERVICE AVAILABILITY
    # --------------------------------------------------------

    if _has_any(text, SERVICE_AVAILABILITY_STRONG):
        scores["SERVICE_AVAILABILITY"] += 2.0

        scores["TECHNICAL_ISSUE"] = min(
            scores["TECHNICAL_ISSUE"],
            1.0
        )

    # --------------------------------------------------------
    # INTERNET OUTAGE
    # --------------------------------------------------------

    if _has_any(
        text,
        [
            "internet is down",
            "my internet is down",
            "our internet is down",
            "internet is completely down",
            "internet has been completely down",
            "my internet has been completely down",
            "we have no internet",
            "no internet for",
            "internet stopped working",
            "internet has stopped working",
            "internet outage",
            "service outage",
            "wifi is completely down",
        ]
    ):
        scores["INTERNET_OUTAGE"] += 2.0

        scores["TECHNICAL_ISSUE"] = min(
            scores["TECHNICAL_ISSUE"],
            1.0
        )

    # --------------------------------------------------------
    # SLOW INTERNET
    # --------------------------------------------------------

    if _has_any(
        text,
        [
            "internet is slow",
            "internet is very slow",
            "internet is extremely slow",
            "very slow internet",
            "extremely slow internet",
            "slow internet",
            "slow connection",
            "slow speeds",
            "slow wifi",
            "wifi is slow",
            "wifi is very slow",
            "wifi is extremely slow",
            "wi fi is slow",
            "wi fi is very slow",
            "internet speed",
            "slow download",
            "slow upload",
            "download is slow",
            "upload is slow",
        ]
    ):
        scores["SLOW_INTERNET"] += 2.0

        scores["TECHNICAL_ISSUE"] = min(
            scores["TECHNICAL_ISSUE"],
            1.0
        )

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if _has_any(text, PAYMENT_ISSUE_STRONG):
        scores["PAYMENT_ISSUE"] += 2.0

        scores["BILLING_ISSUE"] = min(
            scores["BILLING_ISSUE"],
            1.0
        )

    # --------------------------------------------------------
    # BILLING
    # --------------------------------------------------------

    if _has_any(text, BILLING_ISSUE_STRONG):
        scores["BILLING_ISSUE"] += 2.0

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    if _has_any(text, ACCOUNT_ISSUE_STRONG):
        scores["ACCOUNT_ISSUE"] += 2.0

        scores["TECHNICAL_ISSUE"] = min(
            scores["TECHNICAL_ISSUE"],
            1.0
        )

    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------

    if _has_any(text, TECHNICAL_ISSUE_STRONG):
        scores["TECHNICAL_ISSUE"] += 2.0

    return scores


# ============================================================
# BEST DOMAIN INTENT
# ============================================================

def get_best_domain_intent(text):
    """
    Return the highest-scoring domain intent.

    Returns:
        {
            "intent": "...",
            "score": float,
            "scores": {...}
        }
    """

    scores = get_domain_scores(text)

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    return {
        "intent": best_intent,
        "score": float(best_score),
        "scores": scores,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_messages = [

        # INTERNET OUTAGE
        "My internet has been completely down for the last two hours.",
        "My internet is completely down.",
        "Our internet is down again at home.",
        "We have no internet for three hours.",
        "My internet stopped working this morning.",

        # PLAN CHANGE
        "I want to change my current plan.",
        "I want to upgrade my internet plan.",
        "I want to downgrade my service.",

        # SLOW INTERNET
        "The internet is extremely slow.",
        "My wifi is very slow.",

        # SERVICE AVAILABILITY
        "Can I get Comcast at my new address?",
        "Is Comcast available in my area?",

        # TECHNICAL
        "My router is not working.",
        "My modem has an error.",

        # BILLING
        "Why is my bill so high?",
        "Why is my Comcast bill so high this month?",
        "My Comcast bill is too high.",
        "I was charged twice.",

        # PAYMENT
        "My payment was declined.",
        "My card payment failed.",

        # ACCOUNT
        "I cannot log into my account.",
        "I forgot my password.",

        # CANCELLATION
        "I want to cancel my service.",
        "Please cancel my account.",

        # GENERAL
        "Can you tell me about your services?",
    ]

    print("=" * 70)
    print("DOMAIN FEATURE TEST")
    print("=" * 70)

    for message in test_messages:

        result = get_best_domain_intent(message)

        print()
        print("Message :", message)
        print("Intent  :", result["intent"])
        print("Score   :", result["score"])
        print("Scores  :", result["scores"])