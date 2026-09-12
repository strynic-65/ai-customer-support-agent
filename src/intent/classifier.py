import re


INTENTS = [
    "BILLING_ISSUE",
    "INTERNET_OUTAGE",
    "TECHNICAL_ISSUE",
    "SLOW_INTERNET",
    "ACCOUNT_ISSUE",
    "PLAN_CHANGE",
    "CANCELLATION",
    "PAYMENT_ISSUE",
    "SERVICE_AVAILABILITY",
    "GENERAL_INQUIRY",
]


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Normalize common contractions BEFORE removing punctuation
    contractions = {
        "can't": "cant",
        "cannot": "cannot",
        "won't": "wont",
        "wouldn't": "wouldnt",
        "couldn't": "couldnt",
        "shouldn't": "shouldnt",
        "doesn't": "doesnt",
        "don't": "dont",
        "didn't": "didnt",
        "isn't": "isnt",
        "aren't": "arent",
        "wasn't": "wasnt",
        "weren't": "werent",
        "hasn't": "hasnt",
        "haven't": "havent",
        "hadn't": "hadnt",
        "i'm": "im",
        "i've": "ive",
        "i'll": "ill",
        "i'd": "id",
        "you're": "youre",
        "you've": "youve",
        "you'll": "youll",
        "we're": "were",
        "we've": "weve",
        "they're": "theyre",
        "it's": "its",
        "that's": "thats",
        "what's": "whats",
        "there's": "theres",
    }

    for old, new in contractions.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


# ============================================================
# CANCELLATION
# ============================================================

CANCELLATION_STRONG = [
    "want to cancel",
    "trying to cancel",
    "try to cancel",
    "need to cancel",
    "cancel my service",
    "cancel my account",
    "cancel my subscription",
    "cancel service",
    "cancel subscription",
    "cancel account",
    "cancel appointment",
    "cancelled",
    "canceled",
    "cancelling",
    "canceling",
    "going to cancel",
    "i am cancelling",
    "im cancelling",
    "i am canceling",
    "im canceling",
    "want cancellation",
]


# ============================================================
# PAYMENT
# ============================================================

PAYMENT_STRONG = [
    "payment was declined",
    "payment declined",
    "payment failed",
    "payment failure",
    "failed payment",
    "declined payment",
    "cannot make payment",
    "cant make payment",
    "could not make payment",
    "couldnt make payment",
    "unable to make payment",
    "trying to pay",
    "try to pay",
    "pay my bill",
    "make a payment",
    "payment returned",
    "payment was returned",
    "payment rejected",
    "payment not accepted",
    "card was declined",
    "card declined",
    "credit card declined",
    "debit card declined",
    "automatic payment",
    "autopay",
    "auto pay",
    "money drafted",
    "money was drafted",
]


# ============================================================
# ACCOUNT
# ============================================================

ACCOUNT_STRONG = [
    "someone changed my account",
    "account was changed",
    "account has been changed",
    "someone accessed my account",
    "unauthorized access",
    "hacked account",
    "account hacked",
    "account security",
    "security issue",
    "security concern",
    "identity theft",
    "phishing",
    "personal information",
    "personal info",
    "account locked",
    "locked out of my account",
    "cant access my account",
    "cannot access my account",
    "unable to access my account",
    "login problem",
    "login issue",
    "sign in problem",
    "sign in issue",
]


# ============================================================
# BILLING
# ============================================================

BILLING_STRONG = [
    "my bill",
    "the bill",
    "billing issue",
    "billing problem",
    "bill is higher",
    "bill was higher",
    "higher bill",
    "higher price",
    "price increased",
    "price increase",
    "charged extra",
    "extra charge",
    "wrong charge",
    "incorrect charge",
    "unexpected charge",
    "monthly charge",
    "monthly bill",
    "billing amount",
    "late fee",
    "fee on my bill",
    "contract price",
    "contract prices",
    "price went up",
    "prices went up",
]


# ============================================================
# INTERNET OUTAGE
# ============================================================

OUTAGE_STRONG = [
    "internet is down",
    "internet completely down",
    "internet went down",
    "internet has gone down",
    "internet stopped working",
    "internet stopped",
    "internet is out",
    "internet went out",
    "internet has been out",
    "wifi is down",
    "wifi completely down",
    "wifi went down",
    "wifi has gone down",
    "wifi is out",
    "wifi went out",
    "wifi has been out",
    "connection is down",
    "connection went down",
    "connection is out",
    "service is down",
    "service went down",
    "service is out",
    "network is down",
    "network went down",
    "network is out",
    "no internet",
    "no wifi",
    "internet completely stopped",
    "internet suddenly stopped",
]


def is_outage(text):
    # Strong phrases
    if contains_any(text, OUTAGE_STRONG):
        return True

    # Generic outage patterns
    has_internet = any(
        word in text for word in [
            "internet",
            "wifi",
            "wi fi",
            "connection",
            "network",
        ]
    )

    outage_words = [
        "down",
        "out",
        "dropped",
        "gone",
        "stopped",
        "crashed",
        "not working",
        "isnt working",
        "wont work",
    ]

    return has_internet and contains_any(text, outage_words)


# ============================================================
# SLOW INTERNET
# ============================================================

SLOW_STRONG = [
    "internet is slow",
    "internet so slow",
    "internet very slow",
    "internet extremely slow",
    "internet incredibly slow",
    "internet speed is slow",
    "internet speed so slow",
    "internet speed very slow",
    "internet speed extremely slow",
    "internet speed problem",
    "internet speed issue",
    "internet speed issues",
    "slow internet",
    "slow wifi",
    "wifi is slow",
    "wifi very slow",
    "wifi extremely slow",
    "download speed",
    "upload speed",
    "speed is slow",
    "speed is terrible",
    "speed sucks",
    "speeds are slow",
    "speeds so slow",
    "speeds worse",
    "speed problem",
    "speed issue",
    "slow connection",
    "internet slowing down",
    "internet slowdown",
    "throttling speeds",
    "throttling speed",
]


def is_slow_internet(text):
    if contains_any(text, SLOW_STRONG):
        return True

    # Numeric speed evidence
    speed_pattern = r"\b\d+\s*(mbps|mb|gbps|gb)\b"

    if re.search(speed_pattern, text):
        if any(word in text for word in ["internet", "speed", "download", "upload"]):
            return True

    return False


# ============================================================
# SERVICE AVAILABILITY
# ============================================================

AVAILABILITY_STRONG = [
    "is comcast available",
    "comcast available",
    "is xfinity available",
    "xfinity available",
    "available at my address",
    "available at my new address",
    "available at this address",
    "available in my area",
    "available in my location",
    "service available",
    "service availability",
    "do you service my area",
    "do you service this area",
    "do you service my address",
    "do you cover my area",
    "coverage in my area",
    "coverage at my address",
    "coverage available",
    "does comcast service",
    "does xfinity service",
    "does comcast offer",
    "does xfinity offer",
    "comcast offer",
    "xfinity offer",
    "comcast carry",
    "xfinity carry",
    "doesnt carry",
    "dont carry",
    "not available",
    "not offered",
    "offer this service",
    "new address",
    "new home",
    "new location",
]


def is_service_availability(text):
    if contains_any(text, AVAILABILITY_STRONG):
        return True

    # Address + available/service/coverage
    if "address" in text and any(
        word in text for word in ["available", "service", "coverage", "offer"]
    ):
        return True

    # Area + availability
    if "area" in text and any(
        word in text for word in ["available", "service", "coverage", "offer"]
    ):
        return True

    return False


# ============================================================
# PLAN CHANGE
# ============================================================

PLAN_STRONG = [
    "change my plan",
    "change the plan",
    "change plan",
    "upgrade my plan",
    "upgrade the plan",
    "downgrade my plan",
    "downgrade the plan",
    "new plan",
    "different plan",
    "available plans",
    "what plans",
    "plan options",
    "switch my plan",
    "switch plan",
    "upgrade service",
    "downgrade service",
    "upgrade my service",
    "add internet",
    "add cable",
    "add tv",
    "remove channels",
    "cancel channels",
    "change channels",
    "unlimited data plan",
]


# ============================================================
# TECHNICAL ISSUE
# ============================================================

TECHNICAL_STRONG = [
    "modem is not working",
    "modem not working",
    "router is not working",
    "router not working",
    "equipment not working",
    "device not working",
    "internet equipment",
    "email is not working",
    "email not working",
    "email doesnt work",
    "email wont work",
    "cant connect",
    "cannot connect",
    "unable to connect",
    "connection problem",
    "connection issue",
    "technical issue",
    "technical problem",
    "broken",
    "broken picture",
    "blank screen",
    "channel is broken",
    "channel problem",
    "channel issue",
    "picture problem",
    "picture is broken",
    "hbo isnt working",
    "app isnt working",
    "app not working",
    "website isnt working",
    "website not working",
    "equipment problem",
    "equipment issue",
    "wont connect",
    "doesnt work",
    "not working",
    "non working",
    "nonworking",
    "rarely get service",
    "paying for service i rarely get",
    "wifi trash",
]


def is_technical(text):
    if contains_any(text, TECHNICAL_STRONG):
        return True

    # Connection failures are technical unless clearly a full outage
    if any(word in text for word in ["modem", "router", "email", "hbo"]):
        if any(
            word in text
            for word in [
                "problem",
                "issue",
                "broken",
                "not working",
                "doesnt work",
                "wont work",
                "cant connect",
                "cannot connect",
            ]
        ):
            return True

    return False


# ============================================================
# GENERAL
# ============================================================

def classify_intent(text):
    text = normalize_text(text)

    if not text:
        return {
            "intent": "GENERAL_INQUIRY",
            "confidence": 0.40,
        }

    # --------------------------------------------------------
    # 1. Explicit cancellation
    # --------------------------------------------------------
    if contains_any(text, CANCELLATION_STRONG):

        # Conditional "would cancel if..." should not automatically
        # override the actual service problem.
        if "would cancel if" in text:
            if is_outage(text):
                return {
                    "intent": "INTERNET_OUTAGE",
                    "confidence": 0.91,
                }

        # "might cancel" usually describes a consequence,
        # not necessarily the primary intent.
        if "might cancel" in text:
            if contains_any(text, PLAN_STRONG):
                return {
                    "intent": "PLAN_CHANGE",
                    "confidence": 0.78,
                }

        return {
            "intent": "CANCELLATION",
            "confidence": 0.94,
        }

    # --------------------------------------------------------
    # 2. Payment
    # --------------------------------------------------------
    if contains_any(text, PAYMENT_STRONG):
        return {
            "intent": "PAYMENT_ISSUE",
            "confidence": 0.92,
        }

    # --------------------------------------------------------
    # 3. Security / account
    # --------------------------------------------------------
    if contains_any(text, ACCOUNT_STRONG):
        return {
            "intent": "ACCOUNT_ISSUE",
            "confidence": 0.92,
        }

    # --------------------------------------------------------
    # 4. Billing
    # --------------------------------------------------------
    if contains_any(text, BILLING_STRONG):
        return {
            "intent": "BILLING_ISSUE",
            "confidence": 0.87,
        }

    # --------------------------------------------------------
    # 5. Full internet outage
    # --------------------------------------------------------
    if is_outage(text):
        return {
            "intent": "INTERNET_OUTAGE",
            "confidence": 0.93,
        }

    # --------------------------------------------------------
    # 6. Slow internet
    # --------------------------------------------------------
    if is_slow_internet(text):
        return {
            "intent": "SLOW_INTERNET",
            "confidence": 0.92,
        }

    # --------------------------------------------------------
    # 7. Service availability
    # --------------------------------------------------------
    if is_service_availability(text):
        return {
            "intent": "SERVICE_AVAILABILITY",
            "confidence": 0.91,
        }

    # --------------------------------------------------------
    # 8. Plan change
    # --------------------------------------------------------
    if contains_any(text, PLAN_STRONG):
        return {
            "intent": "PLAN_CHANGE",
            "confidence": 0.91,
        }

    # --------------------------------------------------------
    # 9. Technical issue
    # --------------------------------------------------------
    if is_technical(text):
        return {
            "intent": "TECHNICAL_ISSUE",
            "confidence": 0.88,
        }

    # --------------------------------------------------------
    # 10. General inquiry
    # --------------------------------------------------------
    return {
        "intent": "GENERAL_INQUIRY",
        "confidence": 0.40,
    }


def confidence_level(confidence):
    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def should_escalate(confidence):
    return confidence < 0.50


def analyze_confidence(result):
    confidence = result.get("confidence", 0.0)

    return {
        "intent": result.get("intent", "GENERAL_INQUIRY"),
        "confidence": confidence,
        "level": confidence_level(confidence),
        "should_escalate": should_escalate(confidence),
    }


if __name__ == "__main__":

    test_cases = [
        "I want to cancel my service.",
        "I'm trying to cancel my subscription.",
        "My internet is completely down.",
        "My WiFi has been out for over an hour.",
        "Why is my internet speed so slow?",
        "My download speed is only 3 Mbps.",
        "Someone changed my account.",
        "My payment was declined.",
        "My bill is much higher this month.",
        "Can I change my plan?",
        "Is Comcast available at my new address?",
        "My modem is not working.",
        "My email is not working.",
        "I am paying for service I rarely get.",
        "Can you tell me about your services?",
    ]

    print("=" * 70)
    print("COMCAST INTENT CLASSIFIER TEST")
    print("=" * 70)

    for message in test_cases:
        result = classify_intent(message)

        print()
        print(f"Message    : {message}")
        print(f"Intent     : {result['intent']}")
        print(f"Confidence : {result['confidence']:.2f}")