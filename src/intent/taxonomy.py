INTENTS = {
    "BILLING_ISSUE": {
        "description": "Customer reports an incorrect, unexpected, or disputed bill or charge.",
        "examples": [
            "Why is my bill higher this month?",
            "I was charged extra.",
            "There is a charge on my account I don't recognize."
        ]
    },

    "INTERNET_OUTAGE": {
        "description": "Customer reports that internet or service is completely unavailable.",
        "examples": [
            "My internet is down.",
            "There is no internet connection.",
            "When will the outage be fixed?"
        ]
    },

    "TECHNICAL_ISSUE": {
        "description": "Customer has a technical problem with Wi-Fi, modem, router, or connection.",
        "examples": [
            "My router isn't working.",
            "Wi-Fi keeps disconnecting.",
            "My modem stopped working."
        ]
    },

    "SLOW_INTERNET": {
        "description": "Customer reports slow internet speed or poor performance.",
        "examples": [
            "My internet is very slow.",
            "Why am I not getting my advertised speed?",
            "The connection is too slow."
        ]
    },

    "ACCOUNT_ISSUE": {
        "description": "Customer has a problem with their account, login, credentials, or account information.",
        "examples": [
            "I can't log into my account.",
            "I need to update my account information.",
            "I can't access my account."
        ]
    },

    "PLAN_CHANGE": {
        "description": "Customer wants to upgrade, downgrade, or change their service plan.",
        "examples": [
            "I want to upgrade my internet.",
            "Can I change my plan?",
            "What plans do you offer?"
        ]
    },

    "CANCELLATION": {
        "description": "Customer wants to cancel or terminate their service.",
        "examples": [
            "I want to cancel my service.",
            "How do I disconnect my account?",
            "I don't want Comcast anymore."
        ]
    },

    "PAYMENT_ISSUE": {
        "description": "Customer reports a failed, declined, or problematic payment.",
        "examples": [
            "My payment failed.",
            "Why was my card declined?",
            "I can't pay my bill."
        ]
    },

    "SERVICE_AVAILABILITY": {
        "description": "Customer asks whether Comcast service is available at a particular location.",
        "examples": [
            "Is Comcast available in my area?",
            "Do you provide service at my address?",
            "Can I get internet at this location?"
        ]
    },

    "GENERAL_INQUIRY": {
        "description": "Customer question or request that does not clearly fit another intent.",
        "examples": [
            "What are your support hours?",
            "I have a question.",
            "Can you help me?"
        ]
    }
}


def get_intent_names():
    return list(INTENTS.keys())


def get_intent_description(intent):
    return INTENTS[intent]["description"]