# src/intent/domain_features.py

import re


# ============================================================
# DOMAIN PATTERNS
# ============================================================

INTENT_PATTERNS = {

    "PLAN_CHANGE": [
        r"\bchange (my )?(plan|package|service)\b",
        r"\bchange (my )?current plan\b",
        r"\bupgrade (my )?(plan|service|package)\b",
        r"\bdowngrade (my )?(plan|service|package)\b",
        r"\bnew plan\b",
        r"\bdifferent plan\b",
        r"\bswitch (my )?plan\b",
        r"\bswitch to\b.*\bplan\b",
        r"\bmodify (my )?plan\b",
        r"\bchange package\b",
        r"\bupgrade\b",
        r"\bdowngrade\b",
        r"\badd (more )?internet\b",
        r"\bchange my internet package\b",
    ],

    "CANCELLATION": [
        r"\bcancel\b",
        r"\bcancellation\b",
        r"\bterminate (my )?(service|account)\b",
        r"\bclose (my )?account\b",
        r"\bdisconnect (my )?(service|internet)\b",
        r"\bstop (my )?(service|internet)\b",
        r"\bget rid of (my )?(service|internet)\b",
    ],

    "SERVICE_AVAILABILITY": [
        r"\bavailable\b",
        r"\bavailability\b",
        r"\bservice in (my )?(area|location)\b",
        r"\bservice at my address\b",
        r"\bavailable at my address\b",
        r"\bavailable in my area\b",
        r"\bserve my area\b",
        r"\bserve my address\b",
        r"\bdo you service\b",
        r"\bis service available\b",
        r"\bcan i get (comcast|internet|service)\b",
        r"\bnew address\b",
        r"\bnew home\b",
        r"\bnew apartment\b",
    ],

    # ========================================================
    # INTERNET OUTAGE
    # ========================================================

    "INTERNET_OUTAGE": [
        r"\binternet\s+(is|was|has been|went|went completely)\s+down\b",
        r"\binternet\s+(is|was|has been)\s+completely\s+down\b",
        r"\binternet\s+has\s+completely\s+gone\s+down\b",
        r"\binternet\s+completely\s+stopped\b",
        r"\binternet\s+stopped\s+working\b",
        r"\binternet\s+isn't\s+working\b",
        r"\binternet\s+is\s+not\s+working\b",
        r"\bno\s+internet\b",
        r"\bno\s+internet\s+connection\b",
        r"\bno\s+connection\b",
        r"\bconnection\s+is\s+down\b",
        r"\bservice\s+is\s+down\b",
        r"\bservice\s+went\s+down\b",
        r"\bnetwork\s+is\s+down\b",
        r"\bnetwork\s+went\s+down\b",
        r"\boutage\b",
        r"\btotal\s+outage\b",
        r"\bcompletely\s+offline\b",
        r"\bfully\s+offline\b",
        r"\bno\s+connectivity\b",
        r"\bwithout\s+internet\b",
        r"\bwithout\s+an?\s+internet\s+connection\b",
    ],

    # ========================================================
    # SLOW INTERNET
    # ========================================================

    "SLOW_INTERNET": [
        r"\binternet\s+is\s+slow\b",
        r"\binternet\s+is\s+very\s+slow\b",
        r"\binternet\s+is\s+extremely\s+slow\b",
        r"\binternet\s+too\s+slow\b",
        r"\bslow\s+internet\b",
        r"\bvery\s+slow\s+internet\b",
        r"\bextremely\s+slow\s+internet\b",
        r"\bslow\s+connection\b",
        r"\bconnection\s+is\s+slow\b",
        r"\bslow\s+speed\b",
        r"\bdownload\s+speed\b",
        r"\bupload\s+speed\b",
        r"\bspeed\s+is\s+slow\b",
        r"\bslow\s+speeds\b",
        r"\blagging\s+internet\b",
        r"\binternet\s+keeps\s+lagging\b",
        r"\bhigh\s+latency\b",
        r"\bpoor\s+internet\s+speed\b",
        r"\bslow\s+wifi\b",
        r"\bslow\s+wi-fi\b",
    ],

    # ========================================================
    # TECHNICAL ISSUE
    # ========================================================

    "TECHNICAL_ISSUE": [
        r"\brouter\b",
        r"\bmodem\b",
        r"\bwi[- ]?fi\b",
        r"\bwifi\b",
        r"\bactivation\b",
        r"\bactivate\b",
        r"\berror\b",
        r"\berror\s+message\b",
        r"\bnot\s+working\b",
        r"\bdoes\s+not\s+work\b",
        r"\bdoesn't\s+work\b",
        r"\bkeeps\s+restarting\b",
        r"\brestarts\b",
        r"\breboot\b",
        r"\bdisconnects\b",
        r"\bapp\b",
        r"\bwebsite\b",
        r"\blogin\b",
        r"\bconnect\b",
        r"\bconnection\s+problem\b",
        r"\btechnical\b",
        r"\btroubleshoot\b",
        r"\bdevice\b",
        r"\bsetup\b",
        r"\binstallation\b",
    ],

    # ========================================================
    # ACCOUNT ISSUE
    # ========================================================

    "ACCOUNT_ISSUE": [
        r"\baccount\b",
        r"\blog\s+into\s+my\s+account\b",
        r"\blogin\s+to\s+my\s+account\b",
        r"\bcannot\s+log\s+in\b",
        r"\bcan't\s+log\s+in\b",
        r"\bforgot\s+password\b",
        r"\bpassword\b",
        r"\busername\b",
        r"\baccount\s+information\b",
        r"\baccount\s+details\b",
        r"\baccount\s+access\b",
        r"\baccess\s+my\s+account\b",
        r"\baccount\s+changed\b",
        r"\breset\s+my\s+password\b",
    ],

    # ========================================================
    # PAYMENT ISSUE
    # ========================================================

    "PAYMENT_ISSUE": [
        r"\bpayment\s+declined\b",
        r"\bpayment\s+failed\b",
        r"\bpayment\s+was\s+declined\b",
        r"\bpayment\s+did\s+not\s+go\s+through\b",
        r"\bpayment\s+will\s+not\s+go\s+through\b",
        r"\bpayment\s+won't\s+go\s+through\b",
        r"\bcard\s+declined\b",
        r"\bcredit\s+card\s+declined\b",
        r"\bdebit\s+card\s+declined\b",
        r"\bmake\s+a\s+payment\b",
        r"\bpay\s+my\s+bill\b",
        r"\bpayment\s+problem\b",
        r"\bpayment\s+issue\b",
    ],

    # ========================================================
    # BILLING ISSUE
    # ========================================================

    "BILLING_ISSUE": [
        r"\bbill\b",
        r"\bbilling\b",
        r"\bcharged\b",
        r"\bovercharged\b",
        r"\bcharge\b",
        r"\bfee\b",
        r"\blate\s+fee\b",
        r"\bmonthly\s+bill\b",
        r"\bbill\s+is\s+wrong\b",
        r"\bwrong\s+bill\b",
        r"\bhigh\s+bill\b",
        r"\bdouble\s+charged\b",
        r"\bcharged\s+twice\b",
        r"\bunexpected\s+charge\b",
    ],

    # ========================================================
    # GENERAL INQUIRY
    # ========================================================

    "GENERAL_INQUIRY": [
        r"\bgeneral\s+question\b",
        r"\bquestion\s+about\b",
        r"\bjust\s+wondering\b",
        r"\bwant\s+to\s+know\b",
        r"\bcan\s+you\s+tell\s+me\b",
        r"\bwhat\s+is\b",
        r"\bhow\s+does\b",
        r"\binformation\s+about\b",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    contractions = {
        "can't": "cannot",
        "cant": "cannot",
        "won't": "will not",
        "wont": "will not",
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
    }

    for contraction, replacement in contractions.items():

        text = text.replace(
            contraction,
            replacement
        )

    text = " ".join(text.split())

    return text


# ============================================================
# MATCH PATTERNS
# ============================================================

def get_intent_matches(text):

    text = normalize_text(text)

    matches = {}

    for intent, patterns in INTENT_PATTERNS.items():

        matched_patterns = []

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                matched_patterns.append(pattern)

        if matched_patterns:

            matches[intent] = matched_patterns

    return matches


# ============================================================
# SCORE INTENTS
# ============================================================

def get_domain_scores(text):

    text = normalize_text(text)

    scores = {
        intent: 0.0
        for intent in INTENT_PATTERNS
    }

    matches = get_intent_matches(text)

    for intent, matched_patterns in matches.items():

        for pattern in matched_patterns:

            clean_pattern = pattern.replace(
                r"\b",
                ""
            )

            # Strong semantic phrases
            if intent == "INTERNET_OUTAGE":

                strong_outage_phrases = [
                    "internet is down",
                    "internet was down",
                    "internet has been down",
                    "internet completely down",
                    "internet is completely down",
                    "internet has completely gone down",
                    "internet completely stopped",
                    "no internet",
                    "no internet connection",
                    "service is down",
                    "network is down",
                    "total outage",
                    "completely offline",
                    "fully offline",
                    "without internet",
                    "internet stopped working",
                ]

                if any(
                    phrase in text
                    for phrase in strong_outage_phrases
                ):

                    scores[intent] += 3.0

                else:

                    scores[intent] += 2.0

            elif intent == "SLOW_INTERNET":

                strong_slow_phrases = [
                    "internet is slow",
                    "internet is very slow",
                    "internet is extremely slow",
                    "slow internet",
                    "very slow internet",
                    "slow connection",
                    "slow speed",
                    "slow speeds",
                ]

                if any(
                    phrase in text
                    for phrase in strong_slow_phrases
                ):

                    scores[intent] += 3.0

                else:

                    scores[intent] += 2.0

            elif intent == "PLAN_CHANGE":

                strong_plan_phrases = [
                    "change my plan",
                    "change my current plan",
                    "upgrade my plan",
                    "upgrade my internet plan",
                    "downgrade my plan",
                    "change package",
                    "new plan",
                    "different plan",
                    "switch my plan",
                ]

                if any(
                    phrase in text
                    for phrase in strong_plan_phrases
                ):

                    scores[intent] += 3.0

                else:

                    scores[intent] += 1.5

            elif len(clean_pattern) >= 20:

                scores[intent] += 2.0

            elif len(clean_pattern) >= 10:

                scores[intent] += 1.5

            else:

                scores[intent] += 1.0

    return scores


# ============================================================
# BEST DOMAIN INTENT
# ============================================================

def get_best_domain_intent(text):

    scores = get_domain_scores(text)

    if not scores:

        return {
            "intent": None,
            "score": 0.0,
            "scores": {},
        }

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    if best_score <= 0:

        best_intent = None

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

        "My internet has been completely down for the last two hours.",

        "My internet is completely down.",

        "Our internet is down again at home.",

        "We have no internet for three hours.",

        "My internet stopped working this morning.",

        "I want to change my current plan.",

        "I want to upgrade my internet plan.",

        "The internet is extremely slow.",

        "Can I get Comcast at my new address?",

        "My router is not working.",

        "Why is my bill so high?",

        "My payment was declined.",

        "I cannot log into my account.",

        "I want to cancel my service.",
    ]

    print("=" * 70)
    print("DOMAIN FEATURE TEST")
    print("=" * 70)

    for message in test_messages:

        result = get_best_domain_intent(
            message
        )

        print()

        print(
            f"Message : {message}"
        )

        print(
            f"Intent  : {result['intent']}"
        )

        print(
            f"Score   : {result['score']}"
        )

        print(
            f"Scores  : {result['scores']}"
        )