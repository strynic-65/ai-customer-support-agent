def confidence_level(confidence):
    """
    Convert a numeric confidence score into a readable level.
    """

    if confidence >= 0.80:
        return "HIGH"

    if confidence >= 0.50:
        return "MEDIUM"

    return "LOW"


def should_escalate(confidence):
    """
    Decide whether low confidence should trigger escalation.
    """

    return confidence < 0.50


def analyze_confidence(result):
    """
    Analyze the confidence returned by the intent classifier.

    Example input:
        {
            "intent": "BILLING_ISSUE",
            "confidence": 0.80
        }
    """

    confidence = result.get("confidence", 0.0)

    return {
        "intent": result.get(
            "intent",
            "GENERAL_INQUIRY"
        ),
        "confidence": confidence,
        "level": confidence_level(confidence),
        "should_escalate": should_escalate(confidence)
    }