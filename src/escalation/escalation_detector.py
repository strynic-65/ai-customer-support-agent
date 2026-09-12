# src/escalation/escalation_detector.py

from src.escalation.risk_rules import detect_risks


def detect_escalation(customer_message, intent, confidence):
    """
    Decide whether a customer request should be automatically handled
    or escalated to a human agent.

    High-risk signals always trigger escalation.
    Low intent confidence alone does not automatically trigger escalation
    for routine support requests.
    """

    risks = detect_risks(customer_message)

    reasons = []

    # ---------------------------------------------------------
    # HIGH-RISK CONDITIONS
    # ---------------------------------------------------------

    if risks["security_risk"]:
        reasons.append(
            "Possible security or privacy concern"
        )

    if risks["frustration"]:
        reasons.append(
            "Customer appears highly frustrated"
        )

    if risks["human_requested"]:
        reasons.append(
            "Customer requested a human agent"
        )

    # ---------------------------------------------------------
    # LOW CONFIDENCE
    # ---------------------------------------------------------
    #
    # Only escalate low confidence when the system also lacks
    # a clearly routine support intent.
    #

    routine_intents = {
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
    }

    if (
        confidence < 0.50
        and intent not in routine_intents
    ):
        reasons.append(
            "Low intent confidence"
        )

    # ---------------------------------------------------------
    # FINAL DECISION
    # ---------------------------------------------------------

    if reasons:
        decision = "ESCALATE"
    else:
        decision = "AUTO_HANDLE"

    return {
        "decision": decision,
        "reasons": reasons,
        "risks": risks,
        "intent": intent,
        "confidence": confidence,
    }