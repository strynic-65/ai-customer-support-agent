"""
Final escalation decision layer.
"""


def make_decision(escalation_result):
    """
    Convert escalation detection result into
    a clean final decision.
    """

    if not isinstance(escalation_result, dict):
        raise ValueError("Escalation result must be a dictionary.")

    decision = escalation_result.get("decision")

    if decision == "ESCALATE":
        return {
            "decision": "ESCALATE",
            "reason": escalation_result.get(
                "reasons",
                ["Escalation required."]
            )
        }

    return {
        "decision": "AUTO_HANDLE",
        "reason": []
    }

