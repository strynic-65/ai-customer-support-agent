from src.pipeline import run_pipeline


def main():
    print("=" * 70)
    print("AI CUSTOMER SUPPORT AGENT")
    print("=" * 70)

    customer_message = input("\nCustomer: ").strip()

    if not customer_message:
        print("\nPlease enter a customer message.")
        return

    print("\nProcessing customer request...")

    try:
        result = run_pipeline(
            customer_message,
            top_k=3
        )

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)

        # ============================================================
        # INTENT ANALYSIS
        # ============================================================

        print("\n" + "-" * 70)
        print("INTENT ANALYSIS")
        print("-" * 70)

        final_intent = result.get(
            "intent",
            "UNKNOWN"
        )

        confidence_data = result.get(
            "confidence",
            0.0
        )

        if isinstance(confidence_data, dict):

            final_confidence = float(
                confidence_data.get(
                    "confidence",
                    0.0
                )
            )

            confidence_level = (
                confidence_data.get(
                    "level",
                    "UNKNOWN"
                )
            )

        else:

            final_confidence = float(
                confidence_data
            )

            confidence_level = "UNKNOWN"

        ml_intent = result.get(
            "ml_intent",
            "UNKNOWN"
        )

        ml_confidence = float(
            result.get(
                "ml_confidence",
                0.0
            )
        )

        # Domain information may be returned directly
        # by the pipeline or inside the intent result.

        domain_intent = result.get(
            "domain_intent",
            None
        )

        domain_score = result.get(
            "domain_score",
            0.0
        )

        source = result.get(
            "source",
            "UNKNOWN"
        )

        # ------------------------------------------------------------
        # Final hybrid result
        # ------------------------------------------------------------

        print(
            f"\nFinal Intent       : {final_intent}"
        )

        print(
            f"Final Confidence   : {final_confidence:.2f}"
        )

        print(
            f"Confidence Level   : {confidence_level}"
        )

        # ------------------------------------------------------------
        # Raw ML result
        # ------------------------------------------------------------

        print(
            f"\nML Intent          : {ml_intent}"
        )

        print(
            f"ML Confidence      : {ml_confidence:.2f}"
        )

        # ------------------------------------------------------------
        # Domain rule result
        # ------------------------------------------------------------

        if domain_intent:

            print(
                f"\nDomain Intent      : {domain_intent}"
            )

            print(
                f"Domain Score       : {float(domain_score):.2f}"
            )

        else:

            print(
                "\nDomain Intent      : None"
            )

            print(
                "Domain Score       : 0.00"
            )

        print(
            f"Decision Source    : {source}"
        )

        # ============================================================
        # RETRIEVAL
        # ============================================================

        print("\n" + "-" * 70)
        print("RETRIEVED SIMILAR CASES")
        print("-" * 70)

        retrieved_cases = result.get(
            "retrieved_cases",
            []
        )

        if not retrieved_cases:

            print(
                "No similar historical cases found."
            )

        else:

            for i, case in enumerate(
                retrieved_cases,
                start=1
            ):

                print(
                    f"\nCase {i}"
                )

                score = case.get(
                    "score",
                    0.0
                )

                print(
                    f"Similarity Score : {float(score):.3f}"
                )

                document = case.get(
                    "document",
                    {}
                )

                conversation_id = document.get(
                    "conversation_id",
                    "N/A"
                )

                conversation_text = document.get(
                    "text",
                    ""
                )

                print(
                    f"Conversation ID  : {conversation_id}"
                )

                print(
                    "Conversation:"
                )

                if conversation_text:

                    print(
                        conversation_text
                    )

                else:

                    print(
                        "Conversation content not available."
                    )

        # ============================================================
        # GENERATED REPLY
        # ============================================================

        print("\n" + "-" * 70)
        print("GENERATED SUPPORT REPLY")
        print("-" * 70)

        reply = result.get(
            "reply",
            "No response generated."
        )

        print(reply)

        # ============================================================
        # ESCALATION ANALYSIS
        # ============================================================

        print("\n" + "-" * 70)
        print("ESCALATION ANALYSIS")
        print("-" * 70)

        escalation = result.get(
            "escalation",
            {}
        )

        escalation_decision = escalation.get(
            "decision",
            "UNKNOWN"
        )

        print(
            f"Decision : {escalation_decision}"
        )

        reasons = escalation.get(
            "reasons",
            []
        )

        if reasons:

            print(
                "Reasons:"
            )

            for reason in reasons:

                print(
                    f"- {reason}"
                )

        else:

            print(
                "Reasons  : No escalation reason detected."
            )

        # ============================================================
        # FINAL DECISION
        # ============================================================

        print("\n" + "-" * 70)
        print("FINAL DECISION")
        print("-" * 70)

        final_decision = result.get(
            "decision",
            escalation_decision
        )

        if isinstance(
            final_decision,
            dict
        ):

            decision_value = final_decision.get(
                "decision",
                "UNKNOWN"
            )

            print(
                decision_value
            )

            final_reason = final_decision.get(
                "reason",
                []
            )

            if final_reason:

                print(
                    "\nFinal Reason:"
                )

                if isinstance(
                    final_reason,
                    list
                ):

                    for reason in final_reason:

                        print(
                            f"- {reason}"
                        )

                else:

                    print(
                        f"- {final_reason}"
                    )

        else:

            print(
                final_decision
            )

        # ============================================================
        # COMPLETION
        # ============================================================

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("PIPELINE ERROR")
        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        print(
            "\nPlease send me this complete error output so we can fix it."
        )

        print("=" * 70)


if __name__ == "__main__":
    main()