from src.intent.ml_classifier import (
    predict_intent,
    train_final_model,
)

from src.intent.confidence import analyze_confidence

from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever

from src.generation.response_generator import generate_support_reply

from src.escalation.escalation_detector import detect_escalation
from src.escalation.decision import make_decision


# ============================================================
# LOAD FINAL INTENT CLASSIFIER
# ============================================================

print("Loading final intent classifier...")

_INTENT_VECTORIZER, _INTENT_CLASSIFIER = train_final_model()

print("Final intent classifier ready.")


# ============================================================
# LOAD COMCAST VECTOR DATABASE
# ============================================================

print("Loading Comcast vector database...")

_VECTOR_STORE = VectorStore.load(
    "data/processed/comcast_faiss.index",
    "data/processed/comcast_documents.pkl",
)

print("Vector database ready.")


# ============================================================
# CREATE RETRIEVER
# ============================================================

_RETRIEVER = Retriever(
    vector_store=_VECTOR_STORE
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(customer_message, top_k=3):

    # ========================================================
    # 1. INTENT CLASSIFICATION
    # ========================================================

    intent_result = predict_intent(
        customer_message,
        _INTENT_VECTORIZER,
        _INTENT_CLASSIFIER,
    )

    # --------------------------------------------------------
    # Final hybrid classification
    # --------------------------------------------------------

    final_intent = intent_result.get(
        "intent",
        "GENERAL_INQUIRY"
    )

    final_confidence = float(
        intent_result.get(
            "confidence",
            0.0
        )
    )

    # --------------------------------------------------------
    # Raw ML classification
    # --------------------------------------------------------

    ml_intent = intent_result.get(
        "ml_intent",
        final_intent
    )

    ml_confidence = float(
        intent_result.get(
            "ml_confidence",
            0.0
        )
    )

    # --------------------------------------------------------
    # Domain-rule classification
    # --------------------------------------------------------

    domain_intent = intent_result.get(
        "domain_intent",
        None
    )

    domain_score = float(
        intent_result.get(
            "domain_score",
            0.0
        )
    )

    source = intent_result.get(
        "source",
        "ML"
    )

    # ========================================================
    # 2. CONFIDENCE ANALYSIS
    # ========================================================

    confidence_analysis = analyze_confidence(
        {
            "intent": final_intent,
            "confidence": final_confidence,
        }
    )

    # ========================================================
    # 3. RETRIEVE SIMILAR HISTORICAL CASES
    # ========================================================

    retrieved_cases = _RETRIEVER.retrieve(
        query=customer_message,
        top_k=top_k,
        min_score=0.30,
    )

    # ========================================================
    # 4. GENERATE SUPPORT REPLY
    # ========================================================

    reply = generate_support_reply(
        customer_message=customer_message,
        intent=final_intent,
        retrieved_cases=retrieved_cases,
    )

    # ========================================================
    # 5. ESCALATION ANALYSIS
    # ========================================================

    escalation = detect_escalation(
        customer_message=customer_message,
        intent=final_intent,
        confidence=final_confidence,
    )

    # ========================================================
    # 6. FINAL DECISION
    # ========================================================
    #
    # IMPORTANT:
    # make_decision() accepts the escalation result directly.
    #

    decision = make_decision(
        escalation
    )

    # ========================================================
    # 7. RETURN COMPLETE PIPELINE RESULT
    # ========================================================

    return {

        # ----------------------------------------------------
        # Customer input
        # ----------------------------------------------------

        "customer_message": customer_message,

        # ----------------------------------------------------
        # Final hybrid intent
        # ----------------------------------------------------

        "intent": final_intent,

        "confidence": confidence_analysis,

        # ----------------------------------------------------
        # Raw ML result
        # ----------------------------------------------------

        "ml_intent": ml_intent,

        "ml_confidence": ml_confidence,

        # ----------------------------------------------------
        # Domain rule result
        # ----------------------------------------------------

        "domain_intent": domain_intent,

        "domain_score": domain_score,

        "source": source,

        # ----------------------------------------------------
        # Retrieved historical cases
        # ----------------------------------------------------

        "retrieved_cases": retrieved_cases,

        # ----------------------------------------------------
        # Generated response
        # ----------------------------------------------------

        "reply": reply,

        # ----------------------------------------------------
        # Escalation analysis
        # ----------------------------------------------------

        "escalation": escalation,

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        "decision": decision,
    }


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_message = (
        "My internet has been completely down "
        "for the last two hours. Please help."
    )

    print("\n" + "=" * 70)
    print("PIPELINE TEST")
    print("=" * 70)

    result = run_pipeline(
        test_message,
        top_k=3
    )

    print("\nCustomer:")
    print(result["customer_message"])

    print("\nFinal Intent:")
    print(result["intent"])

    print(
        "\nFinal Confidence:",
        result["confidence"]
    )

    print(
        "\nML Intent:",
        result["ml_intent"]
    )

    print(
        "ML Confidence:",
        result["ml_confidence"]
    )

    print(
        "\nDomain Intent:",
        result["domain_intent"]
    )

    print(
        "Domain Score:",
        result["domain_score"]
    )

    print(
        "Decision Source:",
        result["source"]
    )

    print(
        "\nRetrieved Cases:",
        len(result["retrieved_cases"])
    )

    print("\nReply:")
    print(result["reply"])

    print("\nEscalation:")
    print(result["escalation"])

    print("\nFinal Decision:")
    print(result["decision"])

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETED")
    print("=" * 70)