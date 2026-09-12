"""
End-to-end AI Customer Support Agent Pipeline.

Flow:

Customer Message
        ↓
Hybrid Intent Classification
        ↓
Confidence Analysis
        ↓
Historical Case Retrieval (RAG)
        ↓
Response Generation
        ↓
Escalation Detection
        ↓
Final Decision
"""

from src.intent.ml_classifier import predict_intent, train_final_model
from src.intent.confidence import analyze_confidence

from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever

from src.generation.response_generator import generate_support_reply

from src.escalation.escalation_detector import detect_escalation
from src.escalation.decision import make_decision


# ============================================================
# LOAD INTENT CLASSIFIER
# ============================================================

print("Loading intent classifier...")

_INTENT_VECTORIZER, _INTENT_CLASSIFIER = train_final_model()

print("Intent classifier loaded.")


# ============================================================
# LOAD COMCAST VECTOR DATABASE
# ============================================================

print("Loading Comcast vector database...")

_VECTOR_STORE = VectorStore.load(
    index_path="data/processed/comcast_faiss.index",
    documents_path="data/processed/comcast_documents.pkl"
)

print("Comcast vector database loaded.")

_RETRIEVER = Retriever(
    vector_store=_VECTOR_STORE
)


# ============================================================
# MAIN SUPPORT AGENT
# ============================================================

def run_support_agent(customer_message, top_k=3):
    """
    Run the complete AI customer support pipeline.
    """

    if not isinstance(customer_message, str):
        raise ValueError(
            "customer_message must be a string."
        )

    if not customer_message.strip():
        raise ValueError(
            "customer_message cannot be empty."
        )

    # ========================================================
    # STEP 1: INTENT CLASSIFICATION
    # ========================================================

    intent_result = predict_intent(
        customer_message,
        _INTENT_VECTORIZER,
        _INTENT_CLASSIFIER
    )

    # ========================================================
    # STEP 2: CONFIDENCE ANALYSIS
    # ========================================================

    intent_analysis = analyze_confidence(
        intent_result
    )

    intent = intent_analysis["intent"]

    confidence = intent_analysis["confidence"]

    # ========================================================
    # STEP 3: RAG RETRIEVAL
    # ========================================================

    retrieved_cases = _RETRIEVER.retrieve(
        query=customer_message,
        top_k=top_k,
        min_score=0.30
    )

    # ========================================================
    # STEP 4: RESPONSE GENERATION
    # ========================================================
    #
    # IMPORTANT:
    # generate_support_reply() expects positional arguments
    # in the current project.
    #
    # Therefore we intentionally use:
    #
    # generate_support_reply(
    #     customer_message,
    #     intent,
    #     retrieved_cases
    # )
    #
    # ========================================================

    generated_reply = generate_support_reply(
        customer_message,
        intent,
        retrieved_cases
    )

    # ========================================================
    # STEP 5: ESCALATION DETECTION
    # ========================================================

    escalation_result = detect_escalation(
        customer_message=customer_message,
        intent=intent,
        confidence=confidence
    )

    # ========================================================
    # STEP 6: FINAL DECISION
    # ========================================================

    final_decision = make_decision(
        escalation_result
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "customer_message": customer_message,

        "intent": intent,

        "confidence": confidence,

        "confidence_level": intent_analysis["level"],

        "should_escalate": intent_analysis[
            "should_escalate"
        ],

        # ----------------------------------------------------
        # ML INFORMATION
        # ----------------------------------------------------

        "ml_intent": intent_result.get(
            "ml_intent",
            intent
        ),

        "ml_confidence": intent_result.get(
            "ml_confidence",
            intent_result.get(
                "confidence",
                confidence
            )
        ),

        # ----------------------------------------------------
        # DOMAIN INFORMATION
        # ----------------------------------------------------

        "domain_intent": intent_result.get(
            "domain_intent",
            intent
        ),

        "domain_score": intent_result.get(
            "domain_score",
            0.0
        ),

        "classification_source": intent_result.get(
            "classification_source",
            "HYBRID"
        ),

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        "retrieved_cases": retrieved_cases,

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        "generated_reply": generated_reply,

        # Compatibility with evaluation modules
        "reply": generated_reply,

        # ----------------------------------------------------
        # ESCALATION
        # ----------------------------------------------------

        "escalation": escalation_result,

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        "final_decision": final_decision
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def run_pipeline(customer_message, top_k=3):
    """
    Backward-compatible wrapper used by evaluation modules.
    """

    return run_support_agent(
        customer_message=customer_message,
        top_k=top_k
    )


# ============================================================
# RAG DOCUMENT TEXT HELPER
# ============================================================

def get_case_text(case):
    """
    Extract historical conversation text from a
    VectorStore retrieval result.

    Current VectorStore structure:

        {
            "document": {
                "conversation_id": "...",
                "text": "..."
            },
            "score": 0.81
        }
    """

    if not isinstance(case, dict):
        return ""

    # --------------------------------------------------------
    # Current VectorStore format
    # --------------------------------------------------------

    document = case.get(
        "document"
    )

    if isinstance(document, dict):

        text = document.get(
            "text",
            ""
        )

        if isinstance(text, str):
            return text

    # --------------------------------------------------------
    # Fallback formats
    # --------------------------------------------------------

    text = case.get(
        "text",
        ""
    )

    if isinstance(text, str) and text:
        return text

    customer_text = case.get(
        "customer_message",
        ""
    )

    brand_response = case.get(
        "brand_response",
        ""
    )

    parts = []

    if customer_text:
        parts.append(
            f"Customer: {customer_text}"
        )

    if brand_response:
        parts.append(
            f"Comcast: {brand_response}"
        )

    return "\n".join(parts)


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(result):
    """
    Display the complete support-agent result.
    """

    print("\n")
    print("=" * 70)
    print("CUSTOMER SUPPORT AGENT RESULT")
    print("=" * 70)

    # ========================================================
    # CUSTOMER MESSAGE
    # ========================================================

    print("\nCustomer Message:")
    print(
        result["customer_message"]
    )

    # ========================================================
    # FINAL INTENT
    # ========================================================

    print("\nIntent:")
    print(
        result["intent"]
    )

    print("\nConfidence:")
    print(
        f"{result['confidence']:.2f}"
    )

    print("\nConfidence Level:")
    print(
        result["confidence_level"]
    )

    # ========================================================
    # ML DETAILS
    # ========================================================

    print("\nML Intent:")
    print(
        result["ml_intent"]
    )

    print("\nML Confidence:")

    ml_confidence = result[
        "ml_confidence"
    ]

    if isinstance(
        ml_confidence,
        (int, float)
    ):
        print(
            f"{ml_confidence:.4f}"
        )
    else:
        print(
            ml_confidence
        )

    # ========================================================
    # DOMAIN DETAILS
    # ========================================================

    print("\nDomain Intent:")
    print(
        result["domain_intent"]
    )

    print("\nDomain Score:")
    print(
        result["domain_score"]
    )

    print("\nClassification Source:")
    print(
        result["classification_source"]
    )

    # ========================================================
    # RAG RESULTS
    # ========================================================

    print("\nRetrieved Cases:")
    print(
        len(result["retrieved_cases"])
    )

    if not result["retrieved_cases"]:

        print(
            "No sufficiently similar historical cases found."
        )

    else:

        for i, case in enumerate(
            result["retrieved_cases"],
            start=1
        ):

            print("\n" + "-" * 70)

            score = case.get(
                "score",
                0.0
            )

            print(
                f"Case {i} "
                f"(similarity score={score:.4f})"
            )

            print("-" * 70)

            # ------------------------------------------------
            # Extract actual document
            # ------------------------------------------------

            case_text = get_case_text(
                case
            )

            if case_text:

                print(
                    case_text
                )

            else:

                print(
                    "Historical conversation text unavailable."
                )

    # ========================================================
    # GENERATED RESPONSE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GENERATED RESPONSE")
    print("=" * 70)

    print(
        result["generated_reply"]
    )

    # ========================================================
    # ESCALATION RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ESCALATION RESULT")
    print("=" * 70)

    print(
        result["escalation"]
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL DECISION")
    print("=" * 70)

    print(
        result["final_decision"]
    )

    print("\n")
    print("=" * 70)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    test_message = (
        "Why is my Comcast bill so high this month?"
    )

    result = run_support_agent(
        customer_message=test_message,
        top_k=3
    )

    display_result(
        result
    )