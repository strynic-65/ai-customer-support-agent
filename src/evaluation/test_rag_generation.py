from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.prompts import format_reply_prompt


VECTOR_INDEX = "data/processed/comcast_faiss.index"
DOCUMENTS_PATH = "data/processed/comcast_documents.pkl"


def test_rag_generation():

    print("=" * 70)
    print("RAG + RESPONSE GENERATION CONTEXT TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Load Comcast vector database
    # --------------------------------------------------

    print("\nLoading Comcast vector database...")

    vector_store = VectorStore.load(
        VECTOR_INDEX,
        DOCUMENTS_PATH
    )

    print(f"Vector database size: {vector_store.size()}")

    # --------------------------------------------------
    # Create retriever
    # --------------------------------------------------

    retriever = Retriever(vector_store)

    # --------------------------------------------------
    # Test customer message
    # --------------------------------------------------

    customer_message = (
        "My internet has been completely down for the last "
        "two hours. Please help."
    )

    intent = "INTERNET_OUTAGE"

    print("\nCustomer message:")
    print(customer_message)

    print("\nDetected intent:")
    print(intent)

    # --------------------------------------------------
    # Retrieve similar historical cases
    # --------------------------------------------------

    retrieved_cases = retriever.retrieve(
        query=customer_message,
        top_k=3,
        min_score=0.30
    )

    print(f"\nRetrieved cases: {len(retrieved_cases)}")

    # --------------------------------------------------
    # Display retrieved cases
    # --------------------------------------------------

    for index, result in enumerate(retrieved_cases, start=1):

        print("\n" + "-" * 70)
        print(f"CASE {index}")
        print(f"Similarity Score: {result['score']:.4f}")

        document = result["document"]

        print(f"Conversation ID: {document.get('conversation_id')}")

        print("\nHistorical conversation:")
        print(document.get("text", ""))

    # --------------------------------------------------
    # Build response-generation prompt
    # --------------------------------------------------

    prompt = format_reply_prompt(
        customer_message=customer_message,
        intent=intent,
        historical_cases=retrieved_cases
    )

    print("\n" + "=" * 70)
    print("GENERATED RESPONSE PROMPT")
    print("=" * 70)

    print(prompt)

    print("\n" + "=" * 70)
    print("RAG CONTEXT TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_rag_generation()