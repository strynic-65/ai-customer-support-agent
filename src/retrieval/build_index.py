import os

import pandas as pd

from src.retrieval.embeddings import generate_embeddings
from src.retrieval.vector_store import VectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/processed/conversations.csv"

INDEX_FILE = "data/processed/comcast_faiss.index"

DOCUMENTS_FILE = "data/processed/comcast_documents.pkl"

BATCH_SIZE = 64

MAX_CONVERSATION_CHARS = 3000


# --------------------------------------------------
# Create searchable conversation documents
# --------------------------------------------------

def build_documents(df):
    """
    Convert individual conversation messages into
    one searchable document per conversation.
    """

    documents = []

    grouped = df.groupby(
        "conversation_id",
        sort=False
    )

    for conversation_id, group in grouped:

        group = group.sort_values(
            "created_at"
        )

        messages = []

        for _, row in group.iterrows():

            author = str(
                row.get("author_id", "")
            )

            text = str(
                row.get("text", "")
            ).strip()

            if not text or text == "nan":
                continue

            if row.get("inbound") is True:
                speaker = "Customer"
            else:
                speaker = "Comcast"

            messages.append(
                f"{speaker}: {text}"
            )

        if not messages:
            continue

        conversation_text = "\n".join(
            messages
        )

        # Prevent extremely long conversations
        conversation_text = conversation_text[
            :MAX_CONVERSATION_CHARS
        ]

        documents.append({
            "conversation_id": conversation_id,
            "text": conversation_text
        })

    return documents


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading conversation data...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Messages loaded: {len(df):,}"
    )

    print("Building conversation documents...")

    documents = build_documents(df)

    print(
        f"Conversation documents: {len(documents):,}"
    )

    if not documents:
        raise ValueError(
            "No conversation documents were created."
        )

    # Extract text for embedding
    texts = [
        document["text"]
        for document in documents
    ]

    print()
    print("Generating embeddings...")
    print(
        f"Total documents: {len(texts):,}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # Create vector store
    store = VectorStore(
        dimension=384
    )

    # Generate embeddings in batches
    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(texts)
        )

        batch_texts = texts[
            start:end
        ]

        print(
            f"Embedding {start:,} - {end:,} "
            f"of {len(texts):,}"
        )

        batch_embeddings = generate_embeddings(
            batch_texts
        )

        store.add(
            batch_embeddings,
            documents[start:end]
        )

    print()
    print("Embedding generation completed.")

    # Save index and documents
    print("Saving FAISS index...")

    store.save(
        INDEX_FILE,
        DOCUMENTS_FILE
    )

    print()
    print("========================================")
    print("RAG INDEX BUILD COMPLETED")
    print("========================================")
    print(
        f"Vectors stored: {store.size():,}"
    )
    print(
        f"FAISS index: {INDEX_FILE}"
    )
    print(
        f"Documents: {DOCUMENTS_FILE}"
    )
    print("========================================")


if __name__ == "__main__":
    main()