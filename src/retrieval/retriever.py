from src.retrieval.embeddings import generate_embedding


class Retriever:
    """
    Retrieves semantically similar historical support cases.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, min_score=0.30):
        """
        Retrieve similar cases for a customer query.

        Args:
            query: Customer message.
            top_k: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            List of relevant historical cases.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        # Convert customer query into an embedding
        query_embedding = generate_embedding(query)

        # Search the FAISS vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # Remove weak matches
        filtered_results = [
            result
            for result in results
            if result["score"] >= min_score
        ]

        return filtered_results


def retrieve_similar_cases(
    query,
    vector_store,
    top_k=5,
    min_score=0.30
):
    """
    Convenience function for retrieving similar cases.
    """

    retriever = Retriever(vector_store)

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score
    )