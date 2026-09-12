import os
import pickle

import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector store for semantic similarity search.
    """

    def __init__(self, dimension):
        self.dimension = dimension

        # Inner Product works as cosine similarity
        # because embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

        # Store conversation records alongside vectors
        self.documents = []

    def add(self, embeddings, documents):
        """
        Add embeddings and their corresponding documents.
        """

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        if len(embeddings) == 0:
            return

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(vectors)

        self.documents.extend(
            documents
        )

    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar documents.

        Returns:
            List of dictionaries containing:
            - document
            - score
        """

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            min(
                top_k,
                self.index.ntotal
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append(
                {
                    "document": self.documents[index],
                    "score": float(score)
                }
            )

        return results

    def save(
        self,
        index_path,
        documents_path
    ):
        """
        Save FAISS index and documents to disk.
        """

        directory = os.path.dirname(
            index_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            documents_path,
            "wb"
        ) as file:

            pickle.dump(
                self.documents,
                file
            )

    @classmethod
    def load(
        cls,
        index_path,
        documents_path
    ):
        """
        Load a previously saved vector store.
        """

        index = faiss.read_index(
            index_path
        )

        with open(
            documents_path,
            "rb"
        ) as file:

            documents = pickle.load(
                file
            )

        store = cls(
            index.d
        )

        store.index = index

        store.documents = documents

        return store

    def size(self):
        """
        Return number of stored vectors.
        """

        return self.index.ntotal