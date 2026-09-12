from sentence_transformers import SentenceTransformer


# Lightweight and effective sentence embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


# Load the model once
_model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text):
    """
    Convert a text string into a numerical embedding.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    embedding = _model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding


def generate_embeddings(texts):
    """
    Generate embeddings for multiple texts.
    """

    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


def get_embedding_dimension():
    """
    Return the size of the generated embedding vector.
    """

    return _model.get_sentence_embedding_dimension()