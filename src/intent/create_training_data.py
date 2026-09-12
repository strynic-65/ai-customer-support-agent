import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CONVERSATIONS_PATH = "data/processed/conversations.csv"
REVIEW_PATH = "data/golden/intent_review_grouped.csv"
OUTPUT_PATH = "data/processed/intent_training_data.csv"

# Conservative threshold.
# We only keep messages that are sufficiently similar
# to reviewed examples.
SIMILARITY_THRESHOLD = 0.45

# Require at least two similar examples to agree
# on the same intent.
TOP_K = 5


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Keep words/numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():

    print("=" * 70)
    print("COMCAST INTENT TRAINING DATA GENERATOR")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load customer messages
    # ---------------------------------------------------------

    conversations = pd.read_csv(CONVERSATIONS_PATH)

    customer = conversations[
        conversations["inbound"] == True
    ].copy()

    customer["text_clean"] = customer["text"].astype(str).str.strip()

    print()
    print("Total customer messages:", len(customer))

    # ---------------------------------------------------------
    # 2. Load reviewed examples
    # ---------------------------------------------------------

    reviewed = pd.read_csv(REVIEW_PATH)

    reviewed["customer_message"] = (
        reviewed["customer_message"]
        .astype(str)
        .str.strip()
    )

    reviewed["reviewed_intent"] = (
        reviewed["reviewed_intent"]
        .astype(str)
        .str.strip()
    )

    print("Reviewed examples:", len(reviewed))

    # ---------------------------------------------------------
    # 3. Remove evaluation examples
    # ---------------------------------------------------------

    reviewed_messages = set(
        reviewed["customer_message"]
    )

    training_pool = customer[
        ~customer["text_clean"].isin(reviewed_messages)
    ].copy()

    print(
        "Excluded evaluation examples:",
        len(customer) - len(training_pool)
    )

    print(
        "Training pool:",
        len(training_pool)
    )

    # ---------------------------------------------------------
    # 4. Normalize text
    # ---------------------------------------------------------

    reviewed["normalized"] = reviewed[
        "customer_message"
    ].apply(normalize_text)

    training_pool["normalized"] = training_pool[
        "text_clean"
    ].apply(normalize_text)

    # Remove empty messages
    reviewed = reviewed[
        reviewed["normalized"].str.len() > 0
    ].copy()

    training_pool = training_pool[
        training_pool["normalized"].str.len() > 0
    ].copy()

    # ---------------------------------------------------------
    # 5. TF-IDF representation
    # ---------------------------------------------------------

    print()
    print("Building TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000,
    )

    all_text = pd.concat(
        [
            reviewed["normalized"],
            training_pool["normalized"],
        ]
    )

    vectorizer.fit(all_text)

    reviewed_vectors = vectorizer.transform(
        reviewed["normalized"]
    )

    training_vectors = vectorizer.transform(
        training_pool["normalized"]
    )

    print(
        "Vocabulary size:",
        len(vectorizer.vocabulary_)
    )

    # ---------------------------------------------------------
    # 6. Process in batches
    # ---------------------------------------------------------

    print()
    print("Finding similar reviewed examples...")

    results = []

    batch_size = 1000

    for start in range(
        0,
        len(training_pool),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(training_pool)
        )

        batch_vectors = training_vectors[
            start:end
        ]

        similarities = cosine_similarity(
            batch_vectors,
            reviewed_vectors
        )

        for i, row_similarities in enumerate(
            similarities
        ):

            # Get top K similar reviewed examples
            top_indices = row_similarities.argsort()[
                -TOP_K:
            ][::-1]

            top_scores = row_similarities[
                top_indices
            ]

            # Highest similarity
            best_score = float(
                top_scores[0]
            )

            if best_score < SIMILARITY_THRESHOLD:
                continue

            # Corresponding intents
            top_intents = reviewed.iloc[
                top_indices
            ]["reviewed_intent"].tolist()

            # Count intent agreement
            intent_counts = {}

            for intent in top_intents:
                intent_counts[intent] = (
                    intent_counts.get(intent, 0) + 1
                )

            predicted_intent = max(
                intent_counts,
                key=intent_counts.get
            )

            agreement = intent_counts[
                predicted_intent
            ]

            # Require at least two neighbors
            # to agree.
            if agreement < 2:
                continue

            original_index = (
                training_pool.index[start + i]
            )

            original_row = training_pool.loc[
                original_index
            ]

            results.append(
                {
                    "tweet_id": original_row[
                        "tweet_id"
                    ],
                    "conversation_id": original_row[
                        "conversation_id"
                    ],
                    "customer_message": original_row[
                        "text_clean"
                    ],
                    "pseudo_intent": predicted_intent,
                    "similarity": best_score,
                    "agreement": agreement,
                }
            )

        if end % 5000 == 0 or end == len(training_pool):
            print(
                f"Processed {end}/{len(training_pool)}"
            )

    # ---------------------------------------------------------
    # 7. Save results
    # ---------------------------------------------------------

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print()
        print("ERROR: No high-confidence examples found.")
        return

    result_df = result_df.sort_values(
        by=[
            "similarity",
            "agreement"
        ],
        ascending=False
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING DATA GENERATED")
    print("=" * 70)

    print(
        "Pseudo-labeled examples:",
        len(result_df)
    )

    print()
    print("Intent distribution:")

    print(
        result_df[
            "pseudo_intent"
        ].value_counts().sort_index()
    )

    print()
    print("Similarity statistics:")

    print(
        result_df[
            "similarity"
        ].describe()
    )

    print()
    print("Saved to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()