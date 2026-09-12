"""
Create a balanced Golden Evaluation Set using actual
customer messages from the original TWCS dataset.

Comcast is used as the selected support brand.
"""

import os
import pandas as pd

from src.intent.classifier import classify_intent
from src.intent.taxonomy import INTENTS


INPUT_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/golden/golden_evaluation_set.csv"

BRAND = "comcastcares"

TOTAL_SIZE = 200


def create_golden_set():

    # ---------------------------------------------------------
    # 1. Load original TWCS dataset
    # ---------------------------------------------------------
    df = pd.read_csv(
        INPUT_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id"
        ]
    )

    print(
        "Total TWCS messages:",
        len(df)
    )

    # ---------------------------------------------------------
    # 2. Find Comcast conversations
    # ---------------------------------------------------------
    #
    # A Comcast support tweet has author_id = Comcast brand.
    # We collect Comcast tweet IDs first.
    #
    comcast_tweets = df[
        df["author_id"].astype(str).str.lower() == BRAND
    ].copy()

    print(
        "Comcast support messages:",
        len(comcast_tweets)
    )

    if comcast_tweets.empty:

        raise ValueError(
            "No Comcast messages found in TWCS dataset."
        )

    # ---------------------------------------------------------
    # 3. Collect customer tweets that Comcast responded to
    # ---------------------------------------------------------
    #
    # Comcast's in_response_to_tweet_id points to the
    # customer's message.
    #
    customer_parent_ids = set(
        pd.to_numeric(
            comcast_tweets[
                "in_response_to_tweet_id"
            ],
            errors="coerce"
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    customer_messages = df[
        df["tweet_id"].isin(
            customer_parent_ids
        )
        & (df["inbound"] == True)
    ].copy()

    print(
        "Customer messages in Comcast conversations:",
        len(customer_messages)
    )

    # ---------------------------------------------------------
    # 4. Remove empty messages
    # ---------------------------------------------------------
    customer_messages = customer_messages[
        customer_messages["text"].notna()
        & (
            customer_messages["text"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    # ---------------------------------------------------------
    # 5. Remove duplicate customer messages
    # ---------------------------------------------------------
    customer_messages = (
        customer_messages
        .drop_duplicates(
            subset=["text"]
        )
    )

    print(
        "Unique customer messages:",
        len(customer_messages)
    )

    # ---------------------------------------------------------
    # 6. Generate provisional intent labels
    # ---------------------------------------------------------
    print(
        "\nGenerating provisional intent labels..."
    )

    customer_messages["expected_intent"] = (
        customer_messages["text"].apply(
            lambda text:
                classify_intent(text)["intent"]
        )
    )

    # ---------------------------------------------------------
    # 7. Remove GENERAL_INQUIRY
    # ---------------------------------------------------------
    customer_messages = customer_messages[
        customer_messages["expected_intent"]
        != "GENERAL_INQUIRY"
    ].copy()

    # ---------------------------------------------------------
    # 8. Supported intents
    # ---------------------------------------------------------
    intents = [
        intent
        for intent in INTENTS
        if intent != "GENERAL_INQUIRY"
    ]

    examples_per_intent = (
        TOTAL_SIZE // len(intents)
    )

    print(
        "\nTarget examples per intent:",
        examples_per_intent
    )

    golden_parts = []

    # ---------------------------------------------------------
    # 9. Balanced sampling
    # ---------------------------------------------------------
    for intent in intents:

        intent_df = customer_messages[
            customer_messages["expected_intent"]
            == intent
        ]

        available = len(intent_df)

        if available == 0:

            print(
                f"WARNING: No examples found for {intent}"
            )

            continue

        sample_size = min(
            examples_per_intent,
            available
        )

        sample = intent_df.sample(
            n=sample_size,
            random_state=42
        )

        golden_parts.append(
            sample
        )

        print(
            f"{intent:25} "
            f"Available: {available:6} "
            f"Selected: {sample_size:3}"
        )

    # ---------------------------------------------------------
    # 10. Combine samples
    # ---------------------------------------------------------
    if not golden_parts:

        raise ValueError(
            "No suitable customer examples found."
        )

    golden = pd.concat(
        golden_parts,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # 11. Shuffle
    # ---------------------------------------------------------
    golden = golden.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # 12. Add provisional escalation labels
    # ---------------------------------------------------------
    golden["expected_escalation"] = (
        golden["expected_intent"].apply(
            lambda intent:
                "ESCALATE"
                if intent in [
                    "ACCOUNT_ISSUE",
                    "PAYMENT_ISSUE"
                ]
                else "AUTO_HANDLE"
        )
    )

    # ---------------------------------------------------------
    # 13. Keep evaluation columns
    # ---------------------------------------------------------
    golden = golden[
        [
            "text",
            "expected_intent",
            "expected_escalation"
        ]
    ]

    golden.rename(
        columns={
            "text": "customer_message"
        },
        inplace=True
    )

    # ---------------------------------------------------------
    # 14. Save Golden Set
    # ---------------------------------------------------------
    os.makedirs(
        "data/golden",
        exist_ok=True
    )

    golden.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 15. Print final summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(
        "CUSTOMER-ONLY GOLDEN SET CREATED"
    )
    print("=" * 60)

    print(
        "Total examples:",
        len(golden)
    )

    print(
        "\nIntent distribution:"
    )

    print(
        golden["expected_intent"]
        .value_counts()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    create_golden_set()