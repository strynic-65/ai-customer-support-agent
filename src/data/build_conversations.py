import pandas as pd

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/conversations.csv"

BRAND = "comcastcares"

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]


def build_conversations():

    print("Step 1: Finding Comcast tweets...")

    brand_ids = set()
    parent_ids = set()

    # First pass: find Comcast tweets and their parent tweets
    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=COLUMNS,
        chunksize=100_000
    ):
        brand_rows = chunk[
            chunk["author_id"] == BRAND
        ]

        brand_ids.update(
            brand_rows["tweet_id"].tolist()
        )

        parent_ids.update(
            brand_rows["in_response_to_tweet_id"]
            .dropna()
            .tolist()
        )

    print(f"Comcast tweets: {len(brand_ids):,}")
    print(f"Customer parent tweets: {len(parent_ids):,}")

    print("\nStep 2: Extracting conversation messages...")

    conversations = []

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=COLUMNS,
        chunksize=100_000
    ):

        # Comcast messages
        brand_rows = chunk[
            chunk["tweet_id"].isin(brand_ids)
        ]

        # Customer messages that Comcast replied to
        customer_messages = chunk[
            chunk["tweet_id"].isin(parent_ids)
        ]

        # Customer messages replying to Comcast
        customer_replies = chunk[
            chunk["in_response_to_tweet_id"].isin(brand_ids)
        ]

        selected = pd.concat(
            [
                brand_rows,
                customer_messages,
                customer_replies
            ]
        ).drop_duplicates(
            subset="tweet_id"
        )

        if not selected.empty:
            conversations.append(selected)

    conversations_df = pd.concat(
        conversations,
        ignore_index=True
    )

    # Convert timestamp
    conversations_df["created_at"] = pd.to_datetime(
        conversations_df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce"
    )

    # Sort messages chronologically
    conversations_df = conversations_df.sort_values(
        "created_at"
    ).reset_index(drop=True)

    # Create a conversation ID.
    # For each tweet, walk backwards through its parent tweets.
    tweet_to_parent = dict(
        zip(
            conversations_df["tweet_id"],
            conversations_df["in_response_to_tweet_id"]
        )
    )

    def find_root(tweet_id):
        current = tweet_id
        visited = set()

        while (
            current in tweet_to_parent
            and pd.notna(tweet_to_parent[current])
            and current not in visited
        ):
            visited.add(current)
            parent = tweet_to_parent[current]

            if parent not in tweet_to_parent:
                return parent

            current = parent

        return current

    print("\nStep 3: Creating conversation IDs...")

    conversations_df["conversation_id"] = (
        conversations_df["tweet_id"]
        .apply(find_root)
    )

    # Save
    conversations_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nConversation reconstruction completed!")
    print(f"Total messages: {len(conversations_df):,}")
    print(
        f"Unique conversations: "
        f"{conversations_df['conversation_id'].nunique():,}"
    )
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_conversations()