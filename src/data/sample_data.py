import pandas as pd

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/brand_conversations.csv"

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


def extract_brand_data():
    print("Starting Comcast data extraction...")

    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=COLUMNS,
        chunksize=100_000
    ):
        total_rows += len(chunk)

        # Comcast tweets
        brand_tweets = chunk[
            chunk["author_id"] == BRAND
        ]

        if not brand_tweets.empty:
            chunks.append(brand_tweets)

        print(f"Processed {total_rows:,} rows...")

    if not chunks:
        print("No Comcast tweets found.")
        return

    # Combine all Comcast tweets
    brand_data = pd.concat(
        chunks,
        ignore_index=True
    )

    # Save result
    brand_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Extraction completed!")
    print(f"Comcast tweets: {len(brand_data):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_brand_data()