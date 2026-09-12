import pandas as pd

DATA_PATH = "data/raw/twcs.csv"


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET INFORMATION =====")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    load_dataset()