# Name RollNo
import argparse
import os
from datetime import datetime

import pandas as pd


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def find_raw_csv(user_path: str | None) -> str:
    if user_path and os.path.exists(user_path):
        return user_path

    env_path = os.environ.get("DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    raw_dir = os.path.join("data", "raw")
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.lower().endswith(".csv"):
                return os.path.join(raw_dir, f)

    raise FileNotFoundError(
        "Raw dataset CSV not found. Provide --input or set DATA_PATH or place a CSV in data/raw/."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll_no", type=str, default="123")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=os.path.join("data", "processed", "processed.csv"))
    args = parser.parse_args()

    print(args.roll_no)
    print(_ts())

    in_path = find_raw_csv(args.input)
    df = pd.read_csv(in_path)

    df = df.drop_duplicates()
    df = df.dropna(how="all")

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].fillna(df[c].mode(dropna=True)[0] if not df[c].mode(dropna=True).empty else "Unknown")

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
