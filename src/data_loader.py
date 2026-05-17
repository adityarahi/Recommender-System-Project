"""Load MovieLens 100K, build the user-item matrix, and create train/test splits."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-100k")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_ratings(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return ratings DataFrame with columns [user_id, item_id, rating, timestamp]."""
    path = os.path.join(raw_dir, "u.data")
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df


def load_movies(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return movies DataFrame with columns [item_id, title]."""
    path = os.path.join(raw_dir, "u.item")
    df = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        names=["item_id", "title"] + [f"f{i}" for i in range(22)],
        usecols=["item_id", "title"],
        engine="python",
    )
    return df


def split_ratings(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings per user so every user appears in both train and test."""
    train_frames, test_frames = [], []
    for _, group in df.groupby("user_id"):
        if len(group) < 2:
            train_frames.append(group)
            continue
        tr, te = train_test_split(group, test_size=test_size, random_state=random_state)
        train_frames.append(tr)
        test_frames.append(te)
    train = pd.concat(train_frames).reset_index(drop=True)
    test = pd.concat(test_frames).reset_index(drop=True)
    return train, test


def save_splits(train: pd.DataFrame, test: pd.DataFrame, processed_dir: str = PROCESSED_DIR):
    os.makedirs(processed_dir, exist_ok=True)
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)


def load_splits(processed_dir: str = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    return train, test


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (users × items) pivot table; missing entries are NaN."""
    return df.pivot_table(index="user_id", columns="item_id", values="rating")


def get_data(force_resplit: bool = False):
    """One-call helper: load raw data, create splits (once), return everything."""
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    if force_resplit or not (os.path.exists(train_path) and os.path.exists(test_path)):
        df = load_ratings()
        train, test = split_ratings(df)
        save_splits(train, test)
    else:
        train, test = load_splits()

    movies = load_movies()
    matrix = build_user_item_matrix(train)
    return train, test, matrix, movies
