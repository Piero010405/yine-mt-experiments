"""
Data loading and dataset building for the Yine MT experiments.
"""
import pandas as pd
from datasets import Dataset, DatasetDict

def load_positive(path_parquet: str) -> pd.DataFrame:
    """
    Loads the positive examples from a parquet file and returns a DataFrame. The expected columns in 
    the parquet file are:
    """
    df = pd.read_parquet(path_parquet)
    # expected: pair_id, spanish, yine, source
    return df

def load_negatives(path_parquet: str) -> pd.DataFrame:
    """
    Loads the negative examples from a parquet file and returns a DataFrame. The expected columns in
    """
    df = pd.read_parquet(path_parquet)
    # expected: pair_id, source_text, negative_text, severity, rule_id, ...
    return df

def load_splits(split_json: dict):
    """
    Loads the train/dev/test splits from a JSON object and returns three sets of pair_ids for each 
    split.
    """
    train_ids = set(split_json["train_ids"])
    dev_ids   = set(split_json["dev_ids"])
    test_ids  = set(split_json["test_ids"])
    return train_ids, dev_ids, test_ids

def build_baseline_datasets(pos_df: pd.DataFrame, split_json: dict) -> DatasetDict:
    """
    Builds the baseline datasets for train/dev/test using only the positive examples. The splits are
    """
    train_ids, dev_ids, test_ids = load_splits(split_json)

    train = pos_df[pos_df["pair_id"].isin(train_ids)].copy()
    dev   = pos_df[pos_df["pair_id"].isin(dev_ids)].copy()
    test  = pos_df[pos_df["pair_id"].isin(test_ids)].copy()

    # unify columns
    for d in (train, dev, test):
        d["src_text"] = d["spanish"].astype(str)
        d["tgt_text"] = d["yine"].astype(str)

    return DatasetDict({
        "train": Dataset.from_pandas(train[["pair_id","src_text","tgt_text"]].reset_index(drop=True)),
        "dev":   Dataset.from_pandas(dev[["pair_id","src_text","tgt_text"]].reset_index(drop=True)),
        "test":  Dataset.from_pandas(test[["pair_id","src_text","tgt_text"]].reset_index(drop=True)),
    })

def build_nsl_train_dataset(pos_df: pd.DataFrame, neg_df: pd.DataFrame, split_json: dict) -> DatasetDict:
    """
    NSL only uses negatives in TRAIN.
    DEV/TEST always evaluated on positives.
    """
    baseline = build_baseline_datasets(pos_df, split_json)
    train_ids, _, _ = load_splits(split_json)

    # positives (train)
    pos_train = pos_df[pos_df["pair_id"].isin(train_ids)].copy()
    pos_train["src_text"] = pos_train["spanish"].astype(str)
    pos_train["tgt_text"] = pos_train["yine"].astype(str)
    pos_train["is_negative"] = 0
    pos_train["severity"] = 1.0

    # negatives (train) — align by pair_id in train
    neg_train = neg_df[neg_df["pair_id"].isin(train_ids)].copy()
    neg_train["src_text"] = neg_train["source_text"].astype(str)
    # target for training becomes the NEGATIVE text (we penalize it)
    neg_train["tgt_text"] = neg_train["negative_text"].astype(str)
    neg_train["is_negative"] = 1
    # severity already in file
    if "severity" not in neg_train.columns:
        neg_train["severity"] = 0.6

    # concat
    mix = pd.concat([
        pos_train[["pair_id","src_text","tgt_text","is_negative","severity"]],
        neg_train[["pair_id","src_text","tgt_text","is_negative","severity"]],
    ], ignore_index=True)

    # shuffle for better mixing (stable via seed in trainer)
    mix = mix.sample(frac=1.0, random_state=split_json.get("seed", 42)).reset_index(drop=True)

    return DatasetDict({
        "train": Dataset.from_pandas(mix),
        "dev": baseline["dev"],
        "test": baseline["test"]
    })
