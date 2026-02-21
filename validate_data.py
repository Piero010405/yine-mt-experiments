"""
Validate data files
This script checks the integrity of the data files before training. It ensures that:
"""
import sys
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("data")

def die(msg):
    """Prints an error message and exits the program."""
    print(f"❌ VALIDATION FAILED: {msg}")
    sys.exit(1)

def ok(msg):
    """Prints a success message."""
    print(f"✅ {msg}")

def main():
    """
    Performs validation checks on the data files to ensure they are correctly 
    formatted and consistent.
    """
    pos_path = DATA_DIR / "positive.parquet"
    neg_path = DATA_DIR / "negatives.parquet"
    split_path = DATA_DIR / "split_v1.json"

    for p in [pos_path, neg_path, split_path]:
        if not p.exists():
            die(f"Missing required file: {p}")

    ok("Files exist")

    # Load splits
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    required_split_keys = ["train_ids", "dev_ids", "test_ids"]
    for k in required_split_keys:
        if k not in splits:
            die(f"split_v1.json missing key: {k}")

    train_ids = set(splits["train_ids"])
    dev_ids = set(splits["dev_ids"])
    test_ids = set(splits["test_ids"])

    if len(train_ids) == 0 or len(dev_ids) == 0 or len(test_ids) == 0:
        die("One of split sets is empty")

    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        die("Splits overlap (train/dev/test are not disjoint)")

    ok(f"Splits OK | train={len(train_ids)} dev={len(dev_ids)} test={len(test_ids)}")

    # Load positives
    pos = pd.read_parquet(pos_path)
    required_pos = ["pair_id", "spanish", "yine"]
    for c in required_pos:
        if c not in pos.columns:
            die(f"positive.parquet missing column: {c}")

    if pos["pair_id"].isna().any():
        die("positive.parquet has null pair_id")
    if pos["spanish"].isna().any():
        die("positive.parquet has null spanish")
    if pos["yine"].isna().any():
        die("positive.parquet has null yine")

    ok(f"Positives loaded: {len(pos)}")

    # Ensure split coverage
    pos_ids = set(pos["pair_id"].astype(str))
    missing_train = train_ids - pos_ids
    missing_dev = dev_ids - pos_ids
    missing_test = test_ids - pos_ids

    if missing_train or missing_dev or missing_test:
        die(
            f"Split IDs not fully present in positives | "
            f"missing_train={len(missing_train)} missing_dev={len(missing_dev)} missing_test={len(missing_test)}"
        )

    ok("All split ids exist in positives")

    # Load negatives
    neg = pd.read_parquet(neg_path)
    required_neg = ["pair_id", "source_text", "negative_text", "severity", "rule_id"]
    for c in required_neg:
        if c not in neg.columns:
            die(f"negatives.parquet missing column: {c}")

    if neg["pair_id"].isna().any():
        die("negatives.parquet has null pair_id")
    if neg["source_text"].isna().any():
        die("negatives.parquet has null source_text")
    if neg["negative_text"].isna().any():
        die("negatives.parquet has null negative_text")
    if neg["severity"].isna().any():
        die("negatives.parquet has null severity")

    # severity range sanity
    if (neg["severity"] <= 0).any() or (neg["severity"] > 1.5).any():
        die("negatives.parquet severity has unexpected range (expected >0 and <=1.5)")

    ok(f"Negatives loaded: {len(neg)}")

    # Negatives should not include dev/test ideally (but if they do, we ignore them in training)
    neg_ids = set(neg["pair_id"].astype(str))
    neg_test_overlap = len(neg_ids & test_ids)
    neg_dev_overlap = len(neg_ids & dev_ids)

    ok(
        f"Negatives overlap check | dev_overlap={neg_dev_overlap} test_overlap={neg_test_overlap} (will be ignored if present)"
    )

    print("✅ DATA VALIDATION PASSED. Ready to train.")

if __name__ == "__main__":
    main()
