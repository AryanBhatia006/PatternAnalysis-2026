"""Patient-level train/val/test split of the cleaned ISIC metadata."""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def extract_real_id(value):
    """Return the first non-dummy ID in a possibly comma-separated field, else None."""
    if pd.isna(value):
        return None
    for part in str(value).split(","):
        part = part.strip().strip("\"'").strip()
        if part and not part.lower().startswith("dummy"):
            return part
    return None


def load_clean_metadata(data_dir):
    """Load the csv and keep only rows with a real (non-dummy) ID in `real_id`."""
    df = pd.read_csv(os.path.join(data_dir, "cleaned_metadata.csv"),
                     dtype={"patient_id": str})
    df["real_id"] = df["patient_id"].map(extract_real_id)
    return df.dropna(subset=["real_id"]).reset_index(drop=True)


def split_by_id(df, fractions=FRACTIONS, seed=SEED):
    """Split rows into train/val/test so every real ID lands in exactly one split.

    IDs (not images) are assigned greedily, most-malignant first, to whichever
    split is furthest below its target share of both images and malignant
    images. This keeps split sizes and malignant % close to each other.
    Returns a dict {split_name: DataFrame}.
    """
    rng = np.random.default_rng(seed)
    per_id = df.groupby("real_id")["target"].agg(n="size", mal="sum")
    per_id = per_id.loc[rng.permutation(per_id.index)]  # seeded tie-breaking
    per_id = per_id.sort_values("mal", ascending=False, kind="stable")

    total_n, total_mal = per_id["n"].sum(), per_id["mal"].sum()
    names = list(fractions)
    count = {s: 0.0 for s in names}
    mal = {s: 0.0 for s in names}
    assign = {}
    for real_id, n, m in zip(per_id.index, per_id["n"], per_id["mal"]):
        def deficit(s):
            f = fractions[s]
            return ((f * total_n - count[s]) / (f * total_n)
                    + (f * total_mal - mal[s]) / max(f * total_mal, 1))
        best = max(names, key=deficit)
        assign[real_id] = best
        count[best] += n
        mal[best] += m

    split_col = df["real_id"].map(assign)
    return {s: df[split_col == s].reset_index(drop=True) for s in names}


def check_no_id_overlap(splits):
    """Raise AssertionError if any real ID appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["real_id"]) & set(splits[b]["real_id"])
            if shared:
                raise AssertionError(
                    f"{len(shared)} real IDs in both {a} and {b}, e.g. {sorted(shared)[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder containing cleaned_metadata.csv")
    args = parser.parse_args()

    df = load_clean_metadata(args.data_dir)
    splits = split_by_id(df)
    check_no_id_overlap(splits)

    print(f"{'split':<6}{'images':>8}{'IDs':>7}{'malignant':>11}{'malignant %':>13}")
    for name, part in splits.items():
        mal = int((part["target"] == 1).sum())
        print(f"{name:<6}{len(part):>8}{part['real_id'].nunique():>7}"
              f"{mal:>11}{100 * mal / len(part):>12.2f}%")
    assert sum(len(p) for p in splits.values()) == len(df), "rows lost in split"
    print("Overlap check: PASSED (no real ID appears in more than one split)")


if __name__ == "__main__":
    main()
