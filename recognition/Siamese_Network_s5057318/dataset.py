"""Patient-level train/val/test split of the cleaned ISIC metadata."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SIZE = 224


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


def compute_mean_std(df, image_dir):
    """Per-channel mean/std over the images in `df` (pass the train split only)."""
    to_tensor = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor()])
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    for isic_id in df["isic_id"]:
        img = to_tensor(_load_rgb(image_dir, isic_id)).double()
        total += img.sum(dim=(1, 2))
        total_sq += (img ** 2).sum(dim=(1, 2))
    n_pixels = len(df) * IMAGE_SIZE * IMAGE_SIZE
    mean = total / n_pixels
    std = (total_sq / n_pixels - mean ** 2).sqrt()
    return mean.float().tolist(), std.float().tolist()


def _load_rgb(image_dir, isic_id):
    return Image.open(os.path.join(image_dir, f"{isic_id}.jpg")).convert("RGB")


def build_transform(mean, std, train):
    """Augmentation (flips, small rotation) is applied for the train split only."""
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(15)]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class ISICDataset(Dataset):
    """One split of the cleaned metadata; images are read from <image_dir>/<isic_id>.jpg.

    Items are (image tensor of shape 3x224x224, float label). The split's
    DataFrame (with `real_id`) is kept in `self.df`.
    """

    def __init__(self, split_df, image_dir, transform):
        self.df = split_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(_load_rgb(self.image_dir, row["isic_id"]))
        return image, torch.tensor(row["target"], dtype=torch.float32)


def get_dataloaders(splits, image_dir, batch_size=32, num_workers=0, seed=SEED):
    """Return (train_loader, val_loader, test_loader) from the dict of split_by_id.

    Normalisation statistics come from the train split only; only the train
    loader augments and shuffles.
    """
    mean, std = compute_mean_std(splits["train"], image_dir)
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for name in ("train", "val", "test"):
        train = name == "train"
        dataset = ISICDataset(splits[name], image_dir, build_transform(mean, std, train))
        dataset.mean, dataset.std = mean, std
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=num_workers,
                                  generator=generator if train else None))
    return tuple(loaders)


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
