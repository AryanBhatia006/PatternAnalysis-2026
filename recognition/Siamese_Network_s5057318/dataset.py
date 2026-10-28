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


def build_transform(mean, std, train, augment="basic"):
    """Augmentation is applied for the train split only.

    "basic": flips and small rotation. "strong": basic plus colour jitter and a
    small random affine (translation, scale).
    """
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(15)]
        if augment == "strong":
            steps += [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                      transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1))]
        elif augment != "basic":
            raise ValueError(f"unknown augment {augment!r}")
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


def get_dataloaders(splits, image_dir, batch_size=32, num_workers=0, seed=SEED, augment="basic"):
    """Return (train_loader, val_loader, test_loader) from the dict of split_by_id.

    Normalisation statistics come from the train split only; only the train
    loader augments and shuffles.
    """
    mean, std = compute_mean_std(splits["train"], image_dir)
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for name in ("train", "val", "test"):
        train = name == "train"
        dataset = ISICDataset(splits[name], image_dir, build_transform(mean, std, train, augment))
        dataset.mean, dataset.std = mean, std
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=num_workers,
                                  generator=generator if train else None))
    return tuple(loaders)


class PairDataset(Dataset):
    """Image pairs (img1, img2, label) built from ONE split only.

    Label convention (matches ContrastiveLoss): 0 = similar pair (same class),
    1 = dissimilar pair (different class). Pair kind is set by `idx % 4`:
    0 = benign-benign, 1 = malignant-malignant, 2 and 3 = dissimilar
    (one benign, one malignant). So pairs are exactly 50% similar / 50%
    dissimilar, and malignant images are deliberately oversampled. An image is
    never paired with itself.

    With `fixed_seed=None` (train) pairs are drawn fresh on every access, so
    they change every epoch. With an int `fixed_seed` (val/test) all pairs are
    drawn once at construction and are identical on every run.
    """

    def __init__(self, split_df, image_dir, transform, num_pairs=None, fixed_seed=None):
        self.df = split_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        targets = self.df["target"].to_numpy()
        self.benign = torch.from_numpy(np.flatnonzero(targets == 0))
        self.malignant = torch.from_numpy(np.flatnonzero(targets == 1))
        if len(self.benign) < 2 or len(self.malignant) < 2:
            raise ValueError("each class needs at least 2 images to form same-class pairs")
        n = num_pairs if num_pairs is not None else len(self.df)
        self.num_pairs = n - n % 4  # whole number of (bb, mm, dis, dis) groups
        self.fixed = fixed_seed is not None
        if self.fixed:
            gen = torch.Generator().manual_seed(fixed_seed)
            self._pairs = [self._draw(k, gen) for k in range(self.num_pairs)]

    @staticmethod
    def _randint(n, gen):
        return int(torch.randint(n, (1,), generator=gen))

    def _same_class(self, pool, gen):
        i = self._randint(len(pool), gen)
        j = self._randint(len(pool) - 1, gen)
        if j >= i:  # skip i itself, so the two images are never the same
            j += 1
        return int(pool[i]), int(pool[j])

    def _draw(self, idx, gen):
        kind = idx % 4
        if kind == 0:
            return (*self._same_class(self.benign, gen), 0)
        if kind == 1:
            return (*self._same_class(self.malignant, gen), 0)
        a = int(self.benign[self._randint(len(self.benign), gen)])
        b = int(self.malignant[self._randint(len(self.malignant), gen)])
        if self._randint(2, gen):  # random order within the dissimilar pair
            a, b = b, a
        return a, b, 1

    def draw(self, idx):
        """Return (row_index_1, row_index_2, label) for pair `idx`, without loading images."""
        return self._pairs[idx] if self.fixed else self._draw(idx, None)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        i, j, label = self.draw(idx)
        img1 = self.transform(_load_rgb(self.image_dir, self.df.at[i, "isic_id"]))
        img2 = self.transform(_load_rgb(self.image_dir, self.df.at[j, "isic_id"]))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_pair_dataloaders(splits, image_dir, batch_size=32, num_workers=0,
                         seed=SEED, num_pairs=None, split_seed=None, augment="basic"):
    """Return (train_loader, val_loader, test_loader) of image pairs.

    Same normalisation (train mean/std) and augmentation policy as
    get_dataloaders: only train augments. Train pairs are re-drawn every epoch
    and shuffled; val/test pairs are drawn once from fixed seeds
    (split_seed + 1 and split_seed + 2; split_seed defaults to seed).
    """
    mean, std = compute_mean_std(splits["train"], image_dir)
    generator = torch.Generator().manual_seed(seed)
    split_seed = seed if split_seed is None else split_seed
    fixed_seeds = {"train": None, "val": split_seed + 1, "test": split_seed + 2}
    loaders = []
    for name in ("train", "val", "test"):
        train = name == "train"
        dataset = PairDataset(splits[name], image_dir, build_transform(mean, std, train, augment),
                              num_pairs=num_pairs, fixed_seed=fixed_seeds[name])
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
