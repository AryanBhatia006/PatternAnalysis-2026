"""Train the baseline classifier or the Siamese network on the ISIC melanoma data.

The checkpoint is chosen by validation loss only; the test split is evaluated
once at the end, with the best checkpoint, and never used to pick anything.
"""
import argparse
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import dataset as d
from modules import BaselineCNN, ContrastiveLoss, SiameseEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG_TRAIN, DEBUG_VAL, DEBUG_EPOCHS = 256, 128, 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subsample(splits, seed):
    """Debug mode: small random subsets of each split (still within that split)."""
    sizes = {"train": DEBUG_TRAIN, "val": DEBUG_VAL, "test": DEBUG_VAL}
    out = {name: df.sample(n=min(sizes[name], len(df)), random_state=seed).reset_index(drop=True)
           for name, df in splits.items()}
    for name, df in out.items():
        if (df["target"] == 1).sum() < 2:
            raise RuntimeError(f"debug {name} subset has fewer than 2 malignant images")
    return out


def make_step(model_name, device, args, splits):
    """Return (net, step) where step(batch) -> (loss, batch_size)."""
    if model_name == "baseline":
        net = BaselineCNN().to(device)
        train_targets = splits["train"]["target"]  # pos_weight from the TRAIN split only
        pos_weight = (train_targets == 0).sum() / (train_targets == 1).sum()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(pos_weight)], device=device))

        def step(batch):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # float, same shape as logits [N, 1]
            return criterion(net(images), labels), len(images)
    else:
        net = SiameseEncoder().to(device)
        criterion = ContrastiveLoss(margin=args.margin)

        def step(batch):
            img1, img2, labels = batch  # label 0 = similar, 1 = dissimilar
            n = len(img1)
            emb = net(torch.cat([img1, img2]).to(device))  # same encoder for both images
            return criterion(emb[:n], emb[n:], labels.to(device)), n
    return net, step


def run_epoch(net, step, loader, optimizer):
    """One training pass over `loader`. Returns mean loss per sample."""
    net.train()
    total, count = 0.0, 0
    for batch in loader:
        loss, n = step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * n
        count += n
    return total / count


def evaluate(net, step, loader):
    """Mean loss per sample with no gradients and the model in eval mode."""
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, n = step(batch)
            total += loss.item() * n
            count += n
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
    parser.add_argument("--model", choices=["baseline", "siamese"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs",
                        help="relative paths are taken from this script's folder")
    parser.add_argument("--debug", action="store_true",
                        help=f"{DEBUG_TRAIN} train / {DEBUG_VAL} val images, {DEBUG_EPOCHS} epochs")
    args = parser.parse_args()
    if args.debug:
        args.epochs = DEBUG_EPOCHS
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device: {device} | model: {args.model} | epochs: {args.epochs} | seed: {args.seed}"
          f"{' | DEBUG' if args.debug else ''}")

    image_dir = os.path.join(args.data_dir, "combined_images")
    splits = d.split_by_id(d.load_clean_metadata(args.data_dir), seed=args.seed)
    d.check_no_id_overlap(splits)
    if args.debug:
        splits = subsample(splits, args.seed)
    if args.model == "baseline":
        train_loader, val_loader, test_loader = d.get_dataloaders(
            splits, image_dir, args.batch_size, args.num_workers, seed=args.seed)
    else:
        train_loader, val_loader, test_loader = d.get_pair_dataloaders(
            splits, image_dir, args.batch_size, args.num_workers, seed=args.seed)

    net, step = make_step(args.model, device, args, splits)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    best_val, ckpt_path = float("inf"), os.path.join(out_dir, f"{args.model}_best.pt")
    train_losses, val_losses = [], []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(net, step, train_loader, optimizer)
        val_loss = evaluate(net, step, val_loader)  # val split only
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        saved = ""
        if val_loss < best_val:  # best checkpoint is chosen by validation loss alone
            best_val = val_loss
            torch.save({"model": net.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "args": vars(args)}, ckpt_path)
            saved = "  (saved best)"
        print(f"epoch {epoch:3d} | train loss {train_loss:.6f} | val loss {val_loss:.6f} "
              f"| {time.time() - t0:.1f}s{saved}")
    total_seconds = time.time() - start

    net.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(net, step, test_loader)  # evaluated once, after training
    print(f"test loss (best checkpoint, epoch {val_losses.index(best_val) + 1}): {test_loss:.6f}")

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", label="train")
    plt.plot(epochs, val_losses, marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{args.model} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{args.model}_loss.png"), dpi=120)
    plt.close()

    peak = (f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MB" if device.type == "cuda"
            else "n/a (cpu)")
    print(f"parameters: {n_params} | total training time: {total_seconds:.1f}s "
          f"| peak GPU memory: {peak}")


if __name__ == "__main__":
    main()
