"""Train the single-lead rhythm CNN on PhysioNet/CinC 2017-style data.

Prepare data first:
    python tools/fetch_physionet2017.py --out-dir data/physionet2017
Then train:
    python models/train_rhythm_cnn.py --data-dir data/physionet2017 --epochs 30
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.metrics import f1_score, classification_report

# cuDNN 9.20 (torch cu130) mismatches on GB10; native conv works fine.
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhythm_cnn import RhythmCNN, CLASSES
from ecg_dataset import ECGRhythmDataset


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirs = args.data_dir if isinstance(args.data_dir, list) else [args.data_dir]
    clean = ConcatDataset([ECGRhythmDataset(d, fs=args.fs, seconds=args.seconds,
                                            limit=args.limit, augment=False) for d in dirs])
    aug = ConcatDataset([ECGRhythmDataset(d, fs=args.fs, seconds=args.seconds,
                                          limit=args.limit, augment=args.augment) for d in dirs])
    n = len(clean)
    n_val = max(1, int(0.2 * n))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_ds = Subset(aug, train_idx)   # augmented for training
    val_ds = Subset(clean, val_idx)     # clean for validation
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = RhythmCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    labels = []
    for sub in clean.datasets:
        labels += [lbl for _, lbl in sub.items]
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    weights = (weights / weights.mean()).to(device)  # inverse-frequency, mean-normalized
    print("class counts:", dict(zip(CLASSES, counts.tolist())),
          " weights:", [round(w, 2) for w in weights.cpu().tolist()])
    crit = nn.CrossEntropyLoss(weight=weights)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1 = 0.0
    since_best = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach()) * len(y)
            tr_correct += int((out.argmax(1) == y).sum())
            tr_n += len(y)
        tr_loss /= max(1, tr_n)
        tr_acc = tr_correct / max(1, tr_n)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x.to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                trues.extend(y.tolist())
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        acc = float(np.mean(np.array(preds) == np.array(trues)))
        sched.step(f1)
        # train_acc - val_acc is the overfitting gap.
        print(f"epoch {epoch:3d}  train_loss {tr_loss:.3f}  train_acc {tr_acc:.3f}  "
              f"val_acc {acc:.3f}  macro_F1 {f1:.3f}  gap {tr_acc - acc:+.3f}")
        if f1 >= best_f1:
            best_f1 = f1
            since_best = 0
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "fs": args.fs, "seconds": args.seconds}, args.out)
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"early stop: no macro-F1 gain in {args.patience} epochs")
                break

    print("\nFinal validation report:")
    print(classification_report(trues, preds, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, zero_division=0))
    print(f"best macro-F1: {best_f1:.3f}  -> saved {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, nargs="+",
                   help="one or more labeled dataset dirs (labels.csv + signals/)")
    p.add_argument("--fs", type=float, default=300)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="models/rhythm_cnn.pt")
    p.add_argument("--augment", action="store_true",
                   help="add random noise to training windows for robustness")
    p.add_argument("--patience", type=int, default=8,
                   help="early-stop after N epochs without macro-F1 gain (0=off)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
