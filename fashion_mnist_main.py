"""Run the Fashion-MNIST optimizer comparison with PyTorch Lightning."""
import argparse
import json
import os
import time

import lightning.pytorch as pl
import numpy as np
import torch
from dataloader.fashion_mnist_data import CLASSES, FashionMNISTDataModule, load_fashion_mnist, preprocess
from model.fashion_mnist_model import FashionCNN, model_config, OPTIMIZERS


def train(name, Xtr, ytr, Xev, yev, epochs, seed, batch=128, lr=None):
    """Train one optimizer for a given learning rate using the Lightning Trainer."""
    lr = lr if lr is not None else 1e-3
    torch.manual_seed(seed)
    model = FashionCNN(optimizer_name=name, lr=lr, max_epochs=epochs)
    data = FashionMNISTDataModule(
        Xtr, ytr, Xev, yev, batch_size=batch, seed=seed
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    hist = {
        "train_loss": list(model.train_losses),
        "eval_loss": list(model.eval_losses),
        "eval_acc": list(model.eval_accs),
        "iter_loss": list(model.iter_losses),
    }
    return hist


def save(res, path):
    """Write JSON results atomically to avoid partial writes on interruption."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


def main():
    """Run the learning-rate sweep and the final optimization comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sweep-epochs", type=int, default=2)
    ap.add_argument("--sweep-size", type=int, default=20000)
    ap.add_argument("--out", help="results path; defaults to results/results_fashioncnn.json")
    ap.add_argument("--only", nargs="*", help="subset of optimizers")
    args = ap.parse_args()
    if args.out is None:
        args.out = "results/results_fashioncnn_pytorch.json"
    if min(args.epochs, args.seeds, args.sweep_epochs, args.sweep_size) < 1:
        ap.error("epochs, seeds, sweep-epochs and sweep-size must be positive")
    if args.sweep_size > 50000:
        ap.error("sweep-size must not overlap the last 10000 validation examples")
    names = args.only or list(OPTIMIZERS)
    if len(names) != len(set(names)) or any(name not in OPTIMIZERS for name in names):
        ap.error("--only must contain unique names from the optimizer registry")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    config = dict(epochs=args.epochs, seeds=args.seeds, batch=128,
                  sweep_epochs=args.sweep_epochs, sweep_size=args.sweep_size,
                  **model_config())
    if os.path.exists(args.out):
        with open(args.out) as result_file:
            res = json.load(result_file)
        if res.get("config") != config:
            raise ValueError("Saved results use a different model or configuration; choose a new --out path.")
    else:
        res = {}
    res.setdefault("config", config)
    res.setdefault("sweep", {})
    res.setdefault("best_lr", {})
    res.setdefault("runs", {})

    xtr, ytr, xte, yte = load_fashion_mnist()
    Xtr, Xte = preprocess(xtr), preprocess(xte)

    def sweep_one(name, lr):
        """Evaluate a single learning rate using a short Lightning training run."""
        t0 = time.time()
        h = train(name, Xtr[:args.sweep_size], ytr[:args.sweep_size],
                  Xtr[-10000:], ytr[-10000:], args.sweep_epochs, seed=123, lr=lr)
        v = h["eval_loss"][-1]
        res["sweep"].setdefault(name, {})
        res["sweep"][name][repr(lr)] = float(v) if np.isfinite(v) else None
        print(f"[sweep] {name:9s} lr={lr:<8g} val_loss={v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    for name in names:
        if name in res["best_lr"]:
            continue
        res["sweep"][name] = {}
        _, grid = OPTIMIZERS[name]
        for lr in grid:
            sweep_one(name, lr)
        for _ in range(3):
            ok = {float(k): v for k, v in res["sweep"][name].items() if v is not None}
            if not ok:
                break
            best, lo, hi = min(ok, key=ok.get), min(res["sweep"][name], key=float), max(res["sweep"][name], key=float)
            if best == float(hi):
                sweep_one(name, float(hi) * 3)
            elif best == float(lo):
                sweep_one(name, float(lo) / 3)
            else:
                break
        ok = {float(k): v for k, v in res["sweep"][name].items() if v is not None}
        if not ok:
            raise RuntimeError(f"All learning-rate trials diverged for {name}.")
        res["best_lr"][name] = min(ok, key=ok.get)
        save(res, args.out)

    for name in names:
        lr = res["best_lr"][name]
        runs = res["runs"].setdefault(name, [])
        for seed in range(len(runs), args.seeds):
            t0 = time.time()
            h = train(name, Xtr, ytr, Xte, yte, args.epochs, seed=seed, lr=lr)
            h["seed"] = seed
            runs.append(h)
            save(res, args.out)
            print(f"[train] {name:9s} lr={lr:<7g} seed={seed} test_acc={h['eval_acc'][-1]:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
