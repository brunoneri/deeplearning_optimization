"""Figures and LaTeX table for the FashionCNN optimizer comparison."""
import argparse
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataloader.fashion_mnist_data import CLASSES, load_fashion_mnist

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/results_fashioncnn.json")
parser.add_argument("--images", help="figure directory; defaults to images")
parser.add_argument("--table", help="LaTeX table path; defaults to results/table.tex")
args = parser.parse_args()
res = json.load(open(args.results))
image_dir = Path(args.images or "images")
table_path = Path(args.table or "results/table.tex")
GROUPS = [("SGD, Momentum and classical adaptive methods",
           ["SGD", "Momentum", "Nesterov", "AdaGrad", "AdaDelta", "RMSProp"]),
          ("Adam family, AdaBelief and Lion",
           ["Adam", "AMSGrad", "AdamW", "Nadam", "RAdam", "AdaBelief", "Lion"])]
names = [n for _, g in GROUPS for n in g if n in res["runs"] and res["runs"][n]]
cmap = plt.get_cmap("tab20")
COLORS = {n: cmap(i) for i, n in enumerate(n for _, g in GROUPS for n in g)}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def stack(n, key):
    return np.array([r[key] for r in res["runs"][n]])


def seed_std(values):
    return values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])


def two_panel(fn, key, ylabel, logy=False, ylim=None, smooth=None, upto=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, g) in zip(axes, GROUPS):
        for n in g:
            if n not in names:
                continue
            a = stack(n, key)
            if upto:
                a = a[:, :upto]
            window = min(smooth, a.shape[1]) if smooth else None
            if window:
                k = np.ones(window) / window
                a = np.array([np.convolve(r, k, mode="valid") for r in a])
            m, s = a.mean(0), seed_std(a)
            x = np.arange(1, len(m) + 1) if not window else np.arange(window, window + len(m))
            ax.plot(x, m, label=f"{n}", color=COLORS[n], lw=1.5)
            ax.fill_between(x, m - s, m + s, color=COLORS[n], alpha=0.15)
        ax.set_title(title); ax.legend(fontsize=8)
        ax.set_xlabel("iteration" if smooth else "epoch"); ax.set_ylabel(ylabel)
        if logy: ax.set_yscale("log")
        if ylim: ax.set_ylim(*ylim)
    plt.tight_layout(); plt.savefig(image_dir / f"{fn}.png", dpi=180); plt.close()


image_dir.mkdir(parents=True, exist_ok=True)
table_path.parent.mkdir(parents=True, exist_ok=True)


# 1) dataset samples
xtr, ytr, _, _ = load_fashion_mnist()
fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
for c, ax in enumerate(axes.ravel()):
    i = int((ytr == c).nonzero()[0, 0]); ax.imshow(xtr[i], cmap="gray"); ax.set_title(CLASSES[c], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(image_dir / "fmnist_samples.png", dpi=180); plt.close()

# 2) learning-rate sweep
sw = [n for n in res["sweep"] if res["sweep"][n]]
cols = 4; rows = int(np.ceil(len(sw) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(14, 2.4 * rows), squeeze=False)
for ax, n in zip(axes.ravel(), sw):
    pts = sorted((float(k), v) for k, v in res["sweep"][n].items() if v is not None)
    lr, v = zip(*pts); ax.semilogx(lr, v, "o-", color=COLORS.get(n, "k"), ms=4)
    b = res["best_lr"][n]; ax.plot(b, dict(pts)[b], "k*", ms=11, zorder=5)
    ax.set_title(n, fontsize=9); ax.set_xlabel("learning rate")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="x", labelsize=8)
for ax in axes.ravel()[len(sw):]: ax.axis("off")
for row_index in range(min(2, rows)):
    axes[row_index, 0].set_ylabel("validation loss")
plt.tight_layout(); plt.savefig(image_dir / "fmnist_lr_sweep.png", dpi=180); plt.close()

# 3-5) curves
two_panel("fmnist_train_loss", "train_loss", "training loss", logy=True)
accs = np.concatenate([stack(n, "eval_acc").ravel() for n in names])
accuracy_low = max(0.80, accs.min() - 0.01) if accs.min() >= 0.80 else max(0.0, accs.min() - 0.01)
two_panel("fmnist_test_acc", "eval_acc", "test accuracy", ylim=(accuracy_low, accs.max() + 0.005))
two_panel("fmnist_iter_loss", "iter_loss", "training loss (moving average)", logy=True, smooth=25, upto=2 * 469)

# 6) final accuracy bars
fin = {n: stack(n, "eval_acc")[:, -1] for n in names}
order = sorted(names, key=lambda n: fin[n].mean())
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(order, [100 * fin[n].mean() for n in order], xerr=[100 * seed_std(fin[n]) for n in order],
        color=[COLORS[n] for n in order], capsize=3)
lo = 100 * min(fin[n].mean() for n in order)
ax.set_xlim(lo - 1.5, 100 * max(fin[n].mean() for n in order) + 0.8)
for i, n in enumerate(order):
    ax.text(100 * fin[n].mean() + 0.1 + 100 * seed_std(fin[n]), i, f"{100*fin[n].mean():.2f}", va="center", fontsize=8)
ax.set_xlabel("final test accuracy (%)"); plt.tight_layout(); plt.savefig(image_dir / "fmnist_final_acc.png", dpi=180); plt.close()

# LaTeX table
rows = []
for n in sorted(names, key=lambda n: -fin[n].mean()):
    tl, el = stack(n, "train_loss")[:, -1], stack(n, "eval_loss")[:, -1]
    best_ep = int(np.argmax(stack(n, "eval_acc").mean(0))) + 1
    rows.append(f"{n} & {res['best_lr'][n]:.3g} & {tl.mean():.3f} & {el.mean():.3f} & "
                f"{100*fin[n].mean():.2f} $\\pm$ {100*seed_std(fin[n]):.2f} \\\\")
table_path.write_text("\n".join(rows) + "\n")
print("\n".join(rows))
