# Report figures from the 5,000 held-out training images (seed-0 shuffle, first 5,000), which the 45k models never trained on:
# distortion robustness, top confusions, example errors, and what the distortion tests look like.
# Run from the project root: PYTHONPATH=. py -3.11 report/make_examples.py
import collections, random
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from train import read_rows, DATA
from tta_balance import load, probs, tf, SHIFTS, dev

C = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
rows = read_rows(DATA / "train_labels.csv", "label"); random.Random(0).shuffle(rows); rows = rows[:5000]
img = np.stack([np.array(Image.open(DATA / "train_images" / f).convert("RGB")) for f, _ in rows])
y = np.array([c for _, c in rows]); X = torch.stack([tf(a) for a in img])
ds = torch.utils.data.TensorDataset(X, torch.from_numpy(y))

def blur(t, s):  # separable 5-tap Gaussian, reflect padding
    k = torch.exp(-(torch.arange(5, device=t.device) - 2.0) ** 2 / (2 * s * s)); k = k / k.sum(); c = t.shape[1]
    t = F.conv2d(F.pad(t, (2, 2, 0, 0), mode="reflect"), k.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(t, (0, 0, 2, 2), mode="reflect"), k.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)
low = lambda t: F.interpolate(F.interpolate(t, size=16, mode="bilinear", antialias=True), size=32, mode="bilinear")
small = lambda t: F.pad(F.interpolate(t, size=16, mode="bilinear", antialias=True), [8, 8, 8, 8])  # 0 = mean colour after normalising
TESTS = {"clean": lambda t: t, "blur": lambda t: blur(t, 1.2), "low16": low, "obj16": small}

@torch.no_grad()
def single(m, f):
    return sum((m(f(X[i:i + 500].to(dev))).argmax(1).cpu().numpy() == y[i:i + 500]).sum() for i in range(0, len(y), 500)) / len(y) * 100

MODELS = [("Distilled, no blur/zoom", "checkpoints/best_kd45.pt"), ("+ grayscale", "checkpoints/best_gray45.pt"),
          ("Final recipe (blur + zoom-out)", "checkpoints/best_bz45.pt")]
res = {}
for name, ck in MODELS:
    m = load(ck); tta = (probs(m, ds, SHIFTS).argmax(1).numpy() == y).mean() * 100
    res[name] = [tta] + [single(m, f) for f in TESTS.values()]
    print(f"{name:32s} TTA {tta:6.2f} | " + " ".join(f"{k} {v:6.2f}" for k, v in zip(TESTS, res[name][1:])))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8, "pdf.fonttype": 42})
HOT, SEED, GREY = "#c0630f", "#0f6e73", "#9aa7af"

# Robustness: blur and small objects, before and after the shift-targeted augmentation
show = ["Distilled, no blur/zoom", "Final recipe (blur + zoom-out)"]
fig, ax = plt.subplots(figsize=(3.3, 1.85))
labels = ["Clean", "Blur $\\sigma$=1.2", "Downscaled\nto 16 px", "16-px object"]
for j, (name, col) in enumerate(zip(show, [GREY, SEED])):
    vals = res[name][1:]
    ax.bar(np.arange(4) + (j - .5) * .38, vals, width=.38, color=col, label=name)
    for i, v in enumerate(vals): ax.text(i + (j - .5) * .38, v + 1, f"{v:.0f}", ha="center", fontsize=6)
ax.set_xticks(range(4), labels, fontsize=6.8); ax.set_ylim(0, 108); ax.set_ylabel("Held-out accuracy (%)")
ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("report/figs/robust.pdf"); plt.close(fig)

# Confusions and examples for the final recipe (45k twin)
P = probs(load("checkpoints/best_bz45.pt"), ds, SHIFTS).numpy(); p = P.argmax(1); err = np.where(p != y)[0]
pairs = collections.Counter((C[y[i]], C[p[i]]) for i in err)
print("held-out errors", len(err)); [print(f"{a}->{b} {n}") for (a, b), n in pairs.most_common(10)]
top = pairs.most_common(8)
fig, ax = plt.subplots(figsize=(3.3, 1.9))
ax.barh(range(len(top)), [n for _, n in top], color=[HOT] + [GREY] * (len(top) - 1), height=.65)
for i, (_, n) in enumerate(top): ax.text(n + .3, i, str(n), va="center", fontsize=7)
ax.set_yticks(range(len(top)), [f"{a} $\\to$ {b}" for (a, b), _ in top]); ax.invert_yaxis()
ax.set_xlabel(f"Errors on 5,000 held-out images ({len(err)} total)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("report/figs/confusions.pdf"); plt.close(fig)

pick = []
for (a, b), _ in top[:5]:
    ids = [i for i in err if C[y[i]] == a and C[p[i]] == b]
    pick += sorted(ids, key=lambda i: -P[i, p[i]])[:2]
fig, axs = plt.subplots(2, 5, figsize=(3.3, 1.75))
for ax, i in zip(axs.flat, pick):
    ax.imshow(img[i]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{C[y[i]]}\n$\\to$ {C[p[i]]} {P[i, p[i]]:.0%}", fontsize=5.5, pad=1.5)
fig.subplots_adjust(left=.01, right=.99, top=.86, bottom=.01, wspace=.06, hspace=.45)
fig.savefig("report/figs/examples.pdf"); plt.close(fig)

# What the distortion tests look like, on one held-out dog
x = torch.from_numpy(img[[i for i in range(len(y)) if y[i] == 5][3]]).permute(2, 0, 1)[None].float() / 255
views = [("original", x), ("blur $\\sigma$=1.2", blur(x, 1.2)), ("downscaled to 16 px", low(x)),
         ("16-px object", F.pad(F.interpolate(x, size=16, mode="bilinear", antialias=True), [8, 8, 8, 8], value=float(x.mean())))]
fig, axs = plt.subplots(1, 4, figsize=(3.3, 1.05))
for ax, (n, t) in zip(axs, views):
    ax.imshow(t[0].permute(1, 2, 0).clamp(0, 1).numpy()); ax.set_title(n, fontsize=6.5, pad=2); ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(left=.01, right=.99, top=.8, bottom=.02, wspace=.08)
fig.savefig("report/figs/distortions.pdf"); plt.close(fig)
print("ok")
