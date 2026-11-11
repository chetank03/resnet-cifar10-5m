# Builds the score figures (figs/uploads, seeds, progress) from the public scores in docs/EXPERIMENTS.md; make_examples.py builds the held-out ones. Run: py -3.11 make_figs.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
HOT, SEED, GREY = "#c0630f", "#0f6e73", "#9aa7af"

# 1. Every scored upload
up = sorted([("0.911 recipe, seed 41 (blur + zoom-out)", .911, HOT), ("800 epochs", .902, GREY), ("kd4: 3-teacher distillation", .902, GREY),
      ("t2, 400 ep, no KD", .899, GREY), ("AugMix instead of TrivialAugment", .899, GREY), ("Second KD round", .899, GREY),
      ("Stronger blur + zoom", .898, GREY), ("9 teachers, 1200 ep", .898, GREY), ("600 ep, no KD", .897, GREY),
      ("ResizeMix instead of CutMix", .897, GREY), ("No CutMix", .895, GREY), ("kd4, seed 201", .895, GREY), ("SAM", .895, GREY),
      ("MSE distillation", .894, GREY), ("Cutout 16x16", .894, GREY), ("36.7M teacher", .893, GREY),
      ("Wider, shallower student", .892, GREY), ("Better teachers", .891, GREY), ("Delayed downsampling", .896, GREY), ("WRN-22-4 (4.30M)", .891, GREY), ("TTA + class balancing", .891, GREY),
      ("0.911 recipe, seed 42", .895, SEED), ("0.911 recipe, seed 43", .898, SEED), ("Shake-Shake, 1800 ep", .897, GREY),
      ("t2, 200 ep (all 50k, CutMix)", .889, GREY), ("First KD student (T=4)", .888, GREY), ("Cross-fold distillation", .875, GREY),
      ("Baseline, 45k, 200 ep", .874, GREY)], key=lambda u: -u[1])
fig, ax = plt.subplots(figsize=(7.0, 3.5))
ax.axvspan(.895, .911, color=HOT, alpha=.12, lw=0, label="0.911 recipe across seeds 41-43")
for i, (n, v, c) in enumerate(up):
    ax.barh(i, v - .86, left=.86, color=c, height=.7)
    ax.text(v + .0005, i, f"{v:.3f}", va="center", fontsize=6.5)
ax.set_yticks(range(len(up)), [u[0] for u in up], fontsize=6.5); ax.invert_yaxis()
ax.set_xlim(.86, .916); ax.set_xlabel("Public Kaggle accuracy"); ax.legend(loc="lower right", frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figs/uploads.pdf"); plt.close(fig)

# 2. Seed spread
grp = [("t2, 400 ep, no KD", [.899, .894, .899]), ("kd4 recipe", [.902, .895]), ("0.911 recipe", [.911, .895, .898])]
fig, ax = plt.subplots(figsize=(3.3, 1.5))
for i, (n, vs) in enumerate(grp):
    ax.plot([min(vs), max(vs)], [i, i], color="#d5dce0", lw=5, solid_capstyle="round", zorder=1)
    for j, v in enumerate(vs):
        ax.scatter(v, i, s=28, color=HOT if v == .911 else SEED, zorder=2, edgecolor="white", lw=.8)
ax.set_yticks(range(3), [g[0] for g in grp]); ax.set_ylim(-.6, 2.6); ax.set_xlim(.886, .914)
ax.set_xlabel("Public Kaggle accuracy (one dot per seed)")
fig.tight_layout(); fig.savefig("figs/seeds.pdf"); plt.close(fig)

# 3. How the score moved, one point per milestone
ms = [("Baseline\n45k, 200 ep", .874), ("All 50k,\ndeeper, CutMix", .889), ("400\nepochs", .899), ("Ensemble\ndistillation", .902),
      ("Blur +\nzoom-out", .911)]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
ax.axhspan(.895, .911, color=HOT, alpha=.12, lw=0)
ax.text(3.8, .8955, "same recipe,\nseeds 41-43", fontsize=6, color=HOT, ha="right", va="bottom")
ax.plot(range(5), [m[1] for m in ms], color=SEED, marker="o", ms=4, lw=1.4)
ax.scatter([4, 4], [.895, .898], marker="x", color=HOT, s=22, zorder=3)
for i, (_, v) in enumerate(ms): ax.text(i, v + .0018, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(range(5), [m[0] for m in ms], fontsize=6.3); ax.set_ylim(.868, .918); ax.set_xlim(-.4, 4.4)
ax.set_ylabel("Public Kaggle accuracy"); ax.grid(axis="y", alpha=.3, lw=.5)
fig.tight_layout(); fig.savefig("figs/progress.pdf"); plt.close(fig)
print("ok2")

# 4. Public vs private score for all 33 submissions (private revealed after the deadline)
D = [(.869,.898),(.877,.895),(.877,.897),(.876,.902),(.875,.898),(.866,.895),(.878,.897),(.876,.896),(.873,.891),(.873,.898),(.876,.911),
     (.881,.897),(.870,.891),(.868,.895),(.857,.892),(.874,.894),(.872,.899),(.872,.893),(.877,.894),(.882,.898),(.871,.902),(.872,.899),
     (.871,.902),(.864,.899),(.873,.894),(.877,.891),(.869,.895),(.854,.875),(.870,.897),(.871,.899),(.866,.889),(.863,.888),(.851,.874)]
SEL = [(.876, .911), (.876, .902)]; BEST = (.882, .898)
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.axhspan(.886, .890, color=SEED, alpha=.10, lw=0); ax.text(.8745, .8865, "private top 10", fontsize=6, color=SEED)
for pr, pu in D:
    sel, best = (pr, pu) in SEL, (pr, pu) == BEST
    ax.scatter(pu, pr, s=28 if (sel or best) else 14, color=HOT if sel else SEED if best else GREY, edgecolor="white", lw=.6, zorder=3)
ax.text(.911, .8785, "selected\n(best public)", fontsize=6.2, color=HOT, ha="right")
ax.text(.8985, .8832, "best private", fontsize=6.2, color=SEED, ha="center")
ax.set_xlim(.872, .914); ax.set_ylim(.849, .890); ax.grid(alpha=.3, lw=.5)
ax.set_xlabel("Public score (what we chose by)"); ax.set_ylabel("Private score (final ranking)")
fig.tight_layout(); fig.savefig("figs/private.pdf"); plt.close(fig)
print("ok3")
