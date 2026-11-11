"""Inference-only improvements for a trained checkpoint (no retraining, same model, same 5M budget).

1. More test-time views: average softmax over the image, its mirror, and 2-pixel shifts of both (10 views).
2. Class balancing: the Kaggle test set (2,000 images) has exactly 200 images per class.
   Sinkhorn scaling rescales each class's probability column until every class gets an equal share of the
   probability mass; the argmax then follows the balanced probabilities.

Usage:
  python tta_balance.py check                  # sanity check on run 1's 5k held-out images (best.pt)
  python tta_balance.py best_t2_400ep.pt out   # writes out_tta.csv, out_bal.csv, out_ttabal.csv
"""
import csv, random, sys
import torch, torch.nn.functional as F
import torchvision.transforms as T
from train import ResNet, PNGs, read_rows, n_params, from_ckpt, DATA

dev = "cuda" if torch.cuda.is_available() else "cpu"
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
SHIFTS = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]

def load(path):
    ck = torch.load(path, map_location=dev)
    m = from_ckpt(ck) if "sd" in ck else ResNet()
    m.load_state_dict(ck["sd"] if "sd" in ck else ck)
    assert n_params(m) <= 5_000_000
    return m.to(dev).eval()

def shift(x, dy, dx):  # pad 2 px with zeros (the normalized mean), then crop back to 32x32 at an offset
    p = F.pad(x, (2, 2, 2, 2))
    return p[:, :, 2 + dy:34 + dy, 2 + dx:34 + dx]

@torch.no_grad()
def probs(model, ds, views):
    out = []
    for x, _ in torch.utils.data.DataLoader(ds, 500):
        x = x.to(dev)
        s = sum(model(shift(v, dy, dx)).softmax(1) for v in (x, x.flip(3)) for dy, dx in views)
        out.append((s / (2 * len(views))).cpu())
    return torch.cat(out)

def balance(p, iters=100):  # Sinkhorn: each row sums to 1, each column to n/10
    q = p.clone().double()
    for _ in range(iters):
        q = q / q.sum(0, keepdim=True) * (len(q) / q.shape[1])
        q = q / q.sum(1, keepdim=True)
    return q.float()

def hard_balance(p):  # exactly n/10 per class, maximizing total log-probability (optimal assignment)
    from scipy.optimize import linear_sum_assignment
    k = len(p) // p.shape[1]
    cost = -torch.log(p.clamp_min(1e-12)).repeat_interleave(k, dim=1).numpy()
    r, c = linear_sum_assignment(cost)
    pred = torch.empty(len(p), dtype=torch.long); pred[torch.as_tensor(r)] = torch.as_tensor(c) // k
    return pred

def write(path, pred):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"]); w.writerows(enumerate(pred.tolist()))

if __name__ == "__main__":
    if sys.argv[1] == "check":  # run 1 never trained on these 5k images (seed-0 shuffle, first 5000 rows)
        rows = read_rows(DATA / "train_labels.csv", "label"); random.Random(0).shuffle(rows)
        val, y = PNGs(rows[:5000], DATA / "train_images", tf), torch.tensor([r[1] for r in rows[:5000]])
        m = load("best.pt")
        p1, p5 = probs(m, val, SHIFTS[:1]), probs(m, val, SHIFTS)
        acc = lambda p: (p.argmax(1) == y).float().mean().item()
        print(f"held-out 5k  flip only {acc(p1):.4f} | 10 views {acc(p5):.4f} | flip+balance {acc(balance(p1)):.4f} | 10 views+balance {acc(balance(p5)):.4f}")
        print("true class counts:", torch.bincount(y, minlength=10).tolist())
        g = torch.Generator().manual_seed(0); gains = []  # 20 test-like sets: exactly 200 per class
        for _ in range(20):
            idx = torch.cat([torch.nonzero(y == c).flatten()[torch.randperm(int((y == c).sum()), generator=g)[:200]] for c in range(10)])
            for p in (p1, p5):
                a0 = (p[idx].argmax(1) == y[idx]).sum().item(); a1 = (hard_balance(p[idx]) == y[idx]).sum().item()
                gains.append(a1 - a0)
        fl, tv = gains[0::2], gains[1::2]
        print(f"exact balance on 20 balanced 2,000-image sets: images gained, flip-only views {sum(fl)/20:+.1f} (min {min(fl)}, max {max(fl)}) | 10 views {sum(tv)/20:+.1f} (min {min(tv)}, max {max(tv)})")
    else:
        m = load(sys.argv[1]); out = sys.argv[2]
        test = PNGs(read_rows(DATA / "test.csv", "id"), DATA / "test_images", tf)
        p1, p5 = probs(m, test, SHIFTS[:1]), probs(m, test, SHIFTS)
        base = p1.argmax(1)
        for tag, pred in [("tta", p5.argmax(1)), ("hbal", hard_balance(p1)), ("ttahbal", hard_balance(p5))]:
            write(f"{out}_{tag}.csv", pred)
            print(f"{out}_{tag}.csv  changed vs flip-only: {(pred != base).sum().item():3d}  class counts: {torch.bincount(pred, minlength=10).tolist()}")
        print("flip-only class counts:", torch.bincount(base, minlength=10).tolist())
