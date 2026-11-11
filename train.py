# Starter: ResNet under 5M params on CIFAR-10.
# Block design follows He et al. 2016 and github.com/kuangliu/pytorch-cifar (cite if reused).
# v2: prof's Kaggle data, 45k/5k val split, submission.csv.
# v3: lines marked "# NEW" / "# CHANGED" add TrivialAugment + Cutout, AMP, EMA weights, flip-TTA, EPOCHS env var.
import csv, os, random  # CHANGED: + os (EPOCHS env var)
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # NEW: EMA of weights
import torchvision.transforms as T

DATA = Path(__file__).parent / "data"

class Block(nn.Module):  # ReLU(S(x) + F(x)), F = conv-BN-ReLU-conv-BN
    def __init__(self, cin, cout, stride=1, k=3, se=False, drop_path=0.0):  # CHANGED: optional SE + stochastic depth
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, k, 1, k // 2, bias=False), nn.BatchNorm2d(cout))
        self.s = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))  # 1x1 skip (K_i = 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cout, cout // 16, 1), nn.ReLU(inplace=True),  # NEW: squeeze-and-excitation (Hu et al. 2018)
                                nn.Conv2d(cout // 16, cout, 1), nn.Sigmoid()) if se else None
        self.drop_path = drop_path  # NEW: stochastic depth (Huang et al. 2016): drop F(x) per sample during training

    def forward(self, x):
        f = self.f(x)
        if self.se is not None:
            f = f * self.se(f)  # NEW: channel re-weighting
        if self.training and self.drop_path > 0:  # NEW
            keep = (torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_path).to(f.dtype)
            f = f * keep / (1 - self.drop_path)
        return F.relu(self.s(x) + f)

def shake(f1, f2, a, b):  # NEW: value a*f1+(1-a)*f2, gradient as if b*f1+(1-b)*f2 (Gastaldi 2017, "shake" forward and backward)
    y = b * f1 + (1 - b) * f2
    return y + (a * f1 + (1 - a) * f2 - y).detach()

class ShakeBlock(nn.Module):  # NEW: Shake-Shake block: S(x) + mix of two residual branches, mix drawn per image (and anew for backward) while training; 0.5/0.5 at eval
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        branch = lambda: nn.Sequential(nn.ReLU(), nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                       nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))  # first ReLU not in place: x feeds S(x) and the other branch too
        self.f1, self.f2 = branch(), branch()
        self.s = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))  # 1x1 skip, as in Block

    def forward(self, x):
        f1, f2 = self.f1(x), self.f2(x)
        if not self.training:
            return self.s(x) + 0.5 * (f1 + f2)
        a, b = torch.rand(2, x.size(0), 1, 1, 1, device=x.device, dtype=f1.dtype)
        return self.s(x) + shake(f1, f2, a, b)

class ResNet(nn.Module):
    def __init__(self, C=(64, 128, 256), B=(3, 3, 3), n_classes=10, se=False, drop_path=0.0, shake=False, S=(1, 2, 2)):  # CHANGED: se, drop_path, shake options; S = first-block stride per stage
        super().__init__()
        layers, cin = [nn.Conv2d(3, C[0], 3, 1, 1, bias=False), nn.BatchNorm2d(C[0]), nn.ReLU(inplace=True)], C[0]
        n, k = sum(B), 0
        for i, (c, b) in enumerate(zip(C, B)):
            for j in range(b):
                stride = S[i] if j == 0 else 1  # CHANGED: S=(1,2,1) keeps stage 3 at 16x16 (delayed downsampling)
                layers.append(ShakeBlock(cin, c, stride) if shake else  # NEW: SHAKE=1 -> Shake-Shake blocks (two branches, no SE / drop-path)
                              Block(cin, c, stride, se=se, drop_path=drop_path * k / max(n - 1, 1)))  # NEW: drop rate rises linearly with depth
                cin, k = c, k + 1
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())  # global avg pool = P x P
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        return self.fc(self.body(x))

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def from_ckpt(ck):  # NEW: rebuild the architecture a best*.pt was saved with (teachers, tta_balance)
    return ResNet(ck["C"], ck["B"], se=ck.get("se", False), shake=ck.get("shake", False), S=tuple(ck.get("S", (1,) + (2,) * (len(ck["C"]) - 1))))  # CHANGED: old checkpoints have no "S"; default stride 2 at every stage after the first (4-stage t3 too)

class PNGs(torch.utils.data.Dataset):
    def __init__(self, rows, folder, tf, folds=None):  # CHANGED: optional per-image fold id
        self.rows, self.folder, self.tf, self.folds = rows, folder, tf, folds
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        name, y = self.rows[i]
        x = self.tf(Image.open(self.folder / name).convert("RGB"))
        return (x, y, self.folds[i]) if self.folds else (x, y)  # CHANGED

def predict(m, x):  # NEW: flip-TTA, average softmax of image and its mirror
    return m(x).softmax(1) + m(x.flip(3)).softmax(1)  # NEW

def cutmix(x):  # NEW: paste a random box from a shuffled copy of the batch (Yun et al. 2019)
    perm = torch.randperm(x.size(0), device=x.device)
    h = w = x.size(2)
    if os.environ.get("MIX_RESIZE") == "1":  # NEW: ResizeMix (Qin et al. 2020): paste the WHOLE partner image shrunk to side tau*32, tau ~ U(0.1, 0.8)
        s = max(1, int(h * random.uniform(0.1, 0.8)))  # so the pasted patch always holds the object, never a background-only crop
        y0, x0 = random.randrange(h - s + 1), random.randrange(w - s + 1)
        x[:, :, y0:y0 + s, x0:x0 + s] = F.interpolate(x[perm], size=(s, s), mode="bilinear", antialias=True, align_corners=False)
        return 1 - s * s / (h * w), perm
    lam = random.betavariate(1, 1)
    rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
    cy, cx = random.randrange(h), random.randrange(w)
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, h), max(cx - rw // 2, 0), min(cx + rw // 2, w)
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    return 1 - (y1 - y0) * (x1 - x0) / (h * w), perm  # fraction of the original image kept

class Cutout:  # NEW: zero one size x size square at a random centre (may be clipped at the border); DeVries & Taylor 2017
    def __init__(self, size):
        self.size = size
    def __call__(self, x):
        h, w = x.shape[1:]
        cy, cx = random.randrange(h), random.randrange(w)
        x = x.clone()
        x[:, max(cy - self.size // 2, 0):cy + self.size // 2, max(cx - self.size // 2, 0):cx + self.size // 2] = 0
        return x

def sam_perturb(model, rho):  # NEW: SAM step 1 (Foret et al. 2021): move weights rho along the gradient direction
    ps = [p for p in model.parameters() if p.grad is not None]
    norm = torch.norm(torch.stack([p.grad.norm() for p in ps]))  # grads may be AMP-scaled; the direction is scale-free
    if not torch.isfinite(norm) or norm == 0:
        return []  # fp16 overflow step: skip the perturbation, GradScaler will skip the update too
    with torch.no_grad():
        eps = [(p, p.grad * (rho / (norm + 1e-12))) for p in ps]
        for p, e in eps:
            p.add_(e)
    return eps

def read_rows(csv_file, col):
    with open(csv_file) as f:
        return [(r["filename"], int(r[col])) for r in csv.DictReader(f)]

if __name__ == "__main__":
    EPOCHS, BS, LR, N_VAL = int(os.environ.get("EPOCHS", 200)), 128, float(os.environ.get("LR", 0.1)), 5000  # CHANGED: EPOCHS/LR from env (LR=0.02 for fine-tuning)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    SEED, TAG = int(os.environ.get("SEED", 0)), os.environ.get("TAG", "")  # NEW: output files are best{TAG}.pt, submission{TAG}.csv
    C, B = [tuple(map(int, s.split(","))) for s in os.environ.get("ARCH", "64,128,256;3,3,3").split(";")]  # NEW: ARCH="64,128,256;4,4,3"
    FULL, MIX = os.environ.get("FULL") == "1", os.environ.get("MIX") == "1"  # NEW: FULL=1 trains on all 50k, MIX=1 adds CutMix
    TEACHERS = [p for p in os.environ.get("TEACHERS", "").split(",") if p]  # NEW: TEACHERS="a.pt,b.pt" distills from their averaged softmax
    T_KD, A_KD = float(os.environ.get("T_KD", 4.0)), float(os.environ.get("A_KD", 0.8))  # NEW: distillation temperature and weight of the teacher term (Hinton et al. 2015); A_KD=1 = pure function matching (Beyer et al. 2022)
    SMOOTH = float(os.environ.get("SMOOTH", 0.1))  # NEW: label smoothing; 0 for teachers whose soft labels will be distilled
    FOLD, NFOLD = int(os.environ.get("FOLD", -1)), 3  # NEW: FOLD=k holds out fold k of the (shuffled) training set; -1 = none
    WORKERS = int(os.environ.get("WORKERS", 4))  # NEW: data-loader processes; raise on many-core nodes where PNG decoding is the bottleneck
    SAM = float(os.environ.get("SAM", 0))  # NEW: SAM=0.05 enables sharpness-aware minimization (2 forward/backward passes per step)
    SE, DROP_PATH = os.environ.get("SE") == "1", float(os.environ.get("DROP_PATH", 0))  # NEW: squeeze-and-excitation blocks; stochastic depth max rate
    CUTOUT = int(os.environ.get("CUTOUT", 0))  # NEW: CUTOUT=16 replaces the small random erase with a 16x16 Cutout square
    SHAKE, WD = os.environ.get("SHAKE") == "1", float(os.environ.get("WD", 5e-4))  # NEW: SHAKE=1 Shake-Shake blocks (Gastaldi 2017; paper: lr 0.2, WD=1e-4, 1800 epochs); WD weight decay
    ZOOM = float(os.environ.get("ZOOM", 0))  # NEW: ZOOM=0.3 zooms that fraction of training images out to 60-100% size on a mean-colour canvas (small objects)
    STRIDES = tuple(int(v) for v in os.environ.get("STRIDES", "1,2,2").split(","))  # NEW: delayed downsampling, same params
    ZOOM_S = float(os.environ.get("ZOOM_S", 0.6))  # NEW: smallest zoom-out size (0.6 = 60%)
    BLUR_S = float(os.environ.get("BLUR_S", 1.0))  # NEW: largest blur sigma
    BLUR = float(os.environ.get("BLUR", 0))  # NEW: BLUR=0.2 blurs that fraction of training images (Gaussian, sigma 0.3-1.0) so the model tolerates low detail
    GRAY = float(os.environ.get("GRAY", 0))  # NEW: GRAY=0.1 turns that fraction of training images grey, so shape (not colour) separates e.g. truck vs automobile
    KD_LOSS = os.environ.get("KD_LOSS", "kl")  # NEW: distillation loss, "kl" (default) or "mse" on logits (Kim et al. 2021)
    torch.manual_seed(SEED)  # NEW
    random.seed(SEED)  # NEW: CutMix boxes

    model = ResNet(C, B, se=SE, drop_path=DROP_PATH, shake=SHAKE, S=STRIDES).to(dev)  # CHANGED: architecture from ARCH, SE, DROP_PATH, SHAKE
    print(f"params: {n_params(model):,}  arch C={C} B={B} se={SE} drop_path={DROP_PATH} shake={SHAKE} strides={STRIDES} wd={WD}  seed={SEED}  full={FULL}  cutmix={MIX}  sam={SAM} blur={BLUR}/{BLUR_S} zoom={ZOOM}/{ZOOM_S} gray={GRAY} erase={os.environ.get('ERASE', '1')} zoomin={os.environ.get('ZOOMIN', '0')} resizemix={os.environ.get('MIX_RESIZE', '0')}  teachers={TEACHERS}")  # CHANGED
    assert os.environ.get("TEACHER") == "1" or n_params(model) <= 5_000_000, "over the 5M budget"  # NEW: TEACHER=1 lifts the cap for a distillation teacher (never submitted)
    if os.environ.get("INIT"):  # NEW: INIT=best_x.pt fine-tunes from that model's weights (model soups, Wortsman et al. 2022); EMA below starts from them too
        model.load_state_dict(torch.load(os.environ["INIT"], map_location=dev)["sd"])
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)  # NEW: running avg of weights + BN stats
    amp = dev == "cuda"  # NEW: mixed precision only on NVIDIA GPUs (HPC)
    scaler = torch.amp.GradScaler(enabled=amp)  # NEW: rescales loss so fp16 grads don't underflow
    teachers, t_fold = [], []  # NEW: frozen teacher nets, used only during training; t_fold = fold each teacher did NOT see
    for p in TEACHERS:
        ck = torch.load(p, map_location=dev)
        t = from_ckpt(ck).to(dev)  # CHANGED: teachers may have SE or Shake-Shake blocks
        t.load_state_dict(ck["sd"])
        teachers.append(t.eval())
        t_fold.append(ck.get("fold", -1))
    t_fold = torch.tensor(t_fold, device=dev)
    assert not (t_fold >= 0).any() or (t_fold < 0).any() or set(t_fold.tolist()) == set(range(NFOLD)), "fold teachers must cover every fold"  # NEW

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    aug = T.AugMix() if os.environ.get("AUG") == "augmix" else T.TrivialAugmentWide()  # NEW: AUG=augmix (Hendrycks et al. 2020) targets robustness to the test-set shift
    zoom = [T.RandomApply([T.RandomAffine(0, scale=(ZOOM_S, 1.0), fill=(125, 123, 114))], p=ZOOM)] if ZOOM else []  # NEW: only when on, so old recipes keep their random stream
    blur = [T.RandomApply([T.GaussianBlur(5, sigma=(0.3, BLUR_S))], p=BLUR)] if BLUR else []  # NEW: same
    gray = [T.RandomGrayscale(p=GRAY)] if GRAY else []  # NEW: same
    ZOOMIN = float(os.environ.get("ZOOMIN", 0))  # NEW: ZOOMIN=0.3 crops that fraction of images to 50-100% of their area and scales back to 32x32 (close-ups, partial views)
    zoomin = [T.RandomApply([T.RandomResizedCrop(32, scale=(0.5, 1.0), antialias=True)], p=ZOOMIN)] if ZOOMIN else []  # NEW: only when on, so old recipes keep their random stream
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *zoomin, *zoom, aug, *blur, *gray,  # CHANGED: + TrivialAugment / AugMix, optional ZOOM / BLUR
                          T.ToTensor(), T.Normalize(mean, std),
                          *([Cutout(CUTOUT)] if CUTOUT else [] if os.environ.get("ERASE") == "0" else [T.RandomErasing(scale=(0.02, 0.15), value=0)])])  # CHANGED: ERASE=0 turns the small random erase off  # CHANGED: CUTOUT=16 uses a fixed square (DeVries & Taylor 2017) instead of the small random erase
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    rows = read_rows(DATA / "train_labels.csv", "label")
    random.Random(0).shuffle(rows)
    fold_of = {name: i % NFOLD for i, (name, _) in enumerate(rows)}  # NEW: image -> fold (fixed by the seed-0 shuffle)
    train_rows = rows if FULL else rows[N_VAL:]
    if FOLD >= 0:  # NEW: teacher for cross-fold distillation: never sees its own fold
        train_rows = [r for r in train_rows if fold_of[r[0]] != FOLD]
    train = PNGs(train_rows, DATA / "train_images", train_tf, [fold_of[n] for n, _ in train_rows])  # CHANGED: FULL=1 also trains on the val images (val acc is then not held-out)
    val = PNGs(rows[:N_VAL], DATA / "train_images", test_tf)
    test = PNGs(read_rows(DATA / "test.csv", "id"), DATA / "test_images", test_tf)
    train_dl = torch.utils.data.DataLoader(train, BS, shuffle=True, num_workers=WORKERS, persistent_workers=True, pin_memory=True)  # CHANGED: keep workers alive between epochs (Windows)
    val_dl = torch.utils.data.DataLoader(val, 256, num_workers=WORKERS, persistent_workers=True, pin_memory=True)  # CHANGED: keep workers alive between epochs (Windows)
    test_dl = torch.utils.data.DataLoader(test, 256, num_workers=WORKERS, persistent_workers=True, pin_memory=True)  # CHANGED: keep workers alive between epochs (Windows)

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)  # CHANGED: WD from env
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=SMOOTH)  # CHANGED: from env

    best, start = 0, 0
    STATE = f"state{TAG}.pt"  # NEW: full training state, so a long run can resume after an HPC session limit (RESUME=1)
    if os.environ.get("RESUME") == "1" and os.path.exists(STATE):
        st = torch.load(STATE, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"]); start, best = st["epoch"], st["best"]
        print(f"resumed from {STATE} at epoch {start}", flush=True)
    for ep in range(start, EPOCHS):
        model.train()
        for step, (x, y, fold) in enumerate(train_dl, 1):  # CHANGED: count steps for the progress bar; fold = split each image is in
            x, y, fold = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), fold.to(dev)
            lam, perm = 1.0, None
            if MIX and random.random() < 0.5:  # NEW: CutMix on half the batches
                lam, perm = cutmix(x)
            opt.zero_grad()
            autocast = torch.autocast("cuda" if amp else "cpu", enabled=amp)  # NEW: fp16 forward pass on GPU
            soft = None
            if teachers:  # NEW: teachers' soft targets for this (mixed) batch, computed once per step
                with torch.no_grad(), autocast:  # CHANGED: per image, average only teachers that never trained on it (cross-fold KD)
                    logits = torch.stack([t(x).float() for t in teachers])  # (teachers, batch, 10)
                    probs = logits if KD_LOSS == "mse" else (logits / T_KD).softmax(-1)  # NEW: MSE matches mean logits, KL mean probs (softmax over the 10 classes)
                    w = ((t_fold[:, None] == fold[None, :]) | (t_fold[:, None] < 0)).float()  # fold -1 = teacher used everywhere
                    soft = (probs * w[..., None]).sum(0) / w.sum(0)[:, None]  # ponytail: CutMix patch from another image can leak; fine for 3 folds

            def loss_of(out):  # CHANGED: loss as a function of the logits, so SAM can evaluate it twice
                l = loss_fn(out, y) if perm is None else lam * loss_fn(out, y) + (1 - lam) * loss_fn(out, y[perm])  # mixed labels
                if soft is not None:
                    if KD_LOSS == "mse":  # NEW: Kim et al. 2021, MSE between student and mean teacher logits
                        kd = F.mse_loss(out.float(), soft)  # mean over batch x 10 logits; the per-sample sum diverged at lr 0.1
                    else:
                        kd = F.kl_div((out.float() / T_KD).log_softmax(1), soft, reduction="batchmean") * T_KD ** 2
                    l = (1 - A_KD) * l + A_KD * kd
                return l

            with autocast:
                loss = loss_of(model(x))
            scaler.scale(loss).backward()  # CHANGED: was loss.backward()
            if SAM > 0:  # NEW: SAM step 2: gradient at the perturbed weights, applied to the original weights
                eps = sam_perturb(model, SAM)
                opt.zero_grad()
                with autocast:
                    loss = loss_of(model(x))  # ponytail: BN running stats also see this pass; official impls skip that, effect is small
                scaler.scale(loss).backward()
                with torch.no_grad():
                    for p, e in eps:
                        p.sub_(e)
            if KD_LOSS == "mse":  # NEW: MSE gradients grow with the logit error and run away at lr 0.1 without clipping
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)  # CHANGED: was opt.step()
            scaler.update()  # NEW
            ema.update_parameters(model)  # NEW: update EMA after every step
            done = step * 30 // len(train_dl)  # NEW: progress bar, 30 chars wide
            print(f"\rep {ep+1:3d}/{EPOCHS} [{'#' * done}{'.' * (30 - done)}] {step}/{len(train_dl)}  loss {loss.item():.3f}", end="", flush=True)  # NEW
        sched.step()

        ema.eval()  # CHANGED: evaluate the EMA model (BN -> running stats)
        model.eval()  # NEW: also score the raw model (EMA lags in short runs)
        correct = raw = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(dev)
                correct += (predict(ema, x).argmax(1).cpu() == y).sum().item()  # CHANGED: EMA + flip-TTA
                raw += (predict(model, x).argmax(1).cpu() == y).sum().item()  # NEW
        acc = correct / len(val)
        if acc > best or FULL:  # CHANGED: with FULL=1 val is seen in training, so just keep the latest
            best = max(best, acc)
            torch.save({"C": C, "B": B, "se": SE, "shake": SHAKE, "S": STRIDES, "fold": FOLD, "sd": ema.module.state_dict()}, f"best{TAG}.pt")  # CHANGED: EMA weights + arch + held-out fold, so it can be a teacher
        print(f"\rep {ep+1:3d}/{EPOCHS}  loss {loss.item():.3f}  val acc {acc:.4f}  raw {raw / len(val):.4f}  best {best:.4f}" + " " * 30)  # CHANGED: overwrite the bar with the epoch summary
        if os.environ.get("RESUME") == "1":  # NEW: checkpoint the full state for RESUME (only when asked; ~75 MB per run)
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best}, STATE)

    model.load_state_dict(torch.load(f"best{TAG}.pt", map_location=dev)["sd"])  # CHANGED
    model.eval()
    with torch.no_grad(), open(f"submission{TAG}.csv", "w", newline="") as f:  # CHANGED
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for x, ids in test_dl:
            for i, p in zip(ids.tolist(), predict(model, x.to(dev)).argmax(1).tolist()):  # CHANGED: flip-TTA
                w.writerow([i, p])
    print(f"wrote submission{TAG}.csv")  # CHANGED
