"""COVID-19 detection from chest X-ray / CT.

Reproduces the paper's pipeline and adds improvements:
  paper     : 16-layer CNN (74,018 params) + CNN-feature -> HGB / ERT / GB / DT hybrids
  improved  : CovidNet-Plus (residual + squeeze-excitation, BatchNorm, GAP) trained with augmentation,
              label smoothing, AdamW + OneCycle, flip test-time augmentation
  ensemble  : CovidNet-Plus x HGB-on-embeddings soft vote + temperature scaling (calibration)
  XAI       : Grad-CAM (paper) and Grad-CAM++ (improved)

Class imbalance is handled with class-weighted loss *inside training only* - no synthetic images are
ever created, so nothing leaks into the test fold.

Usage:  python imaging.py --modality xray --profile quick
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import brier_score_loss, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import (CACHE, IMG_ROOT, PROFILES, RESULTS, SEED, SITE, device_name, ensure_dirs, get_device,
                    seed_everything)

DEV = get_device()


# ----------------------------------------------------------------------------- models
class PaperCNN(nn.Module):
    """8 x Conv(32,3x3) + 4 x MaxPool + Dropout + Flatten + Dense(64) + Dense(2)  = 74,018 params."""

    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for _ in range(4):
            for _ in range(2):
                layers += [nn.Conv2d(c, 32, 3), nn.ReLU()]
                c = 32
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        self.cam_layer = self.features[-2]  # last ReLU before the final pooling

    def embed(self, x):
        return torch.flatten(self.drop(self.features(x)), 1)  # 128-d, CNN-as-feature-extractor

    def forward(self, x):
        return self.out(F.relu(self.fc(self.embed(x))))


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.f1, self.f2 = nn.Linear(c, c // r), nn.Linear(c // r, c)

    def forward(self, x):
        s = torch.sigmoid(self.f2(F.relu(self.f1(x.mean((2, 3))))))
        return x * s[:, :, None, None]


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.se = SE(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(self.se(h) + self.skip(x))


class CovidNetPlus(nn.Module):
    """Compact residual + squeeze-excitation network (~250k params, still 16-500x smaller than VGG/DenseNet)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2))
        self.b1, self.b2, self.b3, self.b4 = Block(24, 32), Block(32, 48), Block(48, 64), Block(64, 96)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(96, 2)
        self.cam_layer = self.b4

    def embed(self, x):
        x = self.stem(x)
        x = F.max_pool2d(self.b1(x), 2)
        x = F.max_pool2d(self.b2(x), 2)
        x = F.max_pool2d(self.b3(x), 2)
        return self.b4(x).mean((2, 3))  # 96-d

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- data helpers
def to_tensor(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW


def prep(xb):
    return xb.to(DEV, non_blocking=True).float().div_(255.0)


def augment(x):
    n = x.size(0)
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    ang = (torch.rand(n, device=x.device) - 0.5) * 2 * (10 * np.pi / 180)
    sc = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.2
    tx, ty = [(torch.rand(n, device=x.device) - 0.5) * 0.16 for _ in range(2)]
    theta = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                         torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.3
    c = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.1
    return (x * b + c).clamp(0, 1)


@torch.no_grad()
def predict_logits(model, X, tta=False, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs])
        lg = model(x)
        if tta:
            lg = (lg + model(x.flip(3))) / 2
        out.append(lg.float().cpu())
    return torch.cat(out)


def probs(model, X, tta=False):
    return F.softmax(predict_logits(model, X, tta), 1)[:, 1].numpy()


@torch.no_grad()
def embeddings(model, X, bs=256):
    model.eval()
    return torch.cat([model.embed(prep(X[i:i + bs])).float().cpu() for i in range(0, len(X), bs)]).numpy()


def train_model(model, Xtr, ytr, Xva, yva, epochs, lr, batch, patience, plus, log):
    model.to(DEV)
    cw = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32, device=DEV)
    ytr_t, n = torch.from_numpy(ytr).long(), len(ytr)
    steps = int(np.ceil(n / batch))
    if plus:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.25)
        ls = 0.05
    else:  # paper: plain Adam, constant learning rate
        opt, sched, ls = torch.optim.Adam(model.parameters(), lr=lr), None, 0.0
    use_amp = DEV.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    hist, best, best_state, bad = [], 1e9, None, 0
    for ep in range(epochs):
        model.train()
        t0, perm, tl, tc = time.time(), torch.randperm(n), 0.0, 0
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            x, y = prep(Xtr[idx]), ytr_t[idx].to(DEV)
            if plus:
                x = augment(x)
            with torch.autocast(DEV.type, enabled=use_amp):
                lg = model(x)
                loss = F.cross_entropy(lg, y, weight=cw, label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            tl += loss.item() * len(idx)
            tc += (lg.argmax(1) == y).sum().item()
        vl = predict_logits(model, Xva)
        vloss = F.cross_entropy(vl, torch.from_numpy(yva).long(), weight=cw.cpu()).item()
        vacc = (vl.argmax(1).numpy() == yva).mean()
        hist.append(dict(epoch=ep + 1, loss=tl / n, acc=tc / n, val_loss=vloss, val_acc=float(vacc)))
        log(f"  ep {ep + 1:3d}/{epochs} loss {tl / n:.4f} acc {tc / n:.4f} | val {vloss:.4f} {vacc:.4f} | {time.time() - t0:.0f}s")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("  early stop")
                break
    model.load_state_dict(best_state)
    return hist


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1)
    prec_macro = (prec + tn / max(tn + fn, 1)) / 2  # the paper's precision/FDR columns are two-class macro averages
    return dict(accuracy=100 * (tp + tn) / len(y), sensitivity=100 * sens, specificity=100 * spec, precision=100 * prec,
                balanced_accuracy=100 * (sens + spec) / 2, precision_macro=100 * prec_macro,
                fdr=100 * (1 - prec), f1=100 * 2 * prec * sens / max(prec + sens, 1e-9), mcc=float(matthews_corrcoef(y, pred)),
                auc=float(roc_auc_score(y, p)), ece=ece(y, p), brier=float(brier_score_loss(y, p)),
                cm=[[int(tn), int(fp)], [int(fn), int(tp)]])


def roc_pts(y, p, k=70):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, k).astype(int))
    return dict(fpr=np.round(fpr[idx], 4).tolist(), tpr=np.round(tpr[idx], 4).tolist())


@torch.no_grad()
def ms_per_image(model, X, n=150):
    model.eval()
    x = prep(X[:n])
    for i in range(10):
        model(x[i:i + 1])
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for i in range(n):
        model(x[i:i + 1])
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) * 1000 / n


def fit_temperature(logit, y):
    t = torch.ones(1, requires_grad=True)
    lg, yt = torch.from_numpy(logit).float(), torch.from_numpy(y).float()
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(lg / t.clamp(min=0.05), yt)
        l.backward()
        return l

    opt.step(closure)
    return float(t.clamp(min=0.05))


# ----------------------------------------------------------------------------- explainability
def grad_cam(model, x, cls, plusplus=False):
    """x: (1,3,H,W) float. Returns (H,W) map in [0,1]."""
    acts, grads = {}, {}
    def fwd(m, i, o):
        acts["a"] = o

    def bwd(m, gi, go):
        grads["g"] = go[0]

    h1 = model.cam_layer.register_forward_hook(fwd)
    h2 = model.cam_layer.register_full_backward_hook(bwd)
    model.eval()
    out = model(x.requires_grad_(True))
    model.zero_grad()
    out[0, cls].backward()
    h1.remove(); h2.remove()
    a, g = acts["a"].detach(), grads["g"].detach()
    if plusplus:
        g2, g3 = g ** 2, g ** 3
        alpha = g2 / (2 * g2 + a.sum((2, 3), keepdim=True) * g3 + 1e-8)
        w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
    else:
        w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).cpu().numpy()


def overlay(img, cam, alpha=0.5):
    import matplotlib
    heat = (matplotlib.colormaps["jet"](np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)) / 255.0)[..., :3] * 255)
    return Image.fromarray((np.asarray(img, dtype=float) * (1 - alpha) + heat * alpha).astype(np.uint8))


# ----------------------------------------------------------------------------- main
def smoteenn_images(X, y, log):
    """SMOTE-ENN on flattened 100x100x3 images, as in the base paper (Section 2.4). Returns uint8 images."""
    from imblearn.combine import SMOTEENN
    Xr, yr = SMOTEENN(random_state=SEED).fit_resample(X.reshape(len(X), -1).astype(np.float32), y)
    log(f"  SMOTE-ENN: {np.bincount(y).tolist()} -> {np.bincount(yr).tolist()}  (paper X-ray: [5500, 4044] -> [4186, 4960])")
    return np.clip(np.rint(Xr), 0, 255).astype(np.uint8).reshape(-1, *X.shape[1:]), yr


def run(modality, profile, folds, out_prefix, balance="none"):
    """balance: 'none' = class-weighted loss only (default);
    'smoteenn' = SMOTE-ENN on each training split only (no leakage);
    'smoteenn-all' = SMOTE-ENN on the whole dataset before CV, exactly as published (synthetic images can reach the test fold)."""
    P = PROFILES[profile]
    log = lambda s: print(s, flush=True)
    d = np.load(CACHE / f"{modality}_100.npz")
    X, y, names = d["X"], d["y"], d["names"]
    log(f"{modality}: {X.shape}, COVID={int(y.sum())}, non-COVID={int((1 - y).sum())} | device {device_name()} | balance {balance}")
    if balance == "smoteenn-all":
        X, y = smoteenn_images(X, y, log)
        names = None
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
    fold_res = []
    for fold, (tr_all, te) in enumerate(skf.split(X, y)):
        if fold >= folds:
            break
        seed_everything(SEED + fold)
        tr, va = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all], random_state=SEED)
        Xtr_np, ytr = X[tr], y[tr]
        if balance == "smoteenn":
            Xtr_np, ytr = smoteenn_images(Xtr_np, ytr, log)
        Xtr, Xva, Xte = to_tensor(Xtr_np), to_tensor(X[va]), to_tensor(X[te])
        yva, yte = y[va], y[te]
        R = dict(fold=fold, n_train=len(tr), n_val=len(va), n_test=len(te), models={})

        log(f"[fold {fold}] paper CNN")
        paper = PaperCNN()
        R["paper_params"] = n_params(paper)
        h_paper = train_model(paper, Xtr, ytr, Xva, yva, P["paper_epochs"], P["paper_lr"], P["paper_batch"], P["patience"], False, log)
        p_paper = probs(paper, Xte)
        R["models"]["Paper CNN"] = metrics(yte, p_paper) | dict(ms=ms_per_image(paper, Xte), roc=roc_pts(yte, p_paper))

        log("  hybrid ML on paper-CNN features")
        Etr, Eva, Ete = embeddings(paper, Xtr), embeddings(paper, Xva), embeddings(paper, Xte)
        hybrids = {"HGB": HistGradientBoostingClassifier(random_state=SEED), "ERT": ExtraTreesClassifier(300, n_jobs=-1, random_state=SEED),
                   "GB": GradientBoostingClassifier(random_state=SEED), "DT": DecisionTreeClassifier(random_state=SEED)}
        for k, clf in hybrids.items():
            clf.fit(Etr, ytr)
            t = time.time()
            pk = clf.predict_proba(Ete)[:, 1]
            R["models"][f"CNN+{k}"] = metrics(yte, pk) | dict(ms=ms_per_image(paper, Xte) + (time.time() - t) * 1000 / len(Ete), roc=roc_pts(yte, pk))

        log(f"[fold {fold}] CovidNet-Plus")
        plus = CovidNetPlus()
        R["plus_params"] = n_params(plus)
        h_plus = train_model(plus, Xtr, ytr, Xva, yva, P["plus_epochs"], 3e-3, P["plus_batch"], P["patience"], True, log)
        p_plus_raw = probs(plus, Xte)
        p_plus = probs(plus, Xte, tta=True)
        R["models"]["CovidNet-Plus"] = metrics(yte, p_plus_raw) | dict(ms=ms_per_image(plus, Xte), roc=roc_pts(yte, p_plus_raw))
        R["models"]["CovidNet-Plus + TTA"] = metrics(yte, p_plus) | dict(ms=2 * ms_per_image(plus, Xte), roc=roc_pts(yte, p_plus))

        log("  ensemble + temperature scaling")
        Ptr, Pva = embeddings(plus, Xtr), embeddings(plus, Xva)
        hgb = HistGradientBoostingClassifier(random_state=SEED).fit(Ptr, ytr)
        v_cnn, v_hgb = probs(plus, Xva, True), hgb.predict_proba(Pva)[:, 1]
        t_cnn, t_hgb = p_plus, hgb.predict_proba(embeddings(plus, Xte))[:, 1]
        eps = 1e-6
        logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        best_w = min(np.linspace(0, 1, 11), key=lambda w: -np.mean(yva * np.log(np.clip(w * v_cnn + (1 - w) * v_hgb, eps, 1)) +
                                                                  (1 - yva) * np.log(np.clip(1 - (w * v_cnn + (1 - w) * v_hgb), eps, 1))))
        v_ens, t_ens = best_w * v_cnn + (1 - best_w) * v_hgb, best_w * t_cnn + (1 - best_w) * t_hgb
        T = fit_temperature(logit(v_ens), yva)
        p_final = 1 / (1 + np.exp(-logit(t_ens) / T))
        R["ensemble"] = dict(w_cnn=float(best_w), temperature=T)
        R["models"]["Plus Ensemble (calibrated)"] = metrics(yte, p_final) | dict(ms=2 * ms_per_image(plus, Xte), roc=roc_pts(yte, p_final))
        R["history"] = dict(paper=h_paper, plus=h_plus)

        # ablation of calibration: ECE before / after
        R["calibration"] = dict(before=ece(yte, t_ens), after=ece(yte, p_final), temperature=T)

        for k, v in R["models"].items():
            log(f"  {k:30s} acc {v['accuracy']:.2f} sens {v['sensitivity']:.2f} spec {v['specificity']:.2f} auc {v['auc']:.4f} ece {v['ece']:.4f}")
        fold_res.append(R)

        if fold == 0 and balance == "none":  # gallery and saved models come from the default run only
            gallery(modality, X, y, names, te, Xte, yte, paper, plus, p_paper, p_plus, p_final)
            (RESULTS / "models").mkdir(exist_ok=True)
            torch.save(paper.cpu().state_dict(), RESULTS / "models" / f"{modality}_paper_cnn.pt")
            torch.save(plus.cpu().state_dict(), RESULTS / "models" / f"{modality}_covidnet_plus.pt")

    # aggregate across folds
    agg = {}
    for name in fold_res[0]["models"]:
        agg[name] = {}
        for key in fold_res[0]["models"][name]:
            if key in ("cm", "roc"):
                agg[name][key] = fold_res[0]["models"][name][key]
            else:
                vals = [f["models"][name][key] for f in fold_res]
                agg[name][key] = float(np.mean(vals))
                agg[name][key + "_std"] = float(np.std(vals))
    out = dict(modality=modality, profile=profile, folds=len(fold_res), device=device_name(), n=int(len(y)),
               covid=int(y.sum()), non_covid=int((1 - y).sum()), paper_params=fold_res[0]["paper_params"],
               plus_params=fold_res[0]["plus_params"], models=agg, history=fold_res[0]["history"],
               ensemble=fold_res[0]["ensemble"], calibration=fold_res[0]["calibration"], gallery="gallery_%s.json" % modality,
               balance=balance)
    (RESULTS / f"{out_prefix}.json").write_text(json.dumps(out))
    log(f"saved results/{out_prefix}.json")


def gallery(modality, X, y, names, te, Xte, yte, paper, plus, p_paper, p_plus, p_final, per_class=3):
    rng = np.random.RandomState(SEED)
    gdir = SITE / "assets" / "gradcam"
    gdir.mkdir(parents=True, exist_ok=True)
    pred = (p_final >= 0.5).astype(int)
    pick = []
    for cls in (1, 0):
        ok = np.where((yte == cls) & (pred == cls))[0]
        pick += list(rng.choice(ok, size=per_class, replace=False))
    wrong = np.where(pred != yte)[0]  # a deliberate failure case, shown honestly
    if len(wrong):
        pick.append(int(rng.choice(wrong)))
    items = []
    for j, i in enumerate(pick):
        img = Image.open(IMG_ROOT / names[te[i]]).convert("RGB").resize((256, 256), Image.BICUBIC)
        x = prep(Xte[i:i + 1])
        c_paper = grad_cam(paper.to(DEV), x.clone(), int(p_paper[i] >= 0.5), False)
        c_plus = grad_cam(plus.to(DEV), x.clone(), int(p_plus[i] >= 0.5), True)
        tag = f"{modality}_{j}"
        img.save(gdir / f"{tag}_orig.jpg", quality=88)
        overlay(img, c_paper).save(gdir / f"{tag}_paper.jpg", quality=88)
        overlay(img, c_plus).save(gdir / f"{tag}_plus.jpg", quality=88)
        items.append(dict(id=tag, truth="COVID" if yte[i] else "Non-COVID", p_paper=float(p_paper[i]), p_plus=float(p_plus[i]),
                          p_ens=float(p_final[i]), correct=bool(pred[i] == yte[i])))
    (RESULTS / f"gallery_{modality}.json").write_text(json.dumps(items))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["xray", "ct", "x-ray"], required=True)
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--folds", type=int, default=None, help="number of the 10 CV folds to run (default: profile)")
    ap.add_argument("--epochs-paper", type=int, default=None)
    ap.add_argument("--epochs-plus", type=int, default=None)
    ap.add_argument("--balance", choices=["none", "smoteenn", "smoteenn-all"], default="none",
                    help="none: class-weighted loss (default). smoteenn: SMOTE-ENN on training splits only. "
                         "smoteenn-all: SMOTE-ENN on the whole dataset before CV, as published. Non-default runs save to imaging_<mod>_<balance>.json")
    a = ap.parse_args()
    ensure_dirs()
    if a.epochs_paper:
        PROFILES[a.profile]["paper_epochs"] = a.epochs_paper
    if a.epochs_plus:
        PROFILES[a.profile]["plus_epochs"] = a.epochs_plus
    mod = "x-ray" if a.modality in ("xray", "x-ray") else "ct"
    if a.profile == "paper" and a.balance != "none":
        raise SystemExit("--profile paper already applies SMOTE-ENN to the whole dataset; use --balance none with it")
    prefix = f"imaging_{'xray' if mod == 'x-ray' else 'ct'}" + ("" if a.balance == "none" else f"_{a.balance}")
    run(mod, a.profile, a.folds or PROFILES[a.profile]["folds"], prefix, a.balance)
