"""Evaluation metrics shared across both methods.

The MIA follows the papers' protocol: an attacker is trained ONCE on the
pre-unlearning ("Standard FL") model to separate members from non-members using
per-sample features (loss, max-confidence, entropy, correctness). That fixed
attacker is then applied to every candidate model to score the FORGET client's
training samples (members) vs held-out test samples (non-members). Lower AUC on a
candidate model => the forget client's membership signal is weaker => better
forgetting. Read jointly with test accuracy: a degraded model also lowers AUC.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = torch.nan_to_num(model(x))   # guard against a diverged model
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def backdoor_asr(model, bd_loader, target, device):
    """Fraction of triggered inputs predicted as the target label."""
    model.eval()
    hit = total = 0
    for x, y in bd_loader:
        x = x.to(device)
        pred = torch.nan_to_num(model(x)).argmax(1).cpu()
        hit += (pred == target).sum().item()
        total += pred.numel()
    return hit / max(1, total)


@torch.no_grad()
def _mia_features(model, loader, device):
    model.eval()
    feats = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # clamp guards a diverged (inf/nan) model from producing NaN features
        logits = torch.clamp(torch.nan_to_num(model(x)), -30.0, 30.0)
        p = F.softmax(logits, 1)
        loss = F.cross_entropy(logits, y, reduction="none")
        conf = p.max(1).values
        ent = -(p * (p + 1e-12).log()).sum(1)
        correct = (logits.argmax(1) == y).float()
        f = torch.stack([loss, conf, ent, correct], 1)
        feats.append(f.cpu().numpy())
    return np.nan_to_num(np.concatenate(feats, 0), nan=0.0, posinf=1e6, neginf=-1e6)


def train_mia_attacker(standard_model, member_loader, nonmember_loader, device):
    """Fit the attacker on Standard FL. member=train subset, nonmember=test subset."""
    Xm = _mia_features(standard_model, member_loader, device)
    Xn = _mia_features(standard_model, nonmember_loader, device)
    X = np.concatenate([Xm, Xn], 0)
    y = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xn))])
    # standardize loss/entropy (scale varies across datasets)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    clf = LogisticRegression(max_iter=1000)
    clf.fit((X - mu) / sd, y)
    return dict(clf=clf, mu=mu, sd=sd)


def mia_auc(attacker, model, forget_loader, nonmember_loader, device):
    """Apply the fixed attacker to `model`; AUC over forget-members vs nonmembers."""
    Xm = _mia_features(model, forget_loader, device)
    Xn = _mia_features(model, nonmember_loader, device)
    X = np.concatenate([Xm, Xn], 0)
    y = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xn))])
    Xs = (X - attacker["mu"]) / attacker["sd"]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        scores = attacker["clf"].predict_proba(Xs)[:, 1]
        return roc_auc_score(y, scores)
    except Exception:
        return 0.5    # degenerate model -> attacker gains no signal


def rdp_epsilon(z_bar_per_round, delta=1e-5, lambdas=None):
    """Renyi-DP accounting for AdaptFU (eq. 12).
    Each round is a Gaussian mechanism at scale z_bar_t; compose across rounds.
    eps = min_lambda [ sum_t lambda/(2 z_bar_t^2) - log(delta)/(lambda-1) ]."""
    z = np.asarray([z for z in z_bar_per_round if z > 0], dtype=float)
    if len(z) == 0:
        return float("nan")
    if lambdas is None:
        lambdas = np.concatenate([np.arange(2, 64), np.arange(64, 512, 8)])
    best = np.inf
    for lam in lambdas:
        rdp = np.sum(lam / (2.0 * z ** 2))
        eps = rdp - np.log(delta) / (lam - 1)
        best = min(best, eps)
    return float(best)
