"""FuGuard: client-level unlearning via generative surrogates + optimal transport.

Pipeline (paper Algorithm 1):
  1. Pretrain a VAE (done once, reused).
  2. On a deletion request, encode a small subset of the target client's data,
     fit PCA on the latents, perturb along a principal direction, decode -> proxy
     dataset D_hat that inherits the sampled labels.
  3. Freeze theta_0 (pre-unlearning model). Unlearn by gradient ASCENT on the
     proxy classification loss, regularized by a Sinkhorn OT term that keeps the
     current penultimate embeddings close to theta_0's (limits drift).
  4. Recovery: a few FedAvg rounds on the remaining clients.

Threat model is benign (no adversarial client during unlearning) -- FuGuard has
no robustness mechanism, which is exactly the axis AdaptFU adds.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from .utils import get_vector, set_vector
from .fl_train import federated_train, round_lr


# ----------------------------- VAE pretraining -----------------------------
def pretrain_vae(vae, full_loader, epochs, device, beta=1.0):
    vae = vae.to(device); vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for ep in range(epochs):
        for x, _ in full_loader:
            x = x.to(device)
            xr, mu, lv = vae(x)
            rec = F.mse_loss(xr, x, reduction="sum") / x.size(0)
            kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
            loss = rec + beta * kld
            opt.zero_grad(); loss.backward(); opt.step()
    return vae


# ----------------------- proxy synthesis (PCA in latent) --------------------
@torch.no_grad()
def synthesize_proxy_text(model, client_subset, cfg, device):
    """FuGuard proxy for TEXT.

    The paper's ConvVAE cannot encode token ids, and FuGuard itself notes that FU
    methods are "predominantly evaluated on image datasets ... due to
    incompatibilities in framework design". We therefore keep FuGuard's actual
    mechanism -- PCA-guided perturbation of a latent representation (eq.7-9) --
    but apply it to the classifier's EMBEDDING space rather than VAE latents.
    The returned proxy holds embeddings; TextCNN.forward accepts them directly."""
    fg = cfg.fuguard
    n = max(1, int(fg.proxy_fraction * len(client_subset)))
    idx = np.random.choice(len(client_subset), n,
                           replace=len(client_subset) < n)
    xs, ys = [], []
    for i in idx:
        x, y = client_subset[int(i)]
        xs.append(x); ys.append(y)
    X = torch.stack(xs).to(device)                  # [n, L] token ids
    Y = torch.tensor(ys)
    E = model.emb(X)                                # [n, L, D] latent
    flat = E.reshape(E.size(0), -1)
    c = flat - flat.mean(0, keepdim=True)
    k = min(fg.pca_components, c.shape[0], c.shape[1])
    _, _, V = torch.linalg.svd(c, full_matrices=False)
    r = min(fg.pca_direction, k) - 1
    pert = flat + fg.pca_alpha * V[r]               # eq.8 along principal dir
    return TensorDataset(pert.view_as(E).cpu(), Y)


@torch.no_grad()
def synthesize_proxy(vae, client_subset, cfg, device):
    """Return a TensorDataset of decoded proxy samples inheriting sampled labels."""
    fg = cfg.fuguard
    n = max(1, int(fg.proxy_fraction * len(client_subset)))
    idx = np.random.choice(len(client_subset), n, replace=len(client_subset) < n)
    xs, ys = [], []
    for i in idx:
        x, y = client_subset[int(i)]
        xs.append(x); ys.append(y)
    X = torch.stack(xs).to(device)
    Y = torch.tensor(ys)

    mu, _ = vae.encode(X)                      # posterior mean as latent (eq. 6)
    Z = mu.reshape(mu.size(0), -1)             # flatten
    Zc = Z - Z.mean(0, keepdim=True)
    # PCA via SVD; take K components, perturb along direction r
    k = min(fg.pca_components, Zc.shape[0], Zc.shape[1])
    _, _, V = torch.linalg.svd(Zc, full_matrices=False)
    r = min(fg.pca_direction, k) - 1
    pr = V[r]
    Zt = Z + fg.pca_alpha * pr                 # eq. 8
    Xhat = vae.decode(Zt.view_as(mu))          # decode (eq. 9)
    return TensorDataset(Xhat.cpu(), Y)


# ------------------------------- Sinkhorn OT --------------------------------
def sinkhorn_cost(x, y, eps=0.1, iters=50):
    """Entropic OT cost between empirical measures over rows of x and y (uniform
    weights), squared-euclidean cost. Log-domain, differentiable."""
    C = torch.cdist(x, y) ** 2
    n, m = C.shape
    a = torch.full((n,), 1.0 / n, device=x.device)
    b = torch.full((m,), 1.0 / m, device=x.device)
    lu, lv = torch.zeros(n, device=x.device), torch.zeros(m, device=x.device)
    K = -C / eps
    for _ in range(iters):
        lu = torch.log(a + 1e-12) - torch.logsumexp(K + lv[None, :], dim=1)
        lv = torch.log(b + 1e-12) - torch.logsumexp(K + lu[:, None], dim=0)
    P = torch.exp(lu[:, None] + K + lv[None, :])
    return (P * C).sum()


# ------------------------------- unlearning ---------------------------------
def run_fuguard(base_model, vae, client_loaders, cfg, device, recovery=True,
                logger=None, eval_hook=None, tag="FuGuard"):
    fg = cfg.fuguard
    forget = cfg.forget.forget_clients
    model = copy.deepcopy(base_model).to(device)
    theta0 = copy.deepcopy(base_model).to(device).eval()
    for p in theta0.parameters():
        p.requires_grad_(False)

    # build proxy dataset from all forget clients' local data
    is_text = vae is None                            # text: no ConvVAE available
    proxy_parts = []
    for fc in forget:
        subset = client_loaders[fc].dataset
        proxy_parts.append(
            synthesize_proxy_text(model, subset, cfg, device) if is_text
            else synthesize_proxy(vae, subset, cfg, device))
    proxy = torch.utils.data.ConcatDataset(proxy_parts)
    ploader = DataLoader(proxy, batch_size=cfg.fl.batch_size, shuffle=True)

    opt = torch.optim.SGD(model.parameters(), lr=fg.lr_unlearn, momentum=0.9)
    for ep in range(fg.unlearn_epochs):
        model.train()
        cls_running, ot_running, nb = 0.0, 0.0, 0
        for xh, yh in ploader:
            xh, yh = xh.to(device), yh.to(device)
            logits = model(xh)
            zc = model.features(xh)
            with torch.no_grad():
                zo = theta0.features(xh)
            l_ot = sinkhorn_cost(zc, zo, fg.sinkhorn_eps, fg.sinkhorn_iters)
            l_cls = F.cross_entropy(logits, yh)
            # gradient ASCENT on classification loss => minimize its negative
            loss = -l_cls + fg.lambda_ot * l_ot
            opt.zero_grad(); loss.backward(); opt.step()
            cls_running += float(l_cls); ot_running += float(l_ot); nb += 1
        if logger is not None and nb:
            logger.log("method_epoch", method=tag, epoch=ep,
                       cls_loss=round(cls_running / nb, 4),
                       ot_loss=round(ot_running / nb, 4))
        if eval_hook is not None:
            eval_hook(tag + ":unlearn", ep, model)

    info = dict(proxy_size=len(proxy))
    if recovery and fg.recovery_rounds > 0:
        rec_cfg = copy.deepcopy(cfg.fl)
        rec_cfg.rounds = fg.recovery_rounds
        # recovery continues from a converged model: use the END-of-schedule lr
        # rather than restarting a fresh cosine cycle at the full lr, which would
        # blow the model apart before the few recovery rounds can settle.
        rec_cfg.lr = round_lr(cfg.fl, cfg.fl.rounds - 1)
        rec_cfg.lr_schedule = "constant"
        model, _ = federated_train(model, client_loaders, rec_cfg, device,
                                   exclude=set(forget), cache_history=False,
                                   verbose=False, logger=logger, phase="fuguard_recovery")
        if eval_hook is not None:
            eval_hook(tag, fg.recovery_rounds - 1, model)
    return model, info
