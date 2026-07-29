"""AdaptFU: Adaptive-DP Federated Unlearning (reimplementation of Algorithm 1).

Reconstruction warm-starts at G_s and, each round, re-engages retained clients
for short live calibration. From the *stored history* it derives per-client
signals (sensitivity rho, alignment beta) that drive (a) adaptive per-client DP
noise and (b) a round-local divergence-bound verification applied to the RAW
update before clipping. Survivors are DP-noised, clipped, and alpha-blended into
the running reconstruction.

NOTE: this is a faithful reimplementation from the paper. If you re-upload your
notebook, swap `client_calibration_update` / signal computation for your own
functions -- the interfaces here are intentionally small.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn

from .utils import get_vector, set_vector, param_mask, recompute_bn
from .aggregation import aggregate
from .fl_train import round_lr
from .attacks import apply_attack


def _cosine(a, b, eps0=1e-8):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps0))


def calibrate_constants(hist, forget, warm_frac, q_low, q_high, pmask=None):
    """One pass over stored history (rounds t >= s) -> rho_min, rho_max, median step.
    Norms are taken over parameter coordinates only (pmask) so BatchNorm running
    stats -- which dominate the raw vector norm ~300x -- don't distort the
    sensitivity calibration or the divergence bound."""
    T = len(hist["G"]) - 1
    s = int(warm_frac * T)

    def pn(v):
        return (v[pmask] if pmask is not None else v).norm().item()

    norms = []
    for t in range(s, T):
        if t not in hist["C"]:
            continue
        for k, c in hist["C"][t].items():
            if k in forget:
                continue
            norms.append(pn(c))
    norms = np.array(norms) if norms else np.array([1.0])
    rho_min = float(np.percentile(norms, q_low))
    rho_max = float(np.percentile(norms, q_high))
    steps = [pn(hist["G"][t + 1] - hist["G"][t]) for t in range(s, T)]
    m = float(np.median(steps)) if steps else 1.0
    return rho_min, rho_max, m, s


def reliability(rho, beta, rho_min, rho_max):
    align = np.clip((beta + 1) / 2, 0, 1)
    inv_sens = 1 - np.clip((rho - rho_min) / (rho_max - rho_min + 1e-12), 0, 1)
    return float(align * inv_sens)


def client_calibration_update(global_vec, base_model, loader, epochs, lr, device):
    """Live calibration: train E_c epochs from current reconstruction; return update."""
    local = copy.deepcopy(base_model).to(device)
    set_vector(local, global_vec)
    local.train()
    opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(local(x), y).backward(); opt.step()
    return (get_vector(local) - global_vec).cpu()


def _round_z(a, R, hist, t, pmask, rho_min, rho_max):
    """Per-client noise multipliers for one round, honouring rho_scope and dp_mode."""
    Cbar_p = torch.stack([hist["C"][t][k] for k in R]).mean(0)[pmask]
    rn = [float(hist["C"][t][k][pmask].norm()) for k in R]
    if getattr(a, "rho_scope", "pooled") == "round" and len(rn) > 1:
        lo, hi = float(np.percentile(rn, a.q_low)), float(np.percentile(rn, a.q_high))
        if hi - lo < 1e-12:                       # degenerate round: fall back
            lo, hi = rho_min, rho_max
    else:
        lo, hi = rho_min, rho_max
    z = {}
    for k, rho in zip(R, rn):
        beta = _cosine(hist["C"][t][k][pmask], Cbar_p)
        r = reliability(rho, beta, lo, hi)
        z[k] = a.z_max - (a.z_max - a.z_min) * r          # eq. 6
    rel = {}
    for k, rho in zip(R, rn):
        beta = _cosine(hist["C"][t][k][pmask], Cbar_p)
        rel[k] = reliability(rho, beta, lo, hi)
    # canonical names make the axis of adaptivity explicit; old names still work
    mode = {"adaptive": "per_client", "uniform": "per_round"}.get(
        getattr(a, "dp_mode", "per_client"), getattr(a, "dp_mode", "per_client"))
    if mode == "per_round":          # FedADP's shape: z moves ACROSS rounds but is
                                     # the same for every client WITHIN a round.
                                     # Same round-mean => same epsilon as per_client.
        zb = float(np.mean(list(z.values())))
        z = {k: zb for k in R}
    elif mode == "static":
        z = {k: 0.5 * (a.z_min + a.z_max) for k in R}
    return z, rel


def predict_z_bars(hist, forget, cfg, pmask):
    """Per-round mean noise multipliers, computed from stored history ALONE.

    The reliability score depends only on cached update norms/alignments, not on
    any training, so AdaptFU's whole DP budget is predictable before a single
    round runs. That makes matched-budget comparison possible."""
    a = cfg.adaptfu
    rho_min, rho_max, m, s = calibrate_constants(hist, forget, a.warm_start_frac,
                                                 a.q_low, a.q_high, pmask=pmask)
    T = len(hist["G"]) - 1
    z_bars = []
    for t in range(s, T):
        R = [k for k in (hist["C"].get(t, {}) or {}) if k not in forget]
        if not R:
            continue
        z, _ = _round_z(a, R, hist, t, pmask, rho_min, rho_max)
        z_bars.append(float(np.mean(list(z.values()))))
    return z_bars


def scale_z_for_target_eps(hist, forget, cfg, pmask, target_eps):
    """Scale z_min/z_max so AdaptFU's composed epsilon hits `target_eps`.

    Scaling both bounds by c scales every z_k by c exactly. epsilon(c) is
    monotonically DECREASING in c, but it is NOT exactly proportional to 1/c^2:
    the optimal Renyi order shifts with the noise scale, so a closed-form
    sqrt(eps/target) undershoots. We bisect on log(c) instead, which is exact to
    within the accountant's own lambda grid. Returns (c, eps_before).

    This is what makes 'per-client vs uniform noise AT MATCHED BUDGET' -- the
    paper's actual claim -- testable rather than confounded by config defaults."""
    from .metrics import rdp_epsilon
    z_bars = predict_z_bars(hist, forget, cfg, pmask)
    delta = cfg.adaptfu.delta
    eps_now = rdp_epsilon(z_bars, delta)
    if not np.isfinite(eps_now) or target_eps <= 0 or not z_bars:
        return 1.0, eps_now

    def eps_at(c):
        return rdp_epsilon([z * c for z in z_bars], delta)

    lo, hi = 1e-3, 1e3               # eps_at(lo) huge, eps_at(hi) tiny
    for _ in range(80):
        mid = math.sqrt(lo * hi)     # bisect in log space
        if eps_at(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    c = math.sqrt(lo * hi)
    cfg.adaptfu.z_min *= c
    cfg.adaptfu.z_max *= c
    return c, eps_now


def run_adaptfu(base_model, hist, client_loaders, cfg, device, attack_cfg=None,
                logger=None, eval_hook=None, tag="AdaptFU"):
    """Returns (unlearned_model, info). info includes per-round z_bar for DP accounting
    and verification survivor counts."""
    a = cfg.adaptfu
    forget = set(cfg.forget.forget_clients)
    T = len(hist["G"]) - 1
    pmask = param_mask(base_model).to(device)
    rho_min, rho_max, m, s = calibrate_constants(hist, forget, a.warm_start_frac,
                                                 a.q_low, a.q_high, pmask=pmask.cpu())
    b = a.b_scale * m                                   # divergence bound (eq. 7)

    G = hist["G"][s].clone().to(device)                 # warm-start G_hat <- G_s
    model = copy.deepcopy(base_model).to(device)
    set_vector(model, G)

    sizes = hist["sizes"]
    z_bars, survivors_log = [], []

    n_param = int(pmask.sum())

    def pnorm(v):                                       # norm over params only
        return v[pmask].norm()

    for t in range(s, T):
        R = [k for k in (hist["C"].get(t, {}) or {}) if k not in forget]
        if not R:
            continue
        alpha = a.alpha0 + (a.alphaT - a.alpha0) * (t - s) / max(1, T - s - 1)

        # per-client noise multipliers from stored-history signals (params only)
        z_k, rel_k = _round_z(a, R, hist, t, pmask.cpu(), rho_min, rho_max)
        zvals = [z_k[k] for k in R]
        z_bars.append(float(np.mean(zvals)))

        # live calibration updates (raw)
        raw = [client_calibration_update(G, base_model, client_loaders[k],
                                         a.calib_epochs, round_lr(cfg.fl, t, total=T),
                                         device) for k in R]

        # inject poisoning on the raw calibration updates (unlearning-phase attack)
        n_byz = 0
        if attack_cfg and attack_cfg.enabled:
            n_mal = int(round(attack_cfg.malicious_fraction * len(R)))
            mal_idx = list(range(n_mal))               # first n_mal retained clients
            raw = apply_attack(attack_cfg.kind, raw, mal_idx, attack_cfg.noise_scale)
            n_byz = n_mal

        # all norms below are over PARAMETER coords (BN stats excluded)
        norms = torch.tensor([float(pnorm(u.to(device))) for u in raw])

        # ----- verification (round-local robust threshold, on RAW update) -----
        if a.verify_before_clip:
            med = norms.median()
            mad = (norms - med).abs().median() * 1.4826
            thr = med + a.kappa * mad
            keep = [i for i in range(len(R)) if norms[i] <= thr + 1e-9]
        else:
            keep = list(range(len(R)))                  # ablation: skip pre-clip check

        # DP noise + clip on survivors (eq. 9), restricted to parameter coords.
        # BN running-stat coords ride along unscaled so the reconstruction still
        # tracks the correct normalization statistics.
        proc, w = [], []
        for i in keep:
            u = raw[i].to(device).clone()
            u[~pmask] = 0.0                                        # BN via recompute
            if getattr(a, "clip_order", "after") == "before":
                p0 = pnorm(u)                      # standard DP order: clip, then noise
                if p0 > b:
                    u[pmask] = u[pmask] * (b / p0)
            sigma = z_k[R[i]] * b
            if getattr(a, "noise_mode", "coord") == "norm":
                # calibrate the noise MAGNITUDE (E||noise||=z*b) instead of the
                # per-coordinate std; otherwise Pi_b cancels z entirely (see
                # AdaptFUConfig.noise_mode).
                sigma = sigma / math.sqrt(float(n_param))
            noise = torch.randn_like(u); noise[~pmask] = 0.0
            u = u + noise * sigma                                  # add DP noise (params)
            if getattr(a, "clip_order", "after") == "after":
                pn_u = pnorm(u)
                if pn_u > b:
                    u[pmask] = u[pmask] * (b / pn_u)               # eq.9: clip AFTER
            wt = sizes[R[i]]
            if getattr(a, "agg_weight", "samples") == "reliability":
                wt = wt * max(rel_k[R[i]], 1e-3)                   # down-weight outliers
            proc.append(u.cpu()); w.append(wt)

        # if the (wrong) post-clip ordering is requested, verify AFTER clipping
        if not a.verify_before_clip and proc:
            pn = torch.tensor([float(pnorm(u.to(device))) for u in proc])
            med = pn.median(); mad = (pn - med).abs().median() * 1.4826
            thr = med + a.kappa * mad
            keep2 = [j for j in range(len(proc)) if pn[j] <= thr + 1e-9]
            proc = [proc[j] for j in keep2]; w = [w[j] for j in keep2]

        survivors_log.append((len(proc), len(R)))
        if logger is not None:
            logger.log("method_round", method=tag, round=t,
                       alpha=round(float(alpha), 4), bound=round(float(b), 4),
                       survivors_kept=len(proc), survivors_total=len(R),
                       z_bar=round(z_bars[-1], 4),
                       z_min_r=round(float(min(zvals)), 4),
                       z_max_r=round(float(max(zvals)), 4),
                       z_std=round(float(np.std(zvals)), 4),
                       rho_scope=getattr(a, "rho_scope", "pooled"),
                       dp_mode=getattr(a, "dp_mode", "adaptive"),
                       )
        if not proc:
            continue

        U = torch.stack(proc).to(device)
        W = torch.tensor(w, dtype=torch.float32, device=device)
        agg = aggregate(a.aggregation, U, W, n_byz=n_byz)
        G = (1 - alpha) * G + alpha * (G + agg)         # blend survivors into recon
        # (agg is a delta from the current reconstruction, hence G+agg)
        set_vector(model, G)
        recompute_bn(model, client_loaders, R, device)  # data-consistent BN stats
        G = get_vector(model)
        if eval_hook is not None:
            eval_hook(tag, t, model)

    set_vector(model, G)
    recompute_bn(model, client_loaders,
                 [k for k in hist["sizes"] if k not in forget], device)
    info = dict(z_bar=z_bars, survivors=survivors_log, bound=b, warm_start=s,
                rho_min=rho_min, rho_max=rho_max, median_step=m)
    return model, info
