"""Reconstruction-based unlearning baselines that share the harness interface.

Each function takes (base_model, hist, client_loaders, cfg, device) and returns
(unlearned_model, info), exactly like run_adaptfu / run_fuguard, so the
orchestrator can drop them into the same metric panel.

Convention reminder: the cached history stores per-round deltas
  C[t][k] = w_k^t - G_t          (the change client k contributed at round t)
and FedAvg reconstructs by ADDING the mean delta: G_{t+1} = G_t + mean_k C[t][k].
All three methods below express their update in this same delta-to-add form, so
the model-update sign/scale conventions of the original papers are absorbed
consistently (a gradient g maps to our delta = -eta*g).

  FedEraser  (Liu et al. 2021)  : norm-of-stored-update x direction-of-fresh-update
  FedRecover (Cao et al. 2023)  : L-BFGS Hessian-vector estimation + exact rounds
  FedADP     (Jiang et al. 2024): direction-consistency calibration + round DP
"""
import copy
import math
import numpy as np
import torch

from .utils import get_vector, set_vector, param_mask, recompute_bn
from .adaptfu import client_calibration_update, calibrate_constants
from .fl_train import round_lr


# ----------------------------- FedEraser -----------------------------
def run_federaser(base_model, hist, client_loaders, cfg, device,
                  logger=None, eval_hook=None, tag="FedEraser"):
    fe = cfg.federaser
    forget = set(cfg.forget.forget_clients)
    T = len(hist["G"]) - 1
    dt = max(1, fe.retain_interval)
    cali_epochs = max(1, round(fe.calibration_ratio * cfg.fl.local_epochs))
    rounds = [t for t in range(0, T, dt) if t in hist["C"]]

    model = copy.deepcopy(base_model).to(device)
    G = hist["G"][0].clone().to(device)     # start from init (no target influence)
    set_vector(model, G)
    sizes = hist["sizes"]
    pmask = param_mask(base_model).to(device)   # params True, BN stats False

    for j, t in enumerate(rounds):
        R = [k for k in hist["C"][t] if k not in forget]
        if not R:
            continue
        deltas, w = [], []
        if j == 0:
            # first reconstruction round: use stored retained updates directly
            for k in R:
                deltas.append(hist["C"][t][k].to(device)); w.append(sizes[k])
        else:
            for k in R:
                u_hat = client_calibration_update(G, base_model, client_loaders[k],
                                                  cali_epochs, round_lr(cfg.fl, t, total=T),
                                                  device).to(device)
                orig = hist["C"][t][k].to(device)
                # Rescale the PARAMETER part only (eq. 1). BN running-stat coords
                # are NOT moved by delta (that collapses eval); they are recomputed
                # from data below. Zero them in the update.
                cal = torch.zeros_like(orig)
                on = orig[pmask].norm()
                un = u_hat[pmask].norm() + 1e-12
                cal[pmask] = on * (u_hat[pmask] / un)
                deltas.append(cal); w.append(sizes[k])
        W = torch.tensor(w, dtype=torch.float32, device=device); W = W / W.sum()
        agg = (torch.stack(deltas) * W[:, None]).sum(0)
        G = G + agg                              # eq. 3
        set_vector(model, G)
        recompute_bn(model, client_loaders, R, device)   # data-consistent BN stats
        G = get_vector(model)                            # sync recomputed BN into G
        if logger is not None:
            logger.log("method_round", method=tag, round=t, retained=len(R),
                       calibrated=(j > 0))
        if eval_hook is not None:
            eval_hook(tag, t, model)

    set_vector(model, G)
    all_retained = [k for k in sizes if k not in forget]
    recompute_bn(model, client_loaders, all_retained, device)
    return model, dict(retained_rounds=len(rounds), cali_epochs=cali_epochs)


# ----------------------------- FedRecover -----------------------------
def _lbfgs_hvp(dW, dG, v):
    """Approximate Hessian-vector product (FedRecover Algorithm 2).
    dW, dG: (d, s) with columns = past global-model / model-update differences."""
    A = dW.T @ dG                                # (s, s)
    D = torch.diag(torch.diag(A))
    Lm = torch.tril(A, -1)                       # strictly lower triangular
    denom = (dW[:, -1] @ dW[:, -1]) + 1e-12
    sig = (dG[:, -1] @ dW[:, -1]) / denom
    top = torch.cat([-D, Lm.T], dim=1)
    bot = torch.cat([Lm, sig * (dW.T @ dW)], dim=1)
    M = torch.cat([top, bot], dim=0)             # (2s, 2s)
    rhs = torch.cat([dG.T @ v, sig * (dW.T @ v)])
    p = torch.linalg.solve(M, rhs)
    return sig * v - torch.cat([dG, sig * dW], dim=1) @ p


def _choose_tau(hist, forget, tolerance, scale, bins=100000):
    """Abnormality threshold: (1 - tolerance) coordinate quantile of stored
    retained deltas (paper picks tau so <= alpha fraction of coords exceed it).

    torch.quantile caps input at ~16M elements, but the pooled coordinate count
    here is billions (n_clients x n_rounds x n_params). We instead stream a
    histogram of |coord| over all retained deltas and read the quantile off the
    cumulative counts -- O(total elements) time, O(bins) memory, no giant concat.
    """
    # pass 1: global max magnitude (for histogram range)
    hi, total = 0.0, 0
    for t, cd in hist["C"].items():
        for k, c in cd.items():
            if k in forget:
                continue
            hi = max(hi, float(c.abs().max()))
            total += c.numel()
    if total == 0 or hi == 0.0:
        return float("inf")
    # pass 2: accumulate histogram of magnitudes in [0, hi]
    counts = torch.zeros(bins, dtype=torch.float64)
    for t, cd in hist["C"].items():
        for k, c in cd.items():
            if k in forget:
                continue
            counts += torch.histc(c.abs().float(), bins=bins, min=0.0,
                                   max=hi).double()
    # smallest bin edge whose cumulative fraction >= (1 - tolerance)
    cdf = torch.cumsum(counts, 0) / counts.sum()
    idx = int(torch.searchsorted(cdf, torch.tensor(1.0 - tolerance,
                                                    dtype=torch.float64)))
    idx = min(idx, bins - 1)
    q = (idx + 1) / bins * hi          # right edge of that bin
    return float(q) * scale


def run_fedrecover(base_model, hist, client_loaders, cfg, device,
                   logger=None, eval_hook=None, tag="FedRecover"):
    fr = cfg.fedrecover
    forget = set(cfg.forget.forget_clients)
    T = len(hist["G"]) - 1
    s = fr.buffer_size
    Tw = min(max(s + 1, fr.warmup), T - 1)
    Tc = max(1, fr.correction)
    Tf = min(fr.final_tuning, max(0, T - Tw - 1))
    tau = _choose_tau(hist, forget, fr.tolerance, fr.tau_scale)
    R = [k for k in hist["sizes"] if k not in forget]
    sizes = hist["sizes"]

    model = copy.deepcopy(base_model).to(device)
    w_hat = hist["G"][0].clone().to(device)      # re-initialize at original init
    set_vector(model, w_hat)

    # L-BFGS buffers live on CPU and keep only the last s entries (only dW[-s:]
    # and dGi[k][-s:] are ever used). Storing all exact rounds x all clients of
    # full 11M-param vectors on GPU was the OOM cause on CIFAR-100.
    dW = []                                      # global-model diffs (CPU, len<=s)
    dGi = {k: [] for k in R}                     # per-client update diffs (CPU, len<=s)
    exact_count = 0

    def push(buf, vec):
        buf.append(vec.detach().to("cpu"))
        if len(buf) > s:
            del buf[0]

    for t in range(T):
        w_bar = hist["G"][t].to(device)
        cur_dw = (w_hat - w_bar).detach()
        is_exact = (t < Tw) or ((t - Tw) % Tc == 0 and t >= Tw) or (t >= T - Tf)

        # stream-accumulate the weighted aggregate instead of stacking all R
        # deltas at once (avoids an [R, 11M] tensor on GPU)
        tot = float(sum(sizes[k] for k in R))
        agg = torch.zeros_like(w_hat)
        abn = 0
        for k in R:
            g_bar = (hist["C"][t][k].to(device) if (t in hist["C"] and k in hist["C"][t])
                     else torch.zeros_like(w_hat))
            if is_exact:
                u = client_calibration_update(w_hat, base_model, client_loaders[k],
                                              cfg.fl.local_epochs, round_lr(cfg.fl, t, total=T),
                                              device).to(device)
                push(dGi[k], u - g_bar)
                est = u
            else:
                est = g_bar
                if len(dW) >= s and len(dGi[k]) >= s:
                    Wd = torch.stack(dW[-s:], dim=1).to(device)
                    Gd = torch.stack(dGi[k][-s:], dim=1).to(device)
                    try:
                        hv = _lbfgs_hvp(Wd, Gd, cur_dw)
                        est = g_bar + hv
                    except Exception:
                        est = g_bar
                    del Wd, Gd
                    if est.abs().max() > tau:            # abnormality fixing
                        u = client_calibration_update(w_hat, base_model, client_loaders[k],
                                                      cfg.fl.local_epochs, round_lr(cfg.fl, t, total=T),
                                              device).to(device)
                        push(dGi[k], u - g_bar)
                        est = u
                        abn += 1
            agg = agg + (sizes[k] / tot) * est
            del g_bar, est

        if is_exact:
            push(dW, cur_dw)
            exact_count += 1
        w_hat = w_hat + agg
        del agg, w_bar, cur_dw
        set_vector(model, w_hat)
        if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()
        if logger is not None:
            logger.log("method_round", method=tag, round=t, exact=bool(is_exact),
                       abnormal_fixes=abn)
        if eval_hook is not None:
            eval_hook(tag, t, model)

    set_vector(model, w_hat)
    acp = 100.0 * (1 - exact_count / T)          # avg cost-saving % (client-side)
    return model, dict(exact_rounds=exact_count, total_rounds=T,
                       cost_saving_pct=round(acp, 1), tau=tau)


def scale_fedadp_for_target_eps(hist, forget, cfg, target_eps):
    """Set FedADP's config epsilon so its COMPOSED epsilon equals target_eps.

    FedADP uses one uniform z per round, z = sqrt(2 ln(1.25/delta)) / eps_cfg,
    so composing T identical Gaussian rounds gives the budget. We bisect on z
    (composed eps is monotone decreasing in z) and invert. Pair with AdaptFU's
    --adaptfu-target-eps to sweep both methods at genuinely equal budgets."""
    from .metrics import rdp_epsilon
    T = len(hist["G"]) - 1
    s = int(cfg.adaptfu.warm_start_frac * T)
    n = sum(1 for t in range(s, T)
            if t in hist["C"] and any(k not in forget for k in hist["C"][t]))
    if n == 0 or target_eps <= 0:
        return cfg.fedadp.epsilon
    delta = cfg.fedadp.delta
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if rdp_epsilon([mid] * n, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    z = math.sqrt(lo * hi)
    cfg.fedadp.epsilon = math.sqrt(2 * math.log(1.25 / delta)) / z
    return cfg.fedadp.epsilon


@torch.no_grad()
def _mean_loss(model, client_loaders, clients, device, max_batches=3):
    """Cheap global-training-loss proxy driving FedADP's eq.4 budget update."""
    crit = torch.nn.CrossEntropyLoss()
    model.eval(); tot, n = 0.0, 0
    for k in clients:
        for i, (x, y) in enumerate(client_loaders[k]):
            tot += float(crit(model(x.to(device)), y.to(device))); n += 1
            if i + 1 >= max_batches:
                break
    return tot / max(1, n)


def fedadp_select(hist, pmask, lam=0.6, gam=0.7, mode="staged", n_stages=10):
    """FedADP dual-layered selection (Alg.1 lines 16-21).

    Global models: cosine to the predecessor, ReLU'd; keep the `lam` fraction with
    the LEAST alignment (largest change) -- eq.7/8.
    Updates: within each kept round, cosine to that round's aggregate; keep the
    `gam` fraction with the HIGHEST alignment -- eq.9.
    The paper triggers this online when the loss drops by beta; we apply the same
    criteria post-hoc over the cached history, which selects the same quantity."""
    T = len(hist["G"]) - 1
    scored = []
    for t in range(1, T + 1):
        a_, b_ = hist["G"][t][pmask], hist["G"][t - 1][pmask]
        c = float(torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12))
        scored.append((max(c, 0.0), t - 1))          # ReLU(cos), round index t-1
    if mode == "staged":
        # Alg.1 lines 15-21: models accumulate in T1 and min_alignment(T1, lam)
        # fires each time the loss drops by beta, then T1 is EMPTIED. Selection is
        # therefore per-STAGE and the stored set SPANS the whole trajectory.
        # Ranking all rounds globally instead concentrates the picks on the early
        # rounds (consecutive models differ most there), so the replay can only
        # rebuild toward mid-training quality. We approximate the loss-triggered
        # stages with equal-width segments and take the lam fraction least-aligned
        # within each.
        edges = np.linspace(0, T, n_stages + 1).astype(int)
        rounds = []
        for i in range(n_stages):
            seg = [x for x in scored if edges[i] <= x[1] < edges[i + 1]]
            if not seg:
                continue
            seg = sorted(seg, key=lambda x: x[0])          # least aligned first
            rounds += [r for _, r in seg[:max(1, int(round(lam * len(seg))))]]
        rounds = sorted(set(rounds))
    else:                                                  # legacy global ranking
        scored.sort(key=lambda x: x[0])
        rounds = sorted(i for _, i in scored[:max(1, int(lam * T))])

    sel = {}
    for t in rounds:
        cd = hist["C"].get(t)
        if not cd:
            continue
        Gt = torch.stack([cd[k] for k in cd]).mean(0)[pmask]
        sc = []
        for k, c in cd.items():
            cp = c[pmask]
            sc.append((float(torch.dot(cp, Gt) / (cp.norm() * Gt.norm() + 1e-12)), k))
        sc.sort(key=lambda x: -x[0])                 # most aligned first
        sel[t] = [k for _, k in sc[:max(1, int(gam * len(sc)))]]
    return rounds, sel


# ------------------------------- FedADP -------------------------------
def run_fedadp(base_model, hist, client_loaders, cfg, device,
               logger=None, eval_hook=None, tag="FedADP"):
    fa = cfg.fedadp
    forget = set(cfg.forget.forget_clients)
    T = len(hist["G"]) - 1
    sizes = hist["sizes"]

    # Warm-start like AdaptFU. FedADP's selective storage means M_bar_0 is a
    # partially-trained "first stored" model, not the raw init; warm-starting at
    # the same round as AdaptFU is both faithful to that and required for the DP
    # variant to be numerically viable (rebuilding from scratch THROUGH the DP
    # noise never converges). Only difference from AdaptFU stays: uniform z +
    # direction-consistency calibration.
    _, _, m, s = calibrate_constants(hist, forget, cfg.adaptfu.warm_start_frac,
                                     cfg.adaptfu.q_low, cfg.adaptfu.q_high,
                                     pmask=param_mask(base_model))
    model = copy.deepcopy(base_model).to(device)
    pmask_c = param_mask(base_model)
    pmask = pmask_c.to(device)

    # Alg.1 dual-layered selection: which rounds/updates are stored at all
    if getattr(fa, "dual_select", False):
        sel_rounds, sel_clients = fedadp_select(hist, pmask_c,
                                                fa.lambda_sel, fa.gamma_sel,
                                                mode=getattr(fa, "select_mode", "staged"),
                                                n_stages=getattr(fa, "n_stages", 10))
    else:
        sel_rounds = list(range(T)); sel_clients = {t: list(hist["C"].get(t, {}))
                                                    for t in range(T)}
    # Alg.2 line 1: M_hat_0 <- M_bar_0, the first SELECTED stored global model
    if getattr(fa, "init_mode", "warm") == "first_stored" and sel_rounds:
        s = sel_rounds[0]
    G = hist["G"][s].clone().to(device)
    set_vector(model, G)

    # FedADP's native DP is a TRAINING-phase mechanism (its Algorithm 1 noises the
    # stored updates); the unlearning phase (Algorithm 2) is noise-free calibration.
    # With apply_dp=False (default) we reproduce that: pure direction-consistency
    # calibration, which performs at reconstruction/retrain level. apply_dp=True is
    # an OPTIONAL unlearning-phase perturbation (strong-privacy, utility-trading);
    # it uses the same bounded regime-relative mechanism as AdaptFU with a single
    # round-uniform z, for users who want an unlearning-phase DP contrast.
    z = 0.0
    b = 1.0
    eps_t = fa.epsilon                       # eq.4 state, updated each round
    prev_loss = None
    if fa.apply_dp:
        z = math.sqrt(2 * math.log(1.25 / fa.delta)) / eps_t       # noise / sensitivity
        b = fa.bound_scale * m

    z_bars = []
    for t in [r for r in sel_rounds if r >= s]:
        if t not in hist["C"]:
            continue
        R = [k for k in sel_clients.get(t, []) if k not in forget]
        if not R:
            continue
        cals, w = [], []
        coss = []
        for k in R:
            ce = fa.calib_epochs or cfg.fl.local_epochs   # 0 -> standard FL epochs
            u = client_calibration_update(G, base_model, client_loaders[k],
                                          ce, round_lr(cfg.fl, t, total=T),
                                          device).to(device)
            g_bar = hist["C"][t][k].to(device)
            # direction consistency + magnitude match on PARAMETER coords only
            gp, up = g_bar[pmask], u[pmask]
            cos = torch.dot(gp, up) / (gp.norm() * up.norm() + 1e-12)         # eq. 10
            coss.append(float(cos))
            scale = cos * (gp.norm() / (up.norm() + 1e-12))                   # eq. 11
            cal = torch.zeros_like(u)                # BN part recomputed from data
            cal[pmask] = scale * up
            if fa.apply_dp:
                sig = z * b
                if getattr(fa, "noise_mode", "coord") == "norm":
                    sig = sig / math.sqrt(float(int(pmask.sum())))
                noise = torch.randn_like(cal); noise[~pmask] = 0.0
                cal = cal + noise * sig              # noise magnitude ~ z*b
                pn = cal[pmask].norm()
                if pn > b:                           # clip params AFTER noise (bounded)
                    cal[pmask] = cal[pmask] * (b / pn)
            cals.append(cal); w.append(sizes[k])
        if fa.apply_dp:
            z_bars.append(z)
        W = torch.tensor(w, dtype=torch.float32, device=device); W = W / W.sum()
        U = (torch.stack(cals) * W[:, None]).sum(0)  # eq. 12
        G = G + U
        set_vector(model, G)
        recompute_bn(model, client_loaders, R, device)
        G = get_vector(model)
        # --- FedADP eq.4: adapt the budget across rounds from the loss change ---
        if fa.apply_dp and getattr(fa, "adaptive_eps", False):
            cur = _mean_loss(model, client_loaders, R, device)
            if prev_loss is not None:
                eps_t = min(max(eps_t * math.exp(abs(prev_loss - cur)),
                                fa.eps_min), fa.eps_max)
                z = math.sqrt(2 * math.log(1.25 / fa.delta)) / eps_t
            prev_loss = cur
        if logger is not None:
            logger.log("method_round", method=tag, round=t,
                       mean_cos=round(float(np.mean(coss)), 4),
                       calib_epochs=int(fa.calib_epochs or cfg.fl.local_epochs),
                       init_round=int(s), n_sel_rounds=len(sel_rounds),
                       eps_t=round(eps_t, 4) if fa.apply_dp else None,
                       adaptive_eps=bool(getattr(fa, "adaptive_eps", False)),
                       noise_multiplier=round(z, 3) if fa.apply_dp else None)
        if eval_hook is not None:
            eval_hook(tag, t, model)

    set_vector(model, G)
    recompute_bn(model, client_loaders, [k for k in sizes if k not in forget], device)
    return model, dict(dp=fa.apply_dp, noise_multiplier=round(z, 3) if z else None,
                       z_bar=z_bars)
