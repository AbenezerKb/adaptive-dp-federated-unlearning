"""Federated training (FedAvg) with the history caching AdaptFU needs.

We cache, per round t: the global model vector G_t and each participating
client's update C_t^k = w_t^k - G_t (as flat vectors). AdaptFU replays exactly
this cached history. FuGuard does not need it, but we train one shared base model
so both methods start from an identical `Standard FL` checkpoint.
"""
import copy
import math
import torch
import torch.nn as nn
from .utils import get_vector, set_vector


def round_lr(fl_cfg, t, total=None):
    """Learning rate for round t. FL schedules decay ACROSS rounds (server side),
    not within local epochs. A fixed lr plateaus early on CIFAR; cosine decay to
    `lr_min` lets the model fine-converge. Reconstruction methods replaying round
    t should call this with the ORIGINAL round count so calibration uses the same
    lr the client actually used at that round."""
    T = fl_cfg.rounds if total is None else total
    if getattr(fl_cfg, "lr_schedule", "constant") != "cosine" or T <= 1:
        return fl_cfg.lr
    lo = getattr(fl_cfg, "lr_min", 0.0)
    frac = min(max(t, 0), T - 1) / (T - 1)
    return lo + 0.5 * (fl_cfg.lr - lo) * (1 + math.cos(math.pi * frac))


def local_train(model, loader, epochs, lr, momentum, wd, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    return get_vector(model)


def federated_train(global_model, client_loaders, fl_cfg, device,
                    exclude=None, cache_history=True, verbose=True,
                    attack_hook=None, logger=None, phase="fl", eval_fn=None,
                    eval_every=0):
    """Run FedAvg. Returns (final_model, history).

    history = dict(G=[G_0..G_T], C={round: {client: update_vec}}, sizes={client:n}).
    exclude: set of client indices to skip (used for retrain-from-scratch).
    attack_hook(round, client, update_vec) -> update_vec lets us inject poisoning
        during *training* if desired (not used for AdaptFU's unlearning-phase attack).
    logger/phase/eval_fn/eval_every: structured per-round logging for visualization.
    """
    exclude = set(exclude or [])
    active = [k for k in range(len(client_loaders)) if k not in exclude]
    sizes = {k: len(client_loaders[k].dataset) for k in active}

    G = get_vector(global_model).to(device)
    hist = {"G": [G.clone().cpu()], "C": {}, "sizes": sizes}

    for t in range(fl_cfg.rounds):
        lr_t = round_lr(fl_cfg, t)
        updates, weights, round_c = [], [], {}
        for k in active:
            local = copy.deepcopy(global_model).to(device)
            set_vector(local, G)
            w_k = local_train(local, client_loaders[k], fl_cfg.local_epochs,
                              lr_t, fl_cfg.momentum, fl_cfg.weight_decay, device)
            c_k = (w_k - G).cpu()
            if attack_hook is not None:
                c_k = attack_hook(t, k, c_k)
            round_c[k] = c_k
            updates.append(c_k)
            weights.append(sizes[k])
        W = torch.tensor(weights, dtype=torch.float32)
        W = W / W.sum()
        agg = torch.stack(updates) * W[:, None]
        step = agg.sum(0)
        G = G + step.to(device)
        hist["G"].append(G.clone().cpu())
        if cache_history:
            hist["C"][t] = round_c
        if logger is not None:
            mean_norm = float(torch.stack(updates).norm(dim=1).mean())
            logger.log("fl_round", phase=phase, round=t,
                       lr=round(float(lr_t), 6),
                       mean_update_norm=round(mean_norm, 4),
                       step_norm=round(float(step.norm()), 4))
            if eval_fn is not None and eval_every and (t % eval_every == 0
                                                       or t == fl_cfg.rounds - 1):
                set_vector(global_model, G)
                logger.log("fl_eval", phase=phase, round=t,
                           test_acc=round(eval_fn(global_model), 4))
        if verbose and (t % max(1, fl_cfg.rounds // 5) == 0 or t == fl_cfg.rounds - 1):
            print(f"  [FL:{phase}] round {t+1}/{fl_cfg.rounds}")
    set_vector(global_model, G)
    return global_model, hist
