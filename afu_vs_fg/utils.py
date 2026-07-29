"""Small utilities: seeding, model<->vector flattening, cloning."""
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _float_items(model):
    """Deterministically-ordered (key, tensor) list of all FLOAT tensors in the
    state_dict: trainable params AND float buffers such as BatchNorm
    running_mean / running_var. Integer buffers (e.g. num_batches_tracked) are
    skipped so they neither get averaged nor break the float concat. Aggregating
    BN running stats is required for FedAvg-with-BN to work; leaving them out
    pins a BN network near chance under non-IID shards."""
    return [(k, v) for k, v in model.state_dict().items()
            if torch.is_floating_point(v)]


def param_mask(model):
    """Boolean mask over the flat get_vector(model) that is True on trainable-
    parameter coordinates and False on buffer (BatchNorm running-stat)
    coordinates. Reconstruction methods that rescale by norm/direction MUST
    restrict that math to params: BN running_var lives on a ~100x larger scale
    and otherwise dominates the L2 norm (~300x here), so an unmasked unit
    direction is almost entirely BN stats and reconstruction collapses."""
    pkeys = set(k for k, _ in model.named_parameters())
    parts = []
    for k, v in _float_items(model):
        fill = torch.ones if k in pkeys else torch.zeros
        parts.append(fill(v.numel(), dtype=torch.bool))
    return torch.cat(parts)


def get_vector(model):
    return torch.cat([v.detach().reshape(-1) for _, v in _float_items(model)])


def set_vector(model, vec):
    i = 0
    for _, v in _float_items(model):   # v is the live tensor in the model
        n = v.numel()
        v.copy_(vec[i:i + n].view_as(v))
        i += n


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_state(model, state):
    model.load_state_dict(state)


@torch.no_grad()
def recompute_bn(model, client_loaders, retained, device, max_batches=8):
    """Recompute BatchNorm running statistics for the current weights by
    forward-passing retained-client data. Reconstruction methods must NOT carry
    BN stats as additive deltas (that can drive running_var invalid on the
    reconstruction's own trajectory and collapse eval); instead we reset and
    repopulate them from data, which is the standard, stable approach. No-op for
    BN-free models (e.g. LeNet)."""
    import torch.nn as nn
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None            # cumulative average over the passes
    model.train()
    for k in retained:
        for i, (x, _) in enumerate(client_loaders[k]):
            model(x.to(device))
            if i + 1 >= max_batches:
                break
    for m, mo in zip(bns, saved):
        m.momentum = mo
    model.eval()
