"""Aggregation rules used during AdaptFU reconstruction.

All operate on a stack of flattened update vectors U (n x d) with per-client
weights w (n,). AdaptFU evaluates four rules: FedAvg, Krum, Trimmed Mean, and
Coordinate-wise Median.
"""
import torch


def fedavg(U, w):
    w = w / w.sum()
    return (w[:, None] * U).sum(0)


def krum(U, w=None, n_byz=0):
    """Krum: pick the vector minimizing summed sq-dist to its n-f-2 closest peers."""
    n = U.shape[0]
    if n == 1:
        return U[0]
    d2 = torch.cdist(U, U) ** 2
    k = max(1, n - n_byz - 2)
    scores = []
    for i in range(n):
        vals, _ = torch.sort(d2[i])
        scores.append(vals[1:k + 1].sum())      # skip self (distance 0)
    return U[int(torch.stack(scores).argmin())]


def trimmed_mean(U, w=None, trim=0.1):
    n = U.shape[0]
    k = int(trim * n)
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    s, _ = torch.sort(U, dim=0)
    return s[k:n - k].mean(0) if n - 2 * k > 0 else U.mean(0)


def coord_median(U, w=None):
    return U.median(0).values


def aggregate(name, U, w, n_byz=0):
    if name == "fedavg":
        return fedavg(U, w)
    if name == "krum":
        return krum(U, w, n_byz)
    if name == "trimmed_mean":
        return trimmed_mean(U, w)
    if name == "median":
        return coord_median(U, w)
    raise ValueError(name)
