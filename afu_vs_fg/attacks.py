"""Poisoning attacks on live calibration updates during AdaptFU unlearning.

Two tiers, matching AdaptFU's evaluation:
  - noise  : norm-INFLATING. Add Gaussian scaled to blow up the update norm.
             The divergence-bound verification is designed to catch this.
  - minmax : norm-MATCHED. Craft a deviation that maximizes distance from the
             honest mean while staying inside the honest norm envelope, so it
             passes a norm check by construction.
"""
import torch


def apply_attack(kind, honest_updates, malicious_idx, noise_scale=3.0):
    """honest_updates: list of flat update tensors (one per retained client).
    malicious_idx: indices (into the list) that are adversarial.
    Returns a new list with poisoned entries replaced."""
    U = torch.stack(honest_updates)
    out = [u.clone() for u in honest_updates]
    if not malicious_idx:
        return out

    if kind == "noise":
        for i in malicious_idx:
            u = honest_updates[i]
            noise = torch.randn_like(u)
            noise = noise / (noise.norm() + 1e-12) * u.norm()
            out[i] = u + noise_scale * noise        # norm ~ (1+scale)*||u||
        return out

    if kind == "minmax":
        # Norm-MATCHED attack (Shejwalkar & Houmansadr min-max, plus explicit norm
        # matching). Two constraints must hold or this is not the attack the paper
        # describes:
        #   (a) ||malicious|| ~= median(benign norms), so a median+MAD norm check
        #       cannot separate it -- it "sits inside the benign cluster";
        #   (b) max_i ||malicious - benign_i|| <= max_ij ||benign_i - benign_j||,
        #       the min-max distance budget.
        # Maximise deviation along the top benign-variance direction subject to both.
        benign = [u for j, u in enumerate(honest_updates) if j not in malicious_idx]
        B = torch.stack(benign) if benign else U
        mean = B.mean(0)
        target = float(B.norm(dim=1).median())          # (a) norm to match
        _, _, V = torch.linalg.svd(B - mean, full_matrices=False)
        d = V[0]
        budget = (float(torch.cdist(B, B).max()) if B.shape[0] > 1
                  else float(mean.norm()))              # (b) distance budget

        def craft(g):
            v = mean + g * d
            return v / (v.norm() + 1e-12) * target      # renormalise to benign median

        lo, hi = 0.0, 10.0 * (target + 1e-12)
        for _ in range(60):                             # bisect the largest feasible g
            mid = 0.5 * (lo + hi)
            if float((craft(mid).unsqueeze(0) - B).norm(dim=1).max()) <= budget:
                lo = mid
            else:
                hi = mid
        cand = craft(lo)
        for i in malicious_idx:
            out[i] = cand.clone()
        return out

    raise ValueError(kind)
