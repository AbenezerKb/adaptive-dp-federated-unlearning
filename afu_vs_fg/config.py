"""Configuration for the AdaptFU vs FuGuard comparison harness.

One place to set everything. Per-dataset defaults follow the two papers where
they agree, and are noted where they differ (AdaptFU uses Dirichlet alpha=0.5,
FuGuard uses alpha=0.1 for CIFAR-10 / SVHN). We expose alpha as a single knob so
both methods are compared on the *same* partition; change `dirichlet_alpha` to
reproduce either paper's native setting.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    name: str = "mnist"              # mnist | cifar10 | cifar100
    n_clients: int = 10
    dirichlet_alpha: float = 0.5     # AdaptFU=0.5; FuGuard=0.1 (c10) / 1.0 (c100)
    data_root: str = "./data"
    val_fraction: float = 0.0
    augment: bool = True             # CIFAR train-time RandomCrop+Flip (standard);
                                     # without it ResNet-18 caps ~0.55 on CIFAR-100


@dataclass
class FLConfig:
    rounds: int = 30                 # MNIST 30, CIFAR-10 50, CIFAR-100 80 (papers)
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    lr_schedule: str = "cosine"      # cosine | constant  (decay ACROSS rounds)
    lr_min: float = 1e-3             # cosine floor; fixed lr plateaus early
    momentum: float = 0.9
    weight_decay: float = 5e-4
    participation: float = 1.0       # all clients each round (both papers)


@dataclass
class ForgetConfig:
    # Which clients to forget. Forget ratio 5/10/25% -> 1/1/3 clients (AdaptFU).
    forget_clients: List[int] = field(default_factory=lambda: [0])


@dataclass
class AdaptFUConfig:
    warm_start_frac: float = 0.5     # w in the paper
    calib_epochs: int = 1            # E_c live calibration epochs
    alpha0: float = 0.5              # blend schedule start
    alphaT: float = 0.9              # blend schedule end
    eps_min: float = 1.0
    eps_max: float = 8.0
    z_min: float = 0.5               # noise multiplier for reliable clients
    z_max: float = 1.5               # noise multiplier for unreliable clients
    b_scale: float = 3.0             # divergence bound = b_scale * median global step
    kappa: float = 3.0               # MAD tolerance for verification
    q_low: float = 5.0               # robust percentile for rho_min
    q_high: float = 95.0
    rho_scope: str = "round"         # round | pooled -- scope of the rho_min/rho_max
                                     # percentiles used by the reliability score.
                                     # "pooled" (paper default) spans ALL replay
                                     # rounds; if update norms decay across rounds
                                     # (e.g. cosine LR) that across-round span
                                     # swamps within-round differences and every
                                     # client gets nearly the same z -- i.e. the
                                     # adaptive mechanism silently degenerates to
                                     # uniform. "round" uses the current round's
                                     # retained-client norms, restoring per-client
                                     # discrimination (same reasoning the paper
                                     # applies to the regime-relative bound).
    dp_mode: str = "per_client"      # per_client | per_round | static
                                     #   per_client - z differs BETWEEN CLIENTS
                                     #     within a round (AdaptFU's claim)
                                     #   per_round  - z same for all clients in a
                                     #     round but MOVES ACROSS ROUNDS. This is
                                     #     FedADP's shape: adaptive, just on the
                                     #     round axis rather than the client axis.
                                     #     NOT static. Same round-mean as
                                     #     per_client => identical epsilon, so the
                                     #     pair isolates the client axis exactly.
                                     #   static     - one fixed z everywhere.
                                     # ("adaptive"/"uniform" accepted as aliases)
                                     # "uniform" replaces each z_k by the round
                                     # mean -> identical composed epsilon, but no
                                     # per-client allocation. That is the exact
                                     # ablation the paper's utility claim rests on.
    delta: float = 1e-5              # for RDP -> (eps, delta)
    noise_mode: str = "norm"         # coord | norm
                                     # "coord": eq.9 literal -- per-coordinate std
                                     #   = z*b, so ||noise|| = z*b*sqrt(d). At
                                     #   d~1e7 that is ~4000x the update norm, so
                                     #   Pi_b renormalises it to b*unit(noise) and
                                     #   the z factor CANCELS EXACTLY: per-client
                                     #   allocation becomes mathematically inert
                                     #   and the client signal is fully swamped.
                                     # "norm": std = z*b/sqrt(d) so E||noise|| =
                                     #   z*b, comparable to the clip bound. Signal
                                     #   survives and z affects the output, making
                                     #   the allocation testable. NOTE: this is a
                                     #   deviation from eq.9 -- the (eps,delta)
                                     #   accounting no longer applies as stated.
    agg_weight: str = "samples"      # samples | reliability
                                     # eq.10 aggregates with n_k = |D_k| (sample
                                     # counts). But the paper's prose says an
                                     # outlier client is "to be down-weighted",
                                     # which suggests reliability may also enter the
                                     # WEIGHTS. "reliability" uses n_k * r_k. This is
                                     # a far more direct utility lever than noise
                                     # placement and is NOT subject to the Jensen
                                     # cancellation (uniform z minimises aggregate
                                     # noise variance at fixed mean z).
    clip_order: str = "after"        # after | before
                                     # "after" = eq.9 literal, Pi_b(u + noise);
                                     # partially renormalises away z's effect.
                                     # "before" = standard DP-SGD order (clip to
                                     # bound sensitivity, then perturb), leaving z's
                                     # effect on the noise undiluted.
    aggregation: str = "fedavg"      # fedavg | krum | trimmed_mean | median
    verify_before_clip: bool = True  # ablation switch (paper's key ordering claim)


@dataclass
class FuGuardConfig:
    unlearn_epochs: int = 5
    recovery_rounds: int = 5
    lr_unlearn: float = 1e-3
    lambda_ot: float = 0.1
    sinkhorn_eps: float = 0.1        # entropic regularization strength
    sinkhorn_iters: int = 50
    proxy_fraction: float = 0.10     # proxy samples = 10% of client data (paper)
    pca_components: int = 5           # K
    pca_direction: int = 2            # r (1-indexed principal direction)
    pca_alpha: float = 1.0            # perturbation strength
    vae_pretrain_epochs: int = 5
    vae_latent: int = 32


@dataclass
class FedEraserConfig:
    retain_interval: int = 2         # Delta_t: retain every Delta_t rounds
    calibration_ratio: float = 0.5   # r = E_cali / E_local (paper's default)


@dataclass
class FedRecoverConfig:
    warmup: int = 6                  # T_w exact rounds at the start
    correction: int = 3              # T_c: exact every T_c rounds after warm-up
    final_tuning: int = 5            # T_f exact rounds at the end
    buffer_size: int = 2             # s in L-BFGS
    tolerance: float = 1e-3          # alpha -> abnormality threshold percentile
    tau_scale: float = 1.0           # safety factor on the chosen threshold


@dataclass
class FedADPConfig:
    calib_epochs: int = 0            # 0 = use fl.local_epochs (PAPER-FAITHFUL).
                                     # Alg.2: the remaining clients "train the global
                                     # model as in standard FL training", i.e. the
                                     # same local epochs as training -- not 1. With
                                     # 1 epoch g_hat is weaker AND worse-aligned, so
                                     # cos(theta) drops and the calibrated update
                                     # ||U|| = |cos|*||g_bar|| is over-damped. Under a
                                     # warm start that is survivable; rebuilding from
                                     # M_bar_0 it collapses the model.
    # FedADP's DP lives in the FL TRAINING phase (its Algorithm 1): stored updates
    # are already noised, and the UNLEARNING phase (Algorithm 2) is noise-free
    # calibration. In this shared-non-DP-base harness the faithful default is
    # noise-free unlearning, which reproduces the paper's high-performing
    # "FedADP without DP" configuration. Enabling apply_dp adds an optional
    # unlearning-phase perturbation (a strong-privacy setting that trades utility);
    # it is NOT how FedADP natively obtains DP, so it is off by default.
    apply_dp: bool = False
    epsilon: float = 3.0             # per-round budget when apply_dp=True
    bound_scale: float = 3.0         # DP bound = bound_scale * median step
    noise_mode: str = "norm"         # coord | norm -- MUST match AdaptFU's setting
                                     # or the DP comparison is unfair: "coord" is
                                     # mathematically inert (see AdaptFUConfig).
    adaptive_eps: bool = True        # FedADP's actual mechanism: the budget adapts
                                     # ACROSS ROUNDS via eq.4,
                                     #   eps_{t+1}=min(max(eps_t*e^dLoss, lo), hi)
                                     # so noise shrinks as the loss stabilises. A
                                     # FIXED eps is not FedADP -- it is static DP,
                                     # and drops the very mechanism being contrasted
                                     # with AdaptFU's per-CLIENT allocation.
    init_mode: str = "first_stored"  # first_stored | warm
                                     #   first_stored - Alg.2 line 1: M_hat_0 <- M_bar_0,
                                     #     the FIRST SELECTED stored global model.
                                     #     Faithful; replays the whole selected
                                     #     trajectory.
                                     #   warm - start at AdaptFU's warm-start round.
                                     #     NOT FedADP; lends it AdaptFU's utility/
                                     #     forgetting operating point.
    lambda_sel: float = 0.6          # Alg.1: global-model selection ratio (paper 60%)
    gamma_sel: float = 0.7           # Alg.1: update selection ratio (paper 70%)
    dual_select: bool = True         # apply the dual-layered selection at all
    select_mode: str = "staged"      # staged | global
                                     #   staged - Alg.1: selection fires per STAGE
                                     #     (loss-drop triggered, T1 emptied each
                                     #     time), so stored models SPAN training.
                                     #   global - rank all rounds at once; skews
                                     #     hard to early rounds and leaves the late
                                     #     trajectory unreplayable.
    n_stages: int = 10               # stage count approximating the beta triggers
    eps_min: float = 1.0             # eq.4 lower clamp
    eps_max: float = 8.0             # eq.4 upper clamp
    delta: float = 1e-5


@dataclass
class AttackConfig:
    enabled: bool = False
    kind: str = "noise"              # noise (norm-inflating) | minmax (norm-matched)
    malicious_fraction: float = 0.20 # 20% -> 2 of 10 (AdaptFU robustness eval)
    noise_scale: float = 3.0         # inflation factor for the noise attack
    backdoor: bool = False           # FuGuard's native axis: trigger + target label
    backdoor_target: int = 9
    backdoor_frac: float = 0.5       # fraction of attacker samples triggered


@dataclass
class ExpConfig:
    data: DataConfig = field(default_factory=DataConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    forget: ForgetConfig = field(default_factory=ForgetConfig)
    adaptfu: AdaptFUConfig = field(default_factory=AdaptFUConfig)
    fuguard: FuGuardConfig = field(default_factory=FuGuardConfig)
    federaser: FedEraserConfig = field(default_factory=FedEraserConfig)
    fedrecover: FedRecoverConfig = field(default_factory=FedRecoverConfig)
    fedadp: FedADPConfig = field(default_factory=FedADPConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "./results"


# ---- convenience presets matching each paper's round budget ----
def preset(dataset: str) -> ExpConfig:
    cfg = ExpConfig()
    cfg.data.name = dataset
    if dataset == "mnist":
        cfg.fl.rounds = 30
        cfg.fl.lr = 0.01           # LeNet, no BatchNorm
    elif dataset == "agnews":
        cfg.fl.rounds = 50         # AdaptFU: AG News, 4 classes, TextCNN
        cfg.fl.lr = 0.01
        cfg.data.augment = False   # no image augmentation for text
    elif dataset == "femnist":
        cfg.fl.rounds = 50         # AdaptFU: 50 rounds, LeNet-5, 62 classes
        cfg.fl.lr = 0.01
        cfg.data.dirichlet_alpha = None   # natural per-writer split instead
    elif dataset == "cifar10":
        cfg.fl.rounds = 50
        cfg.fl.lr = 0.1            # ResNet-18 needs a larger step to actually learn
    elif dataset == "cifar100":
        cfg.fl.rounds = 80
        cfg.fl.lr = 0.1
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return cfg
