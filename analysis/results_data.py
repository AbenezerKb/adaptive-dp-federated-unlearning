"""Single source of truth for every number reported in the study.

Provenance is the run_id that produced each row. Anything marked PROVISIONAL
should not be cited without the caveat attached in CAVEATS below.
"""

# ---- Main comparison. All rows share one Standard-FL base per dataset. -------
# config: alpha=0.5, lr=0.01 (cosine -> 1e-3), 10 clients, forget=[0],
#         AdaptFU z in [0.62,1.54], w=0.5, E_c=1, rho_scope=round, noise_mode=norm
MAIN = {
    "CIFAR-10": {             # ResNet-18, 50 rounds         (run: c10_all)
        "Standard FL":  dict(acc=0.9198, forget=0.9868, mia=0.5375),
        "FL Retrain":   dict(acc=0.9152, forget=0.8961, mia=0.4729),
        "FuGuard":      dict(acc=0.9034, forget=0.9408, mia=0.4976),
        "FedEraser":    dict(acc=0.8730, forget=0.8483, mia=0.4272),
        "FedRecover":   dict(acc=0.9050, forget=0.8844, mia=0.4721, cost=52.0),
        "FedADP":       dict(acc=0.3911, forget=0.4204, mia=0.4941, dnc=True),
        "AdaptFU":      dict(acc=0.9116, forget=0.9382, mia=0.4931, eps=27.38),
    },
    "CIFAR-100": {            # ResNet-18, 80 rounds        (run: yc_all)
        "Standard FL":  dict(acc=0.7271, forget=0.9681, mia=0.6397),
        "FL Retrain":   dict(acc=0.7092, forget=0.6659, mia=0.4581),
        "FuGuard":      dict(acc=0.7212, forget=0.9176, mia=0.5981),
        "FedEraser":    dict(acc=0.6546, forget=0.6160, mia=0.4612),
        "FedRecover":   dict(acc=0.6861, forget=0.6433, mia=0.4592, cost=57.5),
        "FedADP":       dict(acc=0.3297, forget=0.2511, mia=0.4652, dnc=True),
        "AdaptFU":      dict(acc=0.7089, forget=0.8530, mia=0.5648, eps=44.66),
    },
    "FEMNIST": {              # LeNet-5, 50 rounds, 60k sub  (run: fem_all)
        "Standard FL":  dict(acc=0.8159, forget=0.9298, mia=0.5279),
        "FL Retrain":   dict(acc=0.8071, forget=0.8510, mia=0.5012),
        "FuGuard":      dict(acc=0.0471, forget=0.0004, mia=0.5241, dnc=True),
        "FedEraser":    dict(acc=0.8256, forget=0.8624, mia=0.4961),
        "FedRecover":   dict(acc=0.7994, forget=0.8396, mia=0.4454, cost=52.0),
        "FedADP":       dict(acc=0.8065, forget=0.8316, mia=0.5018, prov=True),
        "AdaptFU":      dict(acc=0.8233, forget=0.9044, mia=0.5066, eps=26.79),
    },
    "AG News": {              # TextCNN, 50 rounds           (run: ag_all)
        "Standard FL":  dict(acc=0.8989, forget=0.9934, mia=0.5807),
        "FL Retrain":   dict(acc=0.8892, forget=0.8886, mia=0.5059),
        "FuGuard":      dict(acc=0.8908, forget=0.9606, mia=0.5753, prov=True),
        "FedEraser":    dict(acc=0.8339, forget=0.8132, mia=0.5032),
        "FedRecover":   dict(acc=0.8830, forget=0.8762, mia=0.5128, cost=52.0),
        "FedADP":       dict(acc=0.8336, forget=0.8380, mia=0.5002, prov=True),
        "AdaptFU":      dict(acc=0.8938, forget=0.9682, mia=0.5679, eps=30.22),
    },
}
ORDER = ["Standard FL", "FL Retrain", "FuGuard", "FedEraser",
         "FedRecover", "FedADP", "AdaptFU"]

# ---- Robustness: AdaptFU under Byzantine calibration updates (20% malicious) --
# CIFAR-100, alpha=0.5.  minmax cells use the CORRECTED norm-matched attack (v19).
ROBUST_C100 = {                       # (acc, survivors_kept of 9)
    "FedAvg":       {"none": (0.7113, 9), "noise": (0.7129, 7), "minmax": (0.7114, 8)},
    "Krum":         {"none": (0.6869, 9), "noise": (0.6797, 7), "minmax": (0.7007, 8)},
    "Trimmed Mean": {"none": (0.7101, 9), "noise": (0.7086, 7), "minmax": (0.7099, 8)},
    "Median":       {"none": (0.7066, 9), "noise": (0.6963, 7), "minmax": (0.7104, 9)},
}
ROBUST_XDATA = {                      # FedAvg only, other datasets
    "CIFAR-10": {"none": (0.9116, 9), "noise": (0.9048, 7), "minmax": (0.9008, 9)},
    "FEMNIST": {"none": (0.8233, 9), "noise": (0.8220, 7), "minmax": (0.8235, 9)},
    "AG News": {"none": (0.8938, 9), "noise": (0.8913, 7), "minmax": (0.8870, 8)},
}

# ---- Ablation: per-client vs per-round DP at MATCHED epsilon ------------------
# Paired: same base checkpoint, same noise draws within a seed; only dp_mode differs.
DP_ABLATION_C100 = [   # (seed, per_client, per_round)   alpha=1.0, eps=48.51 both
    (0, 0.7075, 0.7021), (1, 0.7003, 0.7027),
    (2, 0.7115, 0.7114), (3, 0.6990, 0.7016),
]
DP_ABLATION_FEMNIST = (0.8233, 0.8239)      # alpha=0.5, eps=26.79 both

# ---- Mechanism diagnostics (measured, not cited) -----------------------------
# (a) eq.9 renormalisation cancels z at high dimension
EQ9 = {"cos(z=.5,z=1)": 1.000002, "cos(z=.5,z=1.5)": 1.000002,
       "signal_share_coord": 0.0014, "signal_share_norm_z_lo": 0.8480,
       "signal_share_norm_z_hi": 0.4698, "rel_model_diff_coord": 1.3e-6,
       "rel_model_diff_norm": 2.4e-4}
# (b) rho-percentile scope controls the per-client z spread (within-round sd)
RHO_SCOPE = {"pooled": [0.008, 0.074, 0.075, 0.096, 0.147],
             "round":  [0.213, 0.248, 0.212, 0.212, 0.265]}
# (c) FedADP's cos-scaling requires reproducible local updates
FEDADP_LR = [(0.005, 0.9903), (0.010, 0.9774), (0.050, 0.0551), (0.100, 0.2456)]
# (d) min-max must be norm-matched to evade a median+MAD check
MINMAX = {"old_ratio": 1.49, "fixed_ratio": 1.00,
          "mad_thr_honest": 48.132, "mad_thr_with_attack": 42.923}

# ---- Residual MIA gap tracks ATTACKER STRENGTH, not the method ---------------
# (Standard-FL MIA, AdaptFU MIA, Retrain MIA) per dataset; Pearson r = 0.995
ATTACKER_STRENGTH = {
    "FEMNIST":   (0.5279, 0.5066, 0.5012),
    "CIFAR-10":  (0.5375, 0.4931, 0.4729),
    "AG News":   (0.5807, 0.5679, 0.5059),
    "CIFAR-100": (0.6397, 0.5648, 0.4581),
}

CAVEATS = [
 ("FedADP", "Seven deviations from Jiang et al. were found and corrected during "
            "this study (DP phase, per-round eq.4 budget, M_bar_0 init, staged "
            "selection, gamma update selection, calibration epochs, select scope). "
            "Its faithful configuration still collapses on CIFAR-100. Treat as "
            "unresolved rather than as a measured baseline."),
 ("FuGuard", "Collapses on FEMNIST (LeNet, 61k params) from unlearning-rate "
             "sensitivity its own paper documents; the AG News proxy uses an "
             "embedding-space adaptation because its ConvVAE cannot encode tokens. "
             "Cited from the authors' institutional record (Qi et al., 2025); no "
             "peer-reviewed venue was locatable at time of writing."),
 ("MIA scale", "Standard-FL leakage differs by dataset (0.640 / 0.581 / 0.528), so "
               "residual gaps are NOT comparable across datasets. CIFAR-100 is the "
               "only setting with a discriminative attacker."),
 ("noise_mode", "All AdaptFU rows use magnitude-calibrated noise. Under eq.9 as "
                "literally written the per-client allocation is provably inert at "
                "these dimensionalities (Fig. 4a); reported epsilon is a relative "
                "reference, not an (eps,delta) guarantee."),
 ("Seeds", "Main and robustness tables are single-seed. The DP ablation is the only "
           "multi-seed result (4 paired seeds)."),
]
