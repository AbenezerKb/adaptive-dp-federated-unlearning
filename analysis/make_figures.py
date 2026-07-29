"""Generate publication figures. Outputs PDF (for LaTeX) and PNG (for preview)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import results_data as R

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# colourblind-safe (Okabe-Ito)
CLR = {"Standard FL": "#999999", "FL Retrain": "#000000", "FuGuard": "#E69F00",
       "FedEraser": "#009E73", "FedRecover": "#56B4E9", "FedADP": "#CC79A7",
       "AdaptFU": "#0072B2"}
MK = {"Standard FL": "s", "FL Retrain": "*", "FuGuard": "^", "FedEraser": "v",
      "FedRecover": "D", "FedADP": "P", "AdaptFU": "o"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")

# ---------------- Fig 1: privacy-utility frontier, 3 datasets ----------------
def fig1():
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.5))
    for ax, (ds, rows) in zip(axes, R.MAIN.items()):
        rt = rows["FL Retrain"]
        ax.axvline(rt["mia"], color="#bbbbbb", ls="--", lw=0.8, zorder=0)
        ax.axhline(rt["acc"], color="#bbbbbb", ls="--", lw=0.8, zorder=0)
        floor = rt["acc"] - 0.15          # anything below this has collapsed
        for m in R.ORDER:
            d = rows[m]
            if d["acc"] < floor:                      # off-scale collapse
                continue
            ax.scatter(d["mia"], d["acc"], c=CLR[m], marker=MK[m],
                       s=95 if m == "FL Retrain" else 46,
                       edgecolors="white", linewidths=0.6, zorder=3,
                       label=m if ax is axes[0] else None)
        accs = [d["acc"] for m, d in rows.items() if d["acc"] >= floor]
        pad = 0.35 * (max(accs) - min(accs) + 1e-3)
        ax.set_ylim(min(accs) - pad, max(accs) + pad)
        ax.set_xlabel("MIA AUC  (lower = more forgetting)")
        ax.set_title(ds)
        # note any excluded collapse
        off = [m for m, d in rows.items() if d["acc"] < floor]
        if off:
            ax.text(0.97, 0.04,
                    "  ".join(f"{m} off-scale ({rows[m]['acc']:.2f})" for m in off),
                    transform=ax.transAxes, fontsize=7, color="#888888", ha="right")
    axes[0].set_ylabel("test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handletextpad=0.3, columnspacing=1.1)
    fig.suptitle("Privacy–utility frontier. Dashed lines mark the FL-Retrain "
                 "reference; the ideal point is its intersection.", y=1.04,
                 fontsize=9)
    save(fig, "fig1_frontier")

# ---------------- Fig 2: robustness grid + survivors -------------------------
def fig2():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    aggs = list(R.ROBUST_C100); atks = ["none", "noise", "minmax"]
    lbl = {"none": "no attack", "noise": "norm-inflating",
           "minmax": "norm-matched"}
    col = {"none": "#999999", "noise": "#0072B2", "minmax": "#E69F00"}
    x = np.arange(len(aggs)); w = 0.26
    for i, atk in enumerate(atks):
        v = [R.ROBUST_C100[a][atk][0] for a in aggs]
        a1.bar(x + (i - 1) * w, v, w, label=lbl[atk], color=col[atk],
               edgecolor="white", linewidth=0.5)
    a1.set_xticks(x); a1.set_xticklabels(aggs)
    a1.set_ylim(0.66, 0.745); a1.set_ylabel("test accuracy")
    a1.set_title("(a) AdaptFU under attack — CIFAR-100, all aggregation rules")
    a1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    # survivors: verification fires only on the norm-inflating tier
    sv = {atk: [R.ROBUST_C100[a][atk][1] for a in aggs] for atk in atks}
    for i, atk in enumerate(atks):
        a2.bar(x + (i - 1) * w, sv[atk], w, color=col[atk],
               edgecolor="white", linewidth=0.5)
    a2.axhline(9, color="#bbbbbb", ls="--", lw=0.8)
    a2.text(len(aggs) - 0.55, 9.06, "all clients admitted", fontsize=7,
            color="#888888", ha="right")
    a2.set_xticks(x); a2.set_xticklabels(aggs, rotation=12)
    a2.set_ylim(6, 9.8); a2.set_ylabel("clients surviving verification (of 9)")
    a2.set_title("(b) Divergence-bound verification")
    save(fig, "fig2_robustness")

# ---------------- Fig 3: DP ablation, per-client vs per-round ----------------
def fig3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.1),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    d = R.DP_ABLATION_C100
    seeds = [s for s, _, _ in d]
    delta = [100 * (pc - pr) for _, pc, pr in d]
    a1.axhline(0, color="#000000", lw=0.8)
    a1.bar(seeds, delta, 0.5, color=["#0072B2" if v > 0 else "#CC79A7" for v in delta],
           edgecolor="white", linewidth=0.5)
    m, sd = np.mean(delta), np.std(delta, ddof=1)
    a1.axhline(m, color="#D55E00", ls="--", lw=1,
               label=f"mean {m:+.3f} pp  (t={m/(sd/2):.2f}, df=3)")
    a1.fill_between([-0.6, 3.6], m - sd, m + sd, color="#D55E00", alpha=0.10)
    a1.set_xlim(-0.6, 3.6); a1.set_xticks(seeds)
    a1.set_xlabel("noise seed (paired: identical base and noise draws)")
    a1.set_ylabel("per-client − per-round  (pp)")
    a1.set_title("(a) CIFAR-100, matched ε = 48.51")
    a1.legend(frameon=False, loc="upper right")

    pc, pr = R.DP_ABLATION_FEMNIST
    a2.bar([0, 1], [pc, pr], 0.45, color=["#0072B2", "#CC79A7"],
           edgecolor="white", linewidth=0.5)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["per-client", "per-round"])
    a2.set_ylim(0.80, 0.83); a2.set_ylabel("test accuracy")
    a2.set_title(f"(b) FEMNIST, matched ε = 26.79\nΔ = {100*(pc-pr):+.2f} pp, "
                 f"MIA identical (0.5066)")
    save(fig, "fig3_dp_ablation")

# ---------------- Fig 4: mechanism diagnostics -------------------------------
def fig4():
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    # (a) eq.9 renormalisation cancels z
    a = ax[0]
    bars = ["eq. 9\n(per-coord std)", "magnitude-\ncalibrated"]
    vals = [R.EQ9["rel_model_diff_coord"], R.EQ9["rel_model_diff_norm"]]
    a.bar(bars, vals, 0.45, color=["#CC79A7", "#0072B2"],
          edgecolor="white", linewidth=0.5)
    a.set_yscale("log"); a.set_ylabel(r"$\|$adaptive $-$ uniform$\|$ / $\|$uniform$\|$")
    a.set_title("(a) Is per-client $z$ observable?")
    for i, v in enumerate(vals):
        a.text(i, v * 1.4, f"{v:.1e}", ha="center", fontsize=7.5)
    a.text(0.5, 0.06, "180× separation", transform=a.transAxes, ha="center",
           fontsize=7.5, color="#888888")

    # (b) rho scope controls the z spread
    b = ax[1]
    b.boxplot([R.RHO_SCOPE["pooled"], R.RHO_SCOPE["round"]],
              tick_labels=["pooled", "round-local"], widths=0.45,
              medianprops=dict(color="#D55E00"))
    b.set_ylabel("within-round sd of per-client $z$")
    b.set_title(r"(b) Scope of the $\rho$ percentiles")
    b.text(0.5, 0.92, "pooled ⇒ collapses to near-uniform", transform=b.transAxes,
           ha="center", fontsize=7.5, color="#888888")

    # (c) FedADP's cos-scaling needs reproducible local updates
    c = ax[2]
    lr = [x for x, _ in R.FEDADP_LR]; cs = [y for _, y in R.FEDADP_LR]
    c.semilogx(lr, cs, "o-", color="#009E73", lw=1.4, ms=5)
    c.axhline(0.9, color="#bbbbbb", ls="--", lw=0.8)
    c.set_xlabel("local learning rate"); c.set_ylabel(r"$\cos(\bar g,\hat g)$")
    c.set_title("(c) FedADP update reproducibility")
    c.set_ylim(-0.05, 1.05)
    c.annotate("paper's lr", xy=(0.005, 0.99), xytext=(0.012, 0.62), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", lw=0.7, color="#888888"))
    c.text(0.5, 0.06, r"$\|U\|=|\cos|\cdot\|\bar g\|$", transform=c.transAxes,
           ha="center", fontsize=8)
    save(fig, "fig4_mechanisms")

def fig5():
    """Residual MIA gap is a property of the attacker, not the method."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    ds = list(R.ATTACKER_STRENGTH)
    x = [R.ATTACKER_STRENGTH[d][0] for d in ds]
    y = [R.ATTACKER_STRENGTH[d][1] - R.ATTACKER_STRENGTH[d][2] for d in ds]
    r = np.corrcoef(x, y)[0, 1]
    b = np.polyfit(x, y, 1)
    xs = np.linspace(min(x) - .01, max(x) + .01, 50)
    ax.plot(xs, np.polyval(b, xs), color="#bbbbbb", lw=1, zorder=1)
    ax.scatter(x, y, s=58, c="#0072B2", edgecolors="white", linewidths=0.6, zorder=3)
    for d, xi, yi in zip(ds, x, y):
        ax.annotate(d, (xi, yi), textcoords="offset points", xytext=(6, -3),
                    fontsize=7.5)
    ax.set_xlabel("attacker strength (MIA AUC vs Standard FL)")
    ax.set_ylabel("residual gap  (AdaptFU $-$ Retrain)")
    ax.set_title(f"Residual leakage scales with the attacker\n"
                 f"Pearson $r={r:.3f}$ (4 datasets)")
    save(fig, "fig5_attacker_strength")




def fig1c():
    """Single-column frontier: every method from every dataset plotted RELATIVE to
    that dataset's FL-Retrain reference, so the origin is the ideal point and the
    cluster structure is visible in one panel instead of four."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
    DSM = {"CIFAR-10": "o", "CIFAR-100": "s", "FEMNIST": "^", "AG News": "D"}
    seen = set()
    for ds, rows in R.MAIN.items():
        rt = rows["FL Retrain"]
        for m in R.ORDER:
            if m == "FL Retrain":
                continue
            d = rows[m]
            if d["acc"] < rt["acc"] - 0.15:      # collapsed, off-scale
                continue
            ax.scatter(d["mia"] - rt["mia"], 100 * (d["acc"] - rt["acc"]),
                       c=CLR[m], marker=DSM[ds], s=34, edgecolors="white",
                       linewidths=0.5, zorder=3,
                       label=m if m not in seen else None)
            seen.add(m)
    ax.scatter([0], [0], marker="*", s=150, c="#000000", zorder=4,
               edgecolors="white", linewidths=0.6, label="FL Retrain")
    ax.set_xlabel(r"MIA $-$ retrain   (right = more leakage)")
    ax.set_ylabel(r"accuracy $-$ retrain (pp)")
    ax.set_title("Frontier relative to FL Retrain", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2,
              handletextpad=0.15, columnspacing=0.5, borderpad=0.15)
    save(fig, "fig1c_frontier_compact")

if __name__ == "__main__":
    print("generating figures ...")
    fig1(); fig1c(); fig2(); fig3(); fig4(); fig5()
