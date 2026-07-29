"""Emit paper-ready LaTeX tables (booktabs). Writes tables.tex."""
import numpy as np, results_data as R

def f(v, n=4):
    return "--" if v is None else f"{v:.{n}f}"

out = []
A = out.append

A(r"% Requires: \usepackage{booktabs,multirow,threeparttable}")
A("")

# ---------------- Table 1: main comparison -----------------------------------
ARCH = {"CIFAR-10": "ResNet-18", "CIFAR-100": "ResNet-18",
        "FEMNIST": "LeNet-5", "AG News": "TextCNN"}
DS = list(R.MAIN)
A(r"\begin{table*}[t]\centering\small")
A(r"\setlength{\tabcolsep}{3.2pt}")
A(r"\caption{Federated unlearning across four datasets and three architectures. "
  r"All methods share one Standard-FL base per dataset (identical partition, "
  r"$\alpha{=}0.5$, 10 clients, forgetting client~0). \textbf{FL Retrain} is the "
  r"gold standard: the ideal method matches its accuracy \emph{and} its MIA. Lower "
  r"forget-accuracy and MIA indicate stronger forgetting, but only when test "
  r"accuracy is preserved.}")
A(r"\label{tab:main}")
A(r"\begin{tabular}{@{}l " + " ".join(["ccc"] * len(DS)) + r"@{}}")
A(r"\toprule")
A("& " + " & ".join(r"\multicolumn{3}{c}{%s (%s)}" % (d, ARCH.get(d, "")) for d in DS) + r" \\")
A(" ".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(DS))))
A(r"Method & " + " & ".join(
    r"Acc$\uparrow$ & Fgt$\downarrow$ & MIA$\downarrow$" for _ in DS) + r" \\")
A(r"\midrule")
for m in R.ORDER:
    cells = []
    for ds in DS:
        d = R.MAIN[ds][m]
        if d.get("dnc"):
            cells += ["---", "---", "---"]
            continue
        mark = r"$^{\dagger}$" if d.get("prov") else ""
        cells += [f(d["acc"], 3) + mark, f(d["forget"], 3), f(d["mia"], 3)]
    name = r"\textbf{Ours}" if m == "AdaptFU" else m
    if m == "FL Retrain":
        A(r"\rowcolor{black!6} " + name + " & " + " & ".join(cells) + r" \\")
    else:
        A(name + " & " + " & ".join(cells) + r" \\")
    if m in ("Standard FL", "FL Retrain"):
        A(r"\midrule")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\begin{flushleft}\footnotesize")
A(r"$^{\dagger}$Our re-implementation departs from the published method: "
  r"FuGuard's proxy is synthesised in embedding space because its ConvVAE "
  r"cannot encode text. A dash marks a configuration that did not converge in "
  r"our harness; those are artefacts of our re-implementation and must not be "
  r"read as the method's performance (see Conclusion). "
  + r"FedRecover also saves "
  + "/".join("%.0f" % R.MAIN[d]["FedRecover"]["cost"] for d in DS)
  + r"\% of client compute. Ours: composed $\varepsilon$ = "
  + "/".join("%.1f" % R.MAIN[d]["AdaptFU"]["eps"] for d in DS)
  + r". Standard-FL MIA differs by dataset ("
  + "/".join("%.3f" % R.MAIN[d]["Standard FL"]["mia"] for d in DS)
  + r"), so residual gaps are not comparable across columns.")
A(r"\end{flushleft}\end{table*}")
A("")

# ---------------- Table 2: robustness ----------------------------------------
A(r"\begin{table}[t]\centering\small")
A(r"\setlength{\tabcolsep}{2.5pt}")
A(r"\caption{AdaptFU under Byzantine calibration updates (20\% malicious). "
  r"The verification is designed for the norm-inflating tier; the norm-matched "
  r"tier evades it by construction.}")
A(r"\label{tab:robust}")
A(r"\begin{tabular}{l ccc}")
A(r"\toprule")
A(r"Aggregation & No attack & Norm-infl. & Norm-matched \\")
A(r"\midrule")
A(r"\multicolumn{4}{l}{\emph{CIFAR-100}} \\")
for agg, row in R.ROBUST_C100.items():
    cs = [f"{row['none'][0]:.3f}", f"{row['noise'][0]:.3f}",
          f"{row['minmax'][0]:.3f}~({row['minmax'][1]}/9)"]
    A(f"\\quad {agg} & " + " & ".join(cs) + r" \\")
A(r"\midrule")
A(r"\multicolumn{4}{l}{\emph{Other modalities (FedAvg)}} \\")
for ds, row in R.ROBUST_XDATA.items():
    cs = [f"{row['none'][0]:.3f}", f"{row['noise'][0]:.3f}",
          (f"{row['minmax'][0]:.3f}~({row['minmax'][1]}/9)"
           if row['minmax'][0] is not None else "---")]
    A(f"\\quad {ds} & " + " & ".join(cs) + r" \\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\begin{flushleft}\footnotesize")
A(r"Survivors are $9/9$ with no attack and $7/9$ under norm inflation in every "
  r"cell: the verification rejects exactly the 2 malicious clients, at $<$1\,pp "
  r"utility cost on every rule and modality. Norm-matched survivors are shown "
  r"per cell; the occasional $8/9$ is a \emph{false} rejection, since placing "
  r"mass at the benign median shrinks the MAD ($48.13\rightarrow42.92$) and "
  r"tightens the threshold onto an honest outlier.")
A(r"\end{flushleft}\end{table}")
A("")

# ---------------- Table 3: DP ablation ---------------------------------------
d = R.DP_ABLATION_C100
delta = [100 * (pc - pr) for _, pc, pr in d]
m, sd = float(np.mean(delta)), float(np.std(delta, ddof=1))
A(r"\begin{table}[t]\centering\small")
A(r"\setlength{\tabcolsep}{4pt}")
A(r"\caption{Per-client vs.\ per-round noise allocation at \emph{matched} privacy "
  r"budget. Paired design: identical base checkpoint and identical noise draws "
  r"within each seed, so only the allocation axis differs. Both modes share the same "
  r"round-mean $z$ and therefore the same composed $\varepsilon$ by construction.}")
A(r"\label{tab:dp}")
A(r"\begin{tabular}{@{}l ccc@{}}")
A(r"\toprule")
A(r"Setting & Per-client & Per-round & $\Delta$ (pp) \\")
A(r"\midrule")
pc_mean = float(np.mean([pc for _, pc, _ in d]))
pr_mean = float(np.mean([pr for _, _, pr in d]))
A(rf"CIFAR-100 & {pc_mean:.4f} & {pr_mean:.4f} & "
  rf"$\mathbf{{{m:+.2f} \pm {sd:.2f}}}$ \\")
A(r"\midrule")
pc, pr = R.DP_ABLATION_FEMNIST
A(f"FEMNIST & {pc:.4f} & {pr:.4f} & ${100*(pc-pr):+.2f}$ \\\\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\begin{flushleft}\footnotesize")
A(rf"CIFAR-100 is 4 paired noise seeds (paired $t{{=}}{m/(sd/2):.2f}$, df${{=}}3$); "
  r"FEMNIST is a single pair at $\alpha{=}0.5$. No measurable effect, and MIA "
  r"identical to four decimals on FEMNIST (0.5066 both). Per-seed scatter is "
  r"$\pm0.5$\,pp, so a single seed cannot resolve an effect below "
  r"$\approx\pm1$\,pp.")
A(r"\end{flushleft}\end{table}")
A("")

# ---------------- Table 4: mechanism findings --------------------------------
A(r"\begin{table}[t]\centering")
A(r"\caption{Mechanism diagnostics. Each row is a measurement, not a citation.}")
A(r"\label{tab:mech}")
A(r"\begin{tabular}{@{}p{0.27\linewidth} p{0.66\linewidth}@{}}")
A(r"\toprule")
A(r"Finding & Measurement \\")
A(r"\midrule")
A(r"Eq.~9 cancels $z$ at high dimension & Noise of per-coordinate std $z b$ has "
  r"$\ell_2$ norm $zb\sqrt{d}\approx 4000\times$ the update; $\Pi_b$ renormalises it, "
  rf"so outputs at $z\in\{{0.5,1,1.5\}}$ have cosine $1.000$ and differ by "
  rf"${R.EQ9['rel_model_diff_coord']:.1e}$ relatively. Magnitude calibration "
  rf"restores a ${R.EQ9['rel_model_diff_norm']:.1e}$ separation ($180\times$). \\")
A(r"\addlinespace")
A(r"Pooled $\rho$ percentiles collapse the allocation & Within-round sd of "
  rf"per-client $z$: {np.mean(R.RHO_SCOPE['pooled']):.3f} (pooled) vs.\ "
  rf"{np.mean(R.RHO_SCOPE['round']):.3f} (round-local). Under an LR schedule the "
  r"pooled range is dominated by across-round decay, so every client receives "
  r"near-identical noise. \\")
A(r"\addlinespace")
A(r"FedADP requires reproducible local updates & $\|U\|=|\cos\theta|\cdot\|\bar g\|$, "
  rf"and $\cos$ between two runs from an identical start falls from "
  rf"{R.FEDADP_LR[0][1]:.3f} at lr$=${R.FEDADP_LR[0][0]} to "
  rf"{R.FEDADP_LR[3][1]:.3f} at lr$=${R.FEDADP_LR[3][0]}. FedEraser, identical but "
  r"without the $\cos$ factor, is unaffected. \\")
A(r"\addlinespace")
A(r"Norm-matched attacks must be verified as such & A na\"ive min-max lands at "
  rf"{R.MINMAX['old_ratio']:.2f}$\times$ the benign median norm and is rejected; "
  r"enforcing the norm match gives "
  rf"{R.MINMAX['fixed_ratio']:.2f}$\times$ and evades. Reporting evasion without "
  r"checking survivor counts inverts the conclusion. \\")
A(r"\bottomrule")
A(r"\end{tabular}\end{table}")

open("tables.tex", "w").write("\n".join(out) + "\n")
print(f"wrote tables.tex ({len(out)} lines, 4 tables)")
