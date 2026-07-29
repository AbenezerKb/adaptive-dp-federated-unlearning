"""Visualize the logged runs.

Reads logs/<run_id>.jsonl (schema in logging_utils.py) and renders, per run:
  1. Unlearning curves: test_acc / forget_acc / mia_auc / (asr) vs round, one
     line per method  -- the paper-style comparison figure.
  2. FL training curves: Standard FL and Retrain test_acc vs round.
  3. AdaptFU robustness: survivors kept/total vs round per attack (if logged).
  4. FuGuard unlearning losses: cls_loss and ot_loss vs epoch.
  5. Final bar chart: test_acc / forget_acc / mia_auc per method (result rows).

Saved as PNGs under --out-dir. Pure matplotlib; no seaborn.
"""
import argparse, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .logging_utils import load_dir


def _series(events, event_type, key_x, key_y, group_key="method"):
    """Return {group: ([x...],[y...])} for events of a type with x,y present."""
    out = defaultdict(lambda: ([], []))
    for e in events:
        if e.get("event") == event_type and key_x in e and key_y in e:
            g = e.get(group_key, "?")
            out[g][0].append(e[key_x]); out[g][1].append(e[key_y])
    return out


def plot_unlearning_curves(run_id, events, out_dir):
    metrics = ["test_acc", "forget_acc", "mia_auc"]
    has_asr = any(e.get("event") == "eval" and "asr" in e for e in events)
    if has_asr:
        metrics.append("asr")
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        series = _series(events, "eval", "round", m)
        for method, (x, y) in sorted(series.items()):
            xy = sorted(zip(x, y))
            if xy:
                xs, ys = zip(*xy)
                ax.plot(xs, ys, marker="o", ms=3, lw=1.3, label=method)
        ax.set_title(m); ax.set_xlabel("round"); ax.grid(alpha=0.3)
        better = "lower better" if m in ("forget_acc", "mia_auc", "asr") else "higher better"
        ax.set_ylabel(f"{m}  ({better})")
    axes[0][-1].legend(fontsize=6, loc="best")
    fig.suptitle(f"{run_id}: unlearning curves")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run_id}_curves.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_fl_curves(run_id, events, out_dir):
    series = _series(events, "fl_eval", "round", "test_acc", group_key="phase")
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    for phase, (x, y) in series.items():
        xy = sorted(zip(x, y))
        xs, ys = zip(*xy)
        ax.plot(xs, ys, marker="o", ms=3, label=phase)
    ax.set_xlabel("round"); ax.set_ylabel("test_acc"); ax.grid(alpha=0.3)
    ax.legend(); ax.set_title(f"{run_id}: FL training")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run_id}_fl.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_survivors(run_id, events, out_dir):
    data = defaultdict(lambda: ([], []))
    for e in events:
        if e.get("event") == "method_round" and "survivors_kept" in e:
            m = e.get("method", "?")
            data[m][0].append(e["round"])
            data[m][1].append(e["survivors_kept"])
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, (x, y) in sorted(data.items()):
        xy = sorted(zip(x, y))
        xs, ys = zip(*xy)
        ax.plot(xs, ys, marker="s", ms=3, label=method)
    ax.set_xlabel("round"); ax.set_ylabel("survivors kept")
    ax.grid(alpha=0.3); ax.legend(fontsize=6)
    ax.set_title(f"{run_id}: AdaptFU verification survivors")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run_id}_survivors.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_fuguard_losses(run_id, events, out_dir):
    cls, ot = ([], []), ([], [])
    for e in events:
        if e.get("event") == "method_epoch" and e.get("method") == "FuGuard":
            cls[0].append(e["epoch"]); cls[1].append(e.get("cls_loss"))
            ot[0].append(e["epoch"]); ot[1].append(e.get("ot_loss"))
    if not cls[0]:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cls[0], cls[1], marker="o", label="cls_loss (ascent target)")
    ax.plot(ot[0], ot[1], marker="s", label="ot_loss (drift penalty)")
    ax.set_xlabel("unlearn epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
    ax.legend(); ax.set_title(f"{run_id}: FuGuard unlearning losses")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run_id}_fuguard_loss.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_final_bars(run_id, events, out_dir):
    rows = [e for e in events if e.get("event") == "result_row"]
    if not rows:
        return None
    methods = [r["method"] for r in rows]
    metrics = ["test_acc", "forget_acc", "mia_auc"]
    if any("asr" in r for r in rows):
        metrics.append("asr")
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.5),
                             squeeze=False)
    for ax, m in zip(axes[0], metrics):
        vals = [r.get(m, float("nan")) for r in rows]
        ax.bar(range(len(methods)), vals)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=6)
        better = "lower better" if m in ("forget_acc", "mia_auc", "asr") else "higher better"
        ax.set_title(f"{m}\n({better})"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{run_id}: final metrics")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run_id}_final_bars.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="./logs")
    ap.add_argument("--out-dir", default="./figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_dir(args.log_dir)
    if not runs:
        print(f"No .jsonl logs found in {args.log_dir}")
        return
    made = []
    for run_id, events in runs.items():
        for fn in (plot_unlearning_curves, plot_fl_curves, plot_survivors,
                   plot_fuguard_losses, plot_final_bars):
            try:
                p = fn(run_id, events, args.out_dir)
                if p:
                    made.append(p)
            except Exception as ex:
                print(f"  [{run_id}] {fn.__name__} skipped: {ex}")
    print(f"Wrote {len(made)} figures to {args.out_dir}/")
    for p in made:
        print("  " + p)


if __name__ == "__main__":
    main()
