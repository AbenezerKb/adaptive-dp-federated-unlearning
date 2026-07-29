"""Run one data-config end-to-end with full step logging.

Key efficiency point: Standard FL + Retrain + the four non-AdaptFU methods do NOT
depend on the aggregation rule or the unlearning-phase attack, so we train the
base ONCE and reuse it across the whole AdaptFU (aggregation x attack) grid. This
is the base-caching lever: it collapses the 13 MNIST+CIFAR-10 invocations into 4
distinct data-configs (see dispatch.py).

Everything is logged to logs/<run_id>.jsonl for later visualization, and a
compact summary is written to results/<run_id>.json.

Usage (normally invoked by dispatch.py):
  python -m afu_vs_fg.run_experiment --run-id cifar10_f0 --dataset cifar10 \
      --forget 0 --adaptfu-agg fedavg krum trimmed_mean median \
      --adaptfu-attack none noise minmax --eval-every 2
"""
import argparse, copy, json, os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from .config import preset
from .data import (get_datasets, get_eval_train, dirichlet_partition,
                   natural_partition, make_client_loaders,
                   make_backdoor_testset, DATASET_META)
from .models import build_classifier, ConvVAE
from .utils import set_seed
from .fl_train import federated_train
from .adaptfu import run_adaptfu, scale_z_for_target_eps
from .fuguard import run_fuguard, pretrain_vae
from .baselines import (run_federaser, run_fedrecover, run_fedadp,
                        scale_fedadp_for_target_eps)
from .metrics import (accuracy, backdoor_asr, train_mia_attacker, mia_auc,
                      rdp_epsilon)
from .logging_utils import RunLogger


def build_eval_loaders(train_ds, test_ds, partition, forget, bs, curve_n=2000,
                       eval_n=5000, seed=0):
    """Evaluation loaders, all CAPPED at `eval_n` samples.

    The cap matters: on a large corpus (EMNIST/byclass is ~698k train) a single
    client holds ~70k samples, so an uncapped MIA would do ~140k forward passes
    on EVERY per-round eval, for every method -- which dominates total runtime
    and scales with dataset size rather than with anything scientific. 5k samples
    give MIA AUC to about +/-0.01 and accuracy to +/-0.7%, which is far below the
    effect sizes being compared."""
    rng = np.random.default_rng(seed)

    def cap(idx, n):
        idx = np.asarray(idx)
        if len(idx) <= n:
            return idx
        return rng.choice(idx, n, replace=False)

    fidx = np.concatenate([partition[k] for k in forget])
    forget_loader = DataLoader(Subset(train_ds, cap(fidx, eval_n)), batch_size=bs)
    retain = [k for k in range(len(partition)) if k not in forget]
    midx = np.concatenate([partition[k] for k in retain])[:len(fidx)]
    member_loader = DataLoader(Subset(train_ds, cap(midx, eval_n)), batch_size=bs)
    # non-members: same count as members so the MIA classes stay balanced
    n_non = min(eval_n, len(fidx), len(test_ds))
    nonmember_loader = DataLoader(Subset(test_ds, list(range(n_non))), batch_size=bs)
    test_loader = DataLoader(Subset(test_ds, cap(np.arange(len(test_ds)),
                                                 max(eval_n, curve_n) * 4)),
                             batch_size=bs)
    curve_loader = DataLoader(Subset(test_ds, list(range(min(curve_n, len(test_ds))))),
                              batch_size=bs)
    return dict(forget=forget_loader, member=member_loader,
                nonmember=nonmember_loader, test=test_loader, curve=curve_loader)


def _ckpt_paths(ckpt_dir, run_id):
    return (os.path.join(ckpt_dir, f"{run_id}_ckpt.pt"),
            os.path.join(ckpt_dir, f"{run_id}_progress.json"))


def save_heavy_ckpt(path, base, hist, attacker, fp16=True):
    """Save the expensive shared state (base weights + full history + MIA
    attacker) ONCE. History is the big item (~35GB fp32 on CIFAR-100); fp16
    halves it. base/hist/attacker never change after Standard FL, so this is
    written a single time before the risky unlearning methods run."""
    conv = (lambda v: v.half()) if fp16 else (lambda v: v)
    payload = dict(
        base=base.state_dict(),
        histG=[conv(g) for g in hist["G"]],
        histC={t: {k: conv(c) for k, c in cd.items()} for t, cd in hist["C"].items()},
        sizes=hist["sizes"], attacker=attacker, fp16=fp16)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)     # atomic: a crash mid-write never corrupts the ckpt


def load_heavy_ckpt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    back = (lambda v: v.float()) if d.get("fp16") else (lambda v: v)
    hist = dict(G=[back(g) for g in d["histG"]],
                C={t: {k: back(c) for k, c in cd.items()}
                   for t, cd in d["histC"].items()},
                sizes=d["sizes"])
    return d["base"], hist, d["attacker"]


def save_progress(path, rows, done):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"rows": rows, "done": sorted(done)}, f, indent=2, default=str)
    os.replace(tmp, path)


def load_progress(path):
    if os.path.exists(path):
        d = json.load(open(path))
        return d.get("rows", []), set(d.get("done", []))
    return [], set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset", required=True, choices=["mnist", "femnist", "cifar10", "cifar100", "agnews"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--forget", type=int, nargs="+", default=[0])
    ap.add_argument("--backdoor", action="store_true")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--adaptfu-agg", nargs="+", default=["fedavg"],
                    choices=["fedavg", "krum", "trimmed_mean", "median"])
    ap.add_argument("--adaptfu-attack", nargs="+", default=["none"],
                    choices=["none", "noise", "minmax"])
    ap.add_argument("--eval-every", type=int, default=2,
                    help="per-round eval cadence for curves (test/forget/mia/asr)")
    ap.add_argument("--adaptfu-rho-scope", default=None, choices=["round","pooled"],
                    help="scope of rho percentiles in the reliability score. "
                         "'pooled' spans all replay rounds and degenerates to "
                         "near-uniform z when update norms decay across rounds; "
                         "'round' uses the current round's norms (default).")
    ap.add_argument("--adaptfu-noise-mode", default=None, choices=["coord","norm"],
                    help="'coord' = eq.9 literal (per-coordinate std z*b). At high "
                         "dimension Pi_b renormalises the noise and z CANCELS, so "
                         "per-client allocation is inert. 'norm' calibrates the "
                         "noise MAGNITUDE (E||noise||=z*b) so allocation works -- "
                         "but dp_epsilon is then NOT a valid (eps,delta) guarantee.")
    ap.add_argument("--adaptfu-agg-weight", default=None,
                    choices=["samples","reliability"],
                    help="eq.10 weights by sample count; 'reliability' uses n_k*r_k "
                         "so outlier clients are down-weighted in the aggregate (the "
                         "paper's prose suggests this; its algorithm does not).")
    ap.add_argument("--adaptfu-clip-order", default=None, choices=["after","before"],
                    help="'after' = eq.9 Pi_b(u+noise) (renormalises away part of z's "
                         "effect); 'before' = standard DP-SGD order, z's effect intact.")
    ap.add_argument("--adaptfu-dp-mode", default=None,
                    choices=["per_client","per_round","static",
                             "adaptive","uniform"],
                    help="per_client = z differs between clients within a round "
                         "(AdaptFU). per_round = z same across clients but MOVES "
                         "ACROSS ROUNDS (FedADP's shape -- adaptive on the round "
                         "axis, not static). Identical epsilon to per_client, so "
                         "the pair isolates the client axis. Aliases: "
                         "adaptive=per_client, uniform=per_round.")
    ap.add_argument("--adaptfu-z-min", type=float, default=None,
                    help="noise multiplier for the MOST reliable client (r->1)")
    ap.add_argument("--adaptfu-z-max", type=float, default=None,
                    help="noise multiplier for the LEAST reliable client (r->0)")
    ap.add_argument("--adaptfu-warm-start", type=float, default=None,
                    help="warm-start fraction w; reconstruction begins at round wT")
    ap.add_argument("--adaptfu-calib-epochs", type=int, default=None,
                    help="E_c local epochs per calibration round")
    ap.add_argument("--adaptfu-target-eps", type=float, default=None,
                    help="rescale AdaptFU's z_min/z_max so its composed epsilon "
                         "equals this. Use FedADP's epsilon to make the "
                         "per-client-vs-uniform DP comparison budget-matched, "
                         "which is what the paper's claim actually asserts.")
    ap.add_argument("--fedadp-init", default=None, choices=["first_stored","warm"],
                    help="'first_stored' = Alg.2 line1 (M_hat_0 <- M_bar_0), faithful. "
                         "'warm' = start at AdaptFU's warm-start round (NOT FedADP; "
                         "lends it AdaptFU's utility/forgetting operating point).")
    ap.add_argument("--fedadp-select-mode", default=None, choices=["staged","global"],
                    help="'staged' = Alg.1 per-stage selection (loss-drop triggered, "
                         "T1 emptied each time) so stored models SPAN training. "
                         "'global' ranks all rounds at once and skews to early "
                         "rounds, leaving the late trajectory unreplayable.")
    ap.add_argument("--fedadp-dual-select", type=int, default=None,
                    help="1 = Alg.1 dual-layered selection (keep lambda=60%% of "
                         "models, gamma=70%% of updates); 0 = replay all history.")
    ap.add_argument("--fedadp-adaptive-eps", type=int, default=None,
                    help="1 = FedADP eq.4 loss-driven per-ROUND budget (its actual "
                         "mechanism); 0 = fixed eps (static DP, NOT FedADP).")
    ap.add_argument("--fedadp-target-eps", type=float, default=None,
                    help="set FedADP's composed epsilon to this. Pair with "
                         "--adaptfu-target-eps at the same value to sweep both "
                         "methods at genuinely equal privacy budgets.")
    ap.add_argument("--lr", type=float, default=None,
                    help="override peak learning rate (per-dataset default in config)")
    ap.add_argument("--lr-min", type=float, default=None,
                    help="cosine floor. Default 1e-3 spends the last ~20%% of rounds "
                         "below lr=0.01 where CIFAR-100 gains flatline; 1e-2 keeps "
                         "the tail in the productive band.")
    ap.add_argument("--lr-schedule", default=None, choices=["cosine","constant"],
                    help="LR decays across ROUNDS. Note this is open-loop (indexed "
                         "by round), not plateau-triggered.")
    ap.add_argument("--eval-n", type=int, default=5000,
                    help="cap on forget/member/nonmember eval subsets. Uncapped, "
                         "MIA cost scales with dataset size and dominates runtime "
                         "on large corpora like EMNIST/byclass.")
    ap.add_argument("--subsample", type=int, default=0,
                    help="randomly subsample the TRAIN set to N examples "
                         "(0 = off). EMNIST/byclass is ~698k, 11.6x MNIST.")
    ap.add_argument("--augment", type=int, default=1,
                    help="CIFAR train-time RandomCrop+HFlip (default on). "
                         "Off caps ResNet-18 near 0.55 on CIFAR-100.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="reseed ONLY the unlearning phase, leaving the data "
                         "partition and base model untouched. Lets you vary the DP "
                         "noise realisation across runs (with --resume) to test "
                         "whether an adaptive-vs-uniform gap is stable, without "
                         "changing --seed (which would repartition and invalidate "
                         "the checkpoint).")
    ap.add_argument("--log-dir", default="./logs")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--standardfl-only", action="store_true",
                    help="SMOKE: train only Standard FL, report final test acc, exit")
    ap.add_argument("--min-base-acc", type=float, default=0.0,
                    help="abort before the (slow) unlearning methods if Standard FL "
                         "test acc is below this (guards against a broken base)")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="if set, save base+history+attacker once and per-method "
                         "progress, so a re-run resumes instead of recomputing")
    ap.add_argument("--resume", action="store_true",
                    help="reload checkpoint (if present) and skip finished methods")
    ap.add_argument("--ckpt-fp16", type=int, default=1,
                    help="store history in fp16 to halve checkpoint size (default on)")
    ap.add_argument("--only-methods", default=None,
                    help="comma-separated methods to run (e.g. 'FedADP,AdaptFU/fedavg'). "
                         "Standard FL always runs since base+history are required. "
                         "Use with --keep-ckpt so later additions can resume.")
    ap.add_argument("--fedadp-dp", type=int, default=None,
                    help="override FedADP DP: 1 = per-round uniform DP noise (the "
                         "apples-to-apples contrast with AdaptFU's per-client DP), "
                         "0 = calibration only. Default: config value.")
    ap.add_argument("--ckpt-run-id", default=None,
                    help="load the checkpoint saved under a DIFFERENT run-id. Lets "
                         "many configs share one base without copying it -- a "
                         "CIFAR-100 checkpoint is ~18GB, so copying it per config "
                         "fills the disk fast.")
    ap.add_argument("--keep-ckpt", action="store_true",
                    help="keep the heavy resume checkpoint after the config finishes "
                         "(default: delete it once done, keeping only base+retrain .pt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    cfg = preset(args.dataset)
    cfg.data.dirichlet_alpha = args.alpha
    cfg.forget.forget_clients = args.forget
    cfg.seed = args.seed
    if args.rounds:
        cfg.fl.rounds = args.rounds
    if args.lr is not None:
        cfg.fl.lr = args.lr
    if args.lr_min is not None:
        cfg.fl.lr_min = args.lr_min
    if args.lr_schedule:
        cfg.fl.lr_schedule = args.lr_schedule
    if args.backdoor:
        cfg.attack.backdoor = True
    if args.fedadp_dp is not None:
        cfg.fedadp.apply_dp = bool(args.fedadp_dp)
    if args.fedadp_adaptive_eps is not None:
        cfg.fedadp.adaptive_eps = bool(args.fedadp_adaptive_eps)
    if args.fedadp_init:
        cfg.fedadp.init_mode = args.fedadp_init
    if args.fedadp_dual_select is not None:
        cfg.fedadp.dual_select = bool(args.fedadp_dual_select)
    if args.fedadp_select_mode:
        cfg.fedadp.select_mode = args.fedadp_select_mode
    if args.adaptfu_z_min is not None:
        cfg.adaptfu.z_min = args.adaptfu_z_min
    if args.adaptfu_z_max is not None:
        cfg.adaptfu.z_max = args.adaptfu_z_max
    if args.adaptfu_warm_start is not None:
        cfg.adaptfu.warm_start_frac = args.adaptfu_warm_start
    if args.adaptfu_calib_epochs is not None:
        cfg.adaptfu.calib_epochs = args.adaptfu_calib_epochs
    if args.adaptfu_rho_scope:
        cfg.adaptfu.rho_scope = args.adaptfu_rho_scope
    if args.adaptfu_dp_mode:
        cfg.adaptfu.dp_mode = args.adaptfu_dp_mode
    if args.adaptfu_agg_weight:
        cfg.adaptfu.agg_weight = args.adaptfu_agg_weight
    if args.adaptfu_clip_order:
        cfg.adaptfu.clip_order = args.adaptfu_clip_order
    if args.adaptfu_noise_mode:
        cfg.adaptfu.noise_mode = args.adaptfu_noise_mode
        cfg.fedadp.noise_mode = args.adaptfu_noise_mode
    if getattr(cfg.adaptfu, "noise_mode", "coord") == "norm":
        print(f"[{args.run_id}] NOTE: noise_mode=norm -- noise magnitude is "
              f"calibrated (E||noise||=z*b), NOT per-coordinate std. The per-client "
              f"allocation is functional, but the reported dp_epsilon is a relative "
              f"reference only and is NOT a valid (eps,delta) guarantee.")
    meta = DATASET_META[args.dataset]
    os.makedirs(args.out_dir, exist_ok=True)

    # decide resume BEFORE opening the logger: on resume we must APPEND, or we
    # would truncate the earlier run's fl_round/eval curves needed for plots
    _will_resume = bool(args.resume and args.checkpoint_dir and os.path.exists(
        _ckpt_paths(args.checkpoint_dir, args.run_id)[0]))

    logger = RunLogger(args.log_dir, args.run_id, append=_will_resume)
    logger.log("run_start", dataset=args.dataset, alpha=args.alpha,
               forget=args.forget, backdoor=args.backdoor, rounds=cfg.fl.rounds,
               adaptfu_agg=args.adaptfu_agg, adaptfu_attack=args.adaptfu_attack,
               eval_every=args.eval_every, seed=args.seed, device=device,
               resumed=_will_resume)

    # ---- data ----
    train_ds, test_ds = get_datasets(args.dataset, cfg.data.data_root,
                                     augment=bool(args.augment))
    # clean (unaugmented) train view for forget-acc + MIA members
    train_eval = (get_eval_train(args.dataset, cfg.data.data_root)
                  if args.augment else train_ds)

    # Optional train subsample. EMNIST/byclass (the FEMNIST fallback) is ~698k
    # examples; at 50 rounds x 5 local epochs that is ~175M sample-passes, far
    # beyond a Kaggle session. Subsampling keeps the comparison valid (every
    # method sees the identical reduced corpus) while making runtime feasible.
    if args.subsample and args.subsample < len(train_ds):
        sub = np.random.default_rng(args.seed).choice(
            len(train_ds), args.subsample, replace=False)
        sub.sort()
        keep = set(sub.tolist())
        train_ds = Subset(train_ds, sub)
        train_eval = Subset(train_eval, sub)
        # targets/writers must follow the subsample for partitioning
        base_t = np.asarray(getattr(train_ds.dataset, "targets", []))
        if len(base_t):
            train_ds.targets = base_t[sub]
        base_w = getattr(train_ds.dataset, "writers", None)
        if base_w is not None:
            train_ds.writers = np.asarray(base_w)[sub]
        print(f"[{args.run_id}] subsampled train set -> {len(train_ds)} examples")
    # FEMNIST: natural per-writer split (its defining non-IID structure).
    # Falls back to Dirichlet if only EMNIST/byclass (no writer ids) is available.
    if args.dataset == "femnist" and hasattr(train_ds, "writers"):
        partition = natural_partition(train_ds, cfg.data.n_clients, seed=args.seed)
        print(f"[{args.run_id}] FEMNIST natural writer partition: "
              f"{[len(p) for p in partition]}")
    else:
        partition = dirichlet_partition(train_ds, cfg.data.n_clients, args.alpha,
                                        meta["n_classes"], seed=args.seed)
    bd_client = args.forget[0] if args.backdoor else None
    client_loaders = make_client_loaders(train_ds, partition, cfg.fl.batch_size,
                                         backdoor_client=bd_client,
                                         attack_cfg=cfg.attack, seed=args.seed)
    for cl in client_loaders:
        cl.num_workers = args.num_workers
    bd_loader = (make_backdoor_testset(test_ds, cfg.attack.backdoor_target,
                                       cfg.fl.batch_size) if args.backdoor else None)
    L = build_eval_loaders(train_eval, test_ds, partition, args.forget,
                           cfg.fl.batch_size, eval_n=args.eval_n, seed=args.seed)

    def eval_full(model, loader):
        return accuracy(model, loader, device)

    ckpt_pt = prog_json = None
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # heavy ckpt may live under another run-id (shared base, no copying);
        # progress stays per-run so each config tracks its own finished methods
        src = args.ckpt_run_id or args.run_id
        ckpt_pt = _ckpt_paths(args.checkpoint_dir, src)[0]
        prog_json = _ckpt_paths(args.checkpoint_dir, args.run_id)[1]

    resumed = args.resume and ckpt_pt and os.path.exists(ckpt_pt)

    # ---- Standard FL (shared base) : train, or reload from checkpoint ----
    base = build_classifier(args.dataset, meta).to(device)
    if resumed:
        print(f"[{args.run_id}] RESUME: loading base + history + attacker from ckpt")
        base_sd, hist, attacker = load_heavy_ckpt(ckpt_pt)
        base.load_state_dict(base_sd)
    else:
        print(f"[{args.run_id}] Standard FL")
        base, hist = federated_train(base, client_loaders, cfg.fl, device,
                                     cache_history=True, logger=logger, phase="standardfl",
                                     eval_fn=lambda m: eval_full(m, L["curve"]),
                                     eval_every=max(1, args.eval_every))
        attacker = train_mia_attacker(base, L["member"], L["nonmember"], device)

    # eval hook for unlearning methods (curves): full metric panel at a cadence
    def eval_hook(tag, rnd, model):
        if args.eval_every and (rnd % args.eval_every != 0):
            return
        rec = dict(method=tag, round=rnd,
                   test_acc=round(eval_full(model, L["curve"]), 4),
                   forget_acc=round(eval_full(model, L["forget"]), 4),
                   mia_auc=round(mia_auc(attacker, model, L["forget"],
                                         L["nonmember"], device), 4))
        if bd_loader is not None:
            rec["asr"] = round(backdoor_asr(model, bd_loader,
                                            cfg.attack.backdoor_target, device), 4)
        logger.log("eval", **rec)

    # reseed just before the unlearning methods: base/partition already fixed
    if args.noise_seed is not None:
        set_seed(args.noise_seed)
        logger.log("noise_reseed", noise_seed=args.noise_seed)
        print(f"[{args.run_id}] unlearning-phase noise seed = {args.noise_seed}")

    rows, done = ([], set())
    if resumed and prog_json:
        rows, done = load_progress(prog_json)
        print(f"[{args.run_id}] RESUME: {len(done)} methods already done: {sorted(done)}")

    def final_row(tag, model, extra=None):
        r = dict(method=tag,
                 test_acc=round(eval_full(model, L["test"]), 4),
                 forget_acc=round(eval_full(model, L["forget"]), 4),
                 mia_auc=round(mia_auc(attacker, model, L["forget"],
                                       L["nonmember"], device), 4))
        if bd_loader is not None:
            r["asr"] = round(backdoor_asr(model, bd_loader,
                                          cfg.attack.backdoor_target, device), 4)
        if extra:
            r.update(extra)
        rows.append(r)
        logger.log("result_row", **r)
        done.add(tag)
        if prog_json:
            save_progress(prog_json, rows, done)
        return r

    only = None
    if args.only_methods:
        only = {m.strip() for m in args.only_methods.split(",") if m.strip()}

    matched = set()

    def skip(tag):
        if only is not None and tag != "Standard FL":
            # prefix match so a bare "AdaptFU" selects AdaptFU/<agg>[|attack].
            # The tag is f"AdaptFU/{agg}", so --only-methods "AdaptFU/fedavg"
            # silently matches NOTHING when --adaptfu-agg is krum/median.
            hit = tag in only or any(tag.startswith(o) for o in only)
            if not hit:
                return True                  # not requested this pass
            matched.add(tag)
        if tag in done:
            print(f"[{args.run_id}] skip {tag} (already in checkpoint)")
            return True
        return False

    def free():
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not skip("Standard FL"):
        base_row = final_row("Standard FL", base)
        base_acc = base_row["test_acc"]
    else:
        base_acc = next((r["test_acc"] for r in rows if r["method"] == "Standard FL"),
                        1.0)

    # ---- save the heavy checkpoint ONCE (base + history + attacker) so any
    #      crash in the methods below can resume without recomputing them ----
    if ckpt_pt and not resumed:
        print(f"[{args.run_id}] saving checkpoint (base+history+attacker) ...")
        save_heavy_ckpt(ckpt_pt, base, hist, attacker, fp16=bool(args.ckpt_fp16))

    # ---- SMOKE PATH: stop after Standard FL (checkpoint already written, so a
    #      smoke run doubles as a reusable shared base for later configs) ----
    if args.standardfl_only:
        logger.close()
        ok = base_acc >= (args.min_base_acc or 0.0)
        print("\n" + "=" * 60)
        print(f"SMOKE [{args.run_id}]  Standard FL test_acc = {base_acc:.4f}")
        if args.min_base_acc:
            print(f"  threshold {args.min_base_acc:.2f} -> "
                  f"{'PASS, base learns' if ok else 'FAIL, base is broken'}")
        print("=" * 60)
        return

    # ---- sanity gate: don't burn hours of unlearning on a broken base ----
    if args.min_base_acc and base_acc < args.min_base_acc:
        logger.log("aborted", reason="base_below_threshold",
                   base_acc=base_acc, threshold=args.min_base_acc)
        logger.close()
        print(f"\n[ABORT] Standard FL test_acc {base_acc:.4f} < "
              f"--min-base-acc {args.min_base_acc}. Base did not learn; "
              f"skipping unlearning methods. Check lr / data / model.")
        return

    # durable model artifact: Standard FL base (kept even after cleanup)
    if args.checkpoint_dir and not resumed:
        torch.save(base.state_dict(),
                   os.path.join(args.checkpoint_dir, f"{args.run_id}_standardfl.pt"))

    # ---- FL Retrain (gold standard) ----
    if not skip("FL Retrain"):
        print(f"[{args.run_id}] FL Retrain")
        retr = build_classifier(args.dataset, meta).to(device)
        logger.log("method_start", method="FL Retrain", variant="retrain")
        retr, _ = federated_train(retr, client_loaders, cfg.fl, device,
                                  exclude=set(args.forget), cache_history=False,
                                  logger=logger, phase="retrain",
                                  eval_fn=lambda m: eval_full(m, L["curve"]),
                                  eval_every=max(1, args.eval_every))
        final_row("FL Retrain", retr)
        if args.checkpoint_dir:      # durable model artifact: gold-standard retrain
            torch.save(retr.state_dict(),
                       os.path.join(args.checkpoint_dir, f"{args.run_id}_retrain.pt"))
        del retr; free()

    # ---- FuGuard (base-independent of agg/attack) ----
    if not skip("FuGuard"):
        print(f"[{args.run_id}] FuGuard")
        logger.log("method_start", method="FuGuard", variant="benign")
        if meta.get("channels") is None:          # TEXT: no ConvVAE; proxy is
            vae = None                            # synthesised in embedding space
        else:
            vae = ConvVAE(meta["channels"], meta["img"], cfg.fuguard.vae_latent)
            full_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                                     num_workers=args.num_workers)
            vae = pretrain_vae(vae, full_loader,
                               cfg.fuguard.vae_pretrain_epochs, device)
        fg_model, _ = run_fuguard(base, vae, client_loaders, cfg, device,
                                  logger=logger, eval_hook=eval_hook)
        final_row("FuGuard", fg_model)
        del fg_model, vae; free()

    # ---- FedEraser ----
    if not skip("FedEraser"):
        print(f"[{args.run_id}] FedEraser")
        logger.log("method_start", method="FedEraser", variant="fedavg")
        fe_model, _ = run_federaser(base, hist, client_loaders, cfg, device,
                                    logger=logger, eval_hook=eval_hook)
        final_row("FedEraser", fe_model)
        del fe_model; free()

    # ---- FedRecover ----
    if not skip("FedRecover"):
        print(f"[{args.run_id}] FedRecover")
        logger.log("method_start", method="FedRecover", variant="fedavg")
        fr_model, fr_info = run_fedrecover(base, hist, client_loaders, cfg, device,
                                           logger=logger, eval_hook=eval_hook)
        final_row("FedRecover", fr_model, {"cost_saving": fr_info["cost_saving_pct"]})
        del fr_model; free()

    # ---- FedADP ----
    if args.fedadp_target_eps and not skip("FedADP"):
        e = scale_fedadp_for_target_eps(hist, set(args.forget), cfg,
                                        args.fedadp_target_eps)
        print(f"[{args.run_id}] FedADP budget match: target "
              f"{args.fedadp_target_eps:.2f} -> eps_cfg={e:.4f}")
    if not skip("FedADP"):
        print(f"[{args.run_id}] FedADP")
        logger.log("method_start", method="FedADP",
                   variant="dp" if cfg.fedadp.apply_dp else "nodp")
        fa_model, fa_info = run_fedadp(base, hist, client_loaders, cfg, device,
                                       logger=logger, eval_hook=eval_hook)
        extra = {}
        if fa_info["dp"]:
            extra["dp_epsilon"] = round(rdp_epsilon(fa_info["z_bar"], cfg.fedadp.delta), 2)
        final_row("FedADP", fa_model, extra)
        del fa_model; free()

    # ---- optional: match AdaptFU's privacy budget to a target epsilon ----
    if args.adaptfu_target_eps:
        from .utils import param_mask as _pm
        c, eps_before = scale_z_for_target_eps(
            hist, set(args.forget), cfg, _pm(base).cpu(), args.adaptfu_target_eps)
        print(f"[{args.run_id}] AdaptFU budget match: eps {eps_before:.2f} -> "
              f"target {args.adaptfu_target_eps:.2f}  (z scaled x{c:.3f} -> "
              f"z_min={cfg.adaptfu.z_min:.3f}, z_max={cfg.adaptfu.z_max:.3f})")
        logger.log("adaptfu_budget_match", scale=round(c, 4),
                   eps_before=round(float(eps_before), 2),
                   target_eps=args.adaptfu_target_eps,
                   z_min=round(cfg.adaptfu.z_min, 4), z_max=round(cfg.adaptfu.z_max, 4))

    # ---- AdaptFU across the (aggregation x attack) grid ----
    for agg in args.adaptfu_agg:
        for atk in args.adaptfu_attack:
            tag = f"AdaptFU/{agg}" + ("" if atk == "none" else f"|{atk}")
            if skip(tag):
                continue
            print(f"[{args.run_id}] {tag}")
            cfg.adaptfu.aggregation = agg
            acfg = None
            if atk != "none":
                acfg = copy.deepcopy(cfg.attack)
                acfg.enabled = True
                acfg.kind = atk
            logger.log("method_start", method=tag, variant=f"{agg}/{atk}")
            model, info = run_adaptfu(base, hist, client_loaders, cfg, device,
                                      attack_cfg=acfg, logger=logger,
                                      eval_hook=eval_hook, tag=tag)
            extra = {"dp_epsilon": round(rdp_epsilon(info["z_bar"], cfg.adaptfu.delta), 2),
                     "noise_mode": getattr(cfg.adaptfu, "noise_mode", "coord"),
                     "dp_mode": getattr(cfg.adaptfu, "dp_mode", "adaptive"),
                     "rho_scope": getattr(cfg.adaptfu, "rho_scope", "pooled"),
                     "agg_weight": getattr(cfg.adaptfu, "agg_weight", "samples"),
                     "clip_order": getattr(cfg.adaptfu, "clip_order", "after"),
                     "survivors_first": info["survivors"][:5]}
            final_row(tag, model, extra)
            del model; free()

    # loud failure if the filter selected nothing -- otherwise a run silently
    # trains only Standard FL and looks "successful"
    if only is not None and not matched:
        msg = (f"--only-methods {sorted(only)} matched NO method. Available tags "
               f"this run: {['FL Retrain','FuGuard','FedEraser','FedRecover','FedADP'] + [f'AdaptFU/{a}' for a in args.adaptfu_agg]}. "
               f"Note the AdaptFU tag follows --adaptfu-agg.")
        print("\n[WARNING] " + msg)
        logger.log("only_methods_matched_nothing", message=msg)

    # ---- summary ----
    summary = dict(run_id=args.run_id, config=vars(args), rows=rows)
    with open(os.path.join(args.out_dir, f"{args.run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.close()

    # all methods finished: the heavy resume checkpoint (base + ~18GB history +
    # attacker) is now dead weight. Remove it, keeping only the small durable
    # model artifacts (standardfl.pt, retrain.pt) plus progress/summary/logs.
    shared = bool(args.ckpt_run_id) and args.ckpt_run_id != args.run_id
    if ckpt_pt and not args.keep_ckpt and not shared and os.path.exists(ckpt_pt):
        os.remove(ckpt_pt)
        print(f"[{args.run_id}] removed resume checkpoint {os.path.basename(ckpt_pt)} "
              f"(kept standardfl.pt + retrain.pt)")

    # console table
    print("\n" + "=" * 90)
    print(f"SUMMARY {args.run_id}")
    print("=" * 90)
    cols = ["method", "test_acc", "forget_acc", "mia_auc"] + \
           (["asr"] if args.backdoor else []) + ["dp_epsilon", "cost_saving"]
    print(" | ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '-')):>14}" for c in cols))
    print(f"\nlog  -> {logger.path}")
    print(f"json -> {os.path.join(args.out_dir, args.run_id + '.json')}")


if __name__ == "__main__":
    main()
