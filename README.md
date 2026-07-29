# Adaptive-DP Federated Unlearning code artifact

Reference implementation and the exact commands that produce every number in the
paper. Seven methods share one Standard-FL checkpoint per dataset, so differences
are attributable to the unlearning procedure alone.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10–3.12, PyTorch ≥2.1, one CUDA GPU (24 GB used in our runs; less works
for MNIST/FEMNIST/AG News). CIFAR-10/100 and EMNIST download automatically via
torchvision; AG News downloads a CSV on first use. FEMNIST is described below.

## Reproduce

```bash
GPU=0 bash scripts/run_all.sh          # ~14 h on one RTX 4090
python analysis/make_tables.py         # -> tables.tex  (Tables 1-3)
python analysis/make_figures.py        # -> fig*.pdf / .png
```

`run_all.sh` is organised by table. Each block is independent and resumable, so
you can run only the part you want. Every run writes a summary to
`results/<run-id>.json` and per-round events to `logs/<run-id>.jsonl`.

`analysis/results_data.py` holds the numbers reported in the paper as a single
source of truth; both generators read from it. To regenerate the paper's assets
from your own runs, replace the values there.

## Layout

```
afu_vs_fg/
  run_experiment.py   single entry point (all flags; --help documents each)
  config.py           per-dataset presets and every mechanism switch
  fl_train.py         FedAvg with the history caching unlearning needs
  adaptfu.py          our method: reliability, per-client DP, verification
  baselines.py        FedEraser, FedRecover, FedADP
  fuguard.py          FuGuard (generative surrogate + optimal transport)
  attacks.py          norm-inflating and norm-matched (min-max) attacks
  aggregation.py      FedAvg, Krum, trimmed mean, coordinate median
  metrics.py          accuracy, forget-set accuracy, MIA AUC, RDP accountant
  data.py             datasets, Dirichlet and per-writer partitions
  models.py           LeNet-5, ResNet-18, TextCNN
analysis/             results_data.py + table/figure generators
scripts/run_all.sh    every command behind the paper
```

## Notes that affect reproduction

**Checkpoint reuse.** `--checkpoint-dir` saves the base model, cached history and
MIA attacker once; later runs pass `--ckpt-run-id <base> --resume` to share it
instead of retraining. This is what makes the ablations cheap, and it guarantees
every method sees an identical base. The CIFAR-100 checkpoint is ~20 GB (fp16),
so provide disk accordingly, or pass `--ckpt-fp16 0` for fp32.

**Mechanism switches.** Three defaults matter and are not arbitrary:

- `--adaptfu-rho-scope round` — the sensitivity percentiles are taken from the
  current round. Pooling them over all replay rounds lets an across-round decay
  in update norms dominate, collapsing the per-client spread of *z* to ≈0.04 and
  making the allocation effectively uniform.
- `--adaptfu-noise-mode norm` — the noise *magnitude* is calibrated to `z·b`. With
  per-coordinate standard deviation `z·b` (the literal reading of the noise-and-clip
  step) the noise has ℓ₂ norm `z·b·√d`, roughly 4000× the update at d≈10⁷; the
  projection then renormalises it and the `z` factor cancels exactly, so the
  allocation has no effect. **Under this mode the reported ε is a relative
  reference, not an (ε,δ) guarantee.**
- `--adaptfu-dp-mode per_client|per_round|static` — `per_round` uses the round-mean
  *z* for every client, giving an identical composed ε by construction. That pair
  isolates the client axis and is what Table 3 measures.

**Min-max attack.** Norm-matched by construction: the crafted update is rescaled
to the benign median norm and constrained by the min-max distance budget. A
version that only maximises deviation is *norm-inflating* and gets rejected by
the verification, which inverts the conclusion — check `survivors_kept` in the
logs (9/9 means it evaded, 7/9 means it was caught).

**FEMNIST.** With LEAF shards laid out as `<root>/train/*.json` and
`<root>/test/*.json` the loader uses the natural per-writer partition; the paths
searched are printed at startup. Without them it falls back to EMNIST/byclass
(same 62 classes, but Dirichlet rather than per-writer) and says so. The paper's
FEMNIST results use the fallback with `--subsample 60000`.

**Baselines.** FedEraser, FedRecover, FedADP and FuGuard are our
re-implementations under a shared protocol, not the authors' code. Two did not
converge in every setting (FuGuard on FEMNIST; FedADP on CIFAR-10 and CIFAR-100)
and appear as dashes in Table 1. Those are artefacts of our re-implementation and
should not be read as evidence about the methods.

**Seeds.** `--seed` controls the data partition and base training. `--noise-seed`
reseeds only the unlearning phase, leaving partition and base fixed, which is what
makes Table 3's comparison paired. Main and robustness results are single-seed;
only Table 3 is multi-seed.
