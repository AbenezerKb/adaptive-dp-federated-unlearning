#!/usr/bin/env bash
# Reproduces every number in the paper. Two 24 GB GPUs assumed; set GPU below.
# Total wall time ~14 h. Each block is independent and resumable.
set -euo pipefail
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="python -m afu_vs_fg.run_experiment"

# Shared AdaptFU configuration used throughout the paper.
A="--adaptfu-z-min 0.62 --adaptfu-z-max 1.54 --adaptfu-warm-start 0.5 \
   --adaptfu-calib-epochs 1 --adaptfu-rho-scope round --adaptfu-noise-mode norm \
   --adaptfu-dp-mode per_client"
F="--fedadp-init first_stored --fedadp-dual-select 1 --fedadp-select-mode staged"
IO="--eval-every 2 --num-workers 2 --eval-n 5000 --log-dir ./logs --out-dir ./results \
    --checkpoint-dir ./ckpt --keep-ckpt"

# ---------------------------------------------------------------- Table 1
# One Standard-FL base per dataset, then all seven methods against it.
# --ckpt-run-id lets every later run share that base instead of retraining.
for CFG in \
  "cifar10   --dataset cifar10  --alpha 0.5 --forget 0 --lr 0.01 --lr-min 0.001" \
  "cifar100  --dataset cifar100 --alpha 0.5 --forget 0 --lr 0.01 --lr-min 0.001" \
  "femnist   --dataset femnist  --alpha 0.5 --forget 0 --lr 0.01 --lr-min 0.001 --subsample 60000" \
  "agnews    --dataset agnews   --alpha 0.5 --forget 0 --lr 0.01 --lr-min 0.001"
do
  set -- $CFG; NAME=$1; shift
  $RUN --run-id ${NAME}_base "$@" $IO --standardfl-only
  $RUN --run-id ${NAME}_all  "$@" $IO $A $F \
       --adaptfu-agg fedavg --adaptfu-attack none --ckpt-run-id ${NAME}_base --resume
done

# ---------------------------------------------------------------- Table 2
# CIFAR-100 across four aggregation rules x three attack conditions.
for AGG in fedavg krum trimmed_mean median; do
  for ATK in none noise minmax; do
    $RUN --run-id c100_rob_${AGG}_${ATK} --dataset cifar100 --alpha 0.5 --forget 0 \
         --lr 0.01 --lr-min 0.001 $IO $A --only-methods "AdaptFU/${AGG}" \
         --adaptfu-agg $AGG --adaptfu-attack $ATK \
         --ckpt-run-id cifar100_base --resume
  done
done
# Same two attacks on the other modalities (FedAvg only).
for D in cifar10 femnist agnews; do
  EXTRA=""; [ "$D" = femnist ] && EXTRA="--subsample 60000"
  for ATK in noise minmax; do
    $RUN --run-id ${D}_rob_${ATK} --dataset $D --alpha 0.5 --forget 0 \
         --lr 0.01 --lr-min 0.001 $EXTRA $IO $A --only-methods "AdaptFU/fedavg" \
         --adaptfu-agg fedavg --adaptfu-attack $ATK --ckpt-run-id ${D}_base --resume
  done
done

# ---------------------------------------------------------------- Table 3
# Paired per-client vs per-round DP at matched epsilon.
# --noise-seed reseeds ONLY the unlearning phase, so the base and partition are
# identical across seeds and the comparison is paired.
for S in 0 1 2 3; do
  for M in per_client per_round; do
    $RUN --run-id c100_dp_s${S}_${M} --dataset cifar100 --alpha 1.0 --forget 0 1 2 \
         --lr 0.01 --lr-min 0.001 $IO $A --adaptfu-dp-mode $M --noise-seed $S \
         --only-methods "AdaptFU" --adaptfu-agg fedavg --adaptfu-attack none \
         --ckpt-run-id cifar100_base --resume
  done
done
for M in per_client per_round; do
  $RUN --run-id fem_dp_${M} --dataset femnist --alpha 0.5 --forget 0 --subsample 60000 \
       --lr 0.01 --lr-min 0.001 $IO $A --adaptfu-dp-mode $M \
       --only-methods "AdaptFU" --adaptfu-agg fedavg --adaptfu-attack none \
       --ckpt-run-id femnist_base --resume
done

echo "Done. Summaries in ./results/*.json, per-round events in ./logs/*.jsonl"
