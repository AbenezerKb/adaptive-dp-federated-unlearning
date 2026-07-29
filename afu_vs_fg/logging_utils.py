"""Structured logging for the comparison harness.

Every run writes a newline-delimited JSON (.jsonl) file where each line is one
event with a shared shape: {"event": ..., "run_id": ..., "t": <wall_clock>, ...}.
One file per run (keyed by run_id) so parallel processes never contend.

Event types emitted across the harness:
  run_start     : run_id, config (dataset, alpha, forget, backdoor, seed, ...)
  fl_round      : phase (standardfl|retrain), round, mean_update_norm, step_norm
  fl_eval       : phase, round, test_acc                         (optional cadence)
  method_start  : method, variant (e.g. aggregation/attack)
  method_round  : method, round, + per-method internals (survivors, z_bar, ...)
  method_epoch  : method, epoch, losses (FuGuard)
  eval          : method, round, test_acc, forget_acc, mia_auc, [asr]
  result_row    : method, final metrics (the panel row)
  run_end       : run_id, wall_seconds

`plot_results.py` consumes exactly these events; keep the schema stable.
"""
import json
import os
import time


class RunLogger:
    def __init__(self, log_dir, run_id, append=False):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_id}.jsonl")
        self.run_id = run_id
        self._t0 = time.time()
        if not append:
            # fresh run: truncate any stale file for this run_id
            open(self.path, "w").close()
        # append=True (resume): keep prior events so the earlier phases' curves
        # (Standard FL / Retrain / already-finished methods) survive for plotting

    def log(self, event, **fields):
        rec = {"event": event, "run_id": self.run_id,
               "t": round(time.time() - self._t0, 3)}
        rec.update(fields)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def close(self):
        self.log("run_end", wall_seconds=round(time.time() - self._t0, 2))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def load_events(path):
    """Read a .jsonl log into a list of dicts."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_dir(log_dir):
    """Load all runs in a directory -> {run_id: [events]}."""
    runs = {}
    for fn in sorted(os.listdir(log_dir)):
        if fn.endswith(".jsonl"):
            evs = load_events(os.path.join(log_dir, fn))
            if evs:
                runs[evs[0].get("run_id", fn[:-6])] = evs
    return runs
