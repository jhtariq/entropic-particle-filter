"""Step-resolved EPF dynamics from stored epf3 traces (no inference).

For every (bench, model, signal, budget, step/iteration) cell, over all items:
  - weight concentration the resampler saw: ESS ratio pre-temper (from particle
    log-weights) and post-temper (stored probabilities), annealing temperature,
    fraction of rounds where annealing fired / would fire at 0.5
  - weight spread: std and max-min gap of log-weights across the swarm, max p
  - diversity: unique-ancestor ratio after each resample (lineage survival),
    fraction of particles already stopped
  - raw signals: mean_logprob / entropy per step (deduped by trajectory prefix so
    resampling copies don't double-count)

Output: step_dynamics.json (per-cell aggregates incl. percentiles + ESS histogram).

  python step_dynamics.py --runs mmar:/u/awaheed/mmar_epf3_out/samples \
      mmau-pro-d1k:/u/awaheed/mmau_pro_d1k_epf3_out/samples \
      --models qwen-omni-3b,qwen-omni,qwen2-audio --workers 6 \
      --out step_dynamics.json
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np


def softmax(xs):
    x = np.asarray(xs, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def ess_ratio(p):
    p = np.asarray(p, dtype=float)
    s = float((p ** 2).sum())
    return (1.0 / s) / len(p) if s > 0 else 1.0


def process_file(args):
    path, bench = args
    model = os.path.basename(path).split("__")[0]
    # cell key -> metric -> list of per-item values
    vals = defaultdict(lambda: defaultdict(list))
    softmax_mismatch = rows = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or int(r.get("method", -1)) != 4:
                continue
            sig = r.get("signal")
            if sig not in ("mean_logprob", "entropy"):
                continue
            budget = int(r["budget"])
            rows += 1

            # ---- per-iteration concentration + diversity from the trace ----
            tr = r.get("trace")
            if tr and tr.get("iterations"):
                n = budget
                anc = list(range(n))
                for k, it in enumerate(tr["iterations"]):
                    key = (sig, budget, k)
                    snaps = it.get("particles") or []
                    res = it.get("resample") or {}
                    lw = [s.get("cum_log_weight", 0.0) for s in snaps]
                    if len(lw) >= 2:
                        p_pre = softmax(lw)
                        e_pre = ess_ratio(p_pre)
                        temp = float(res.get("temperature", 1.0))
                        probs = res.get("probabilities")
                        e_post = ess_ratio(probs) if probs else e_pre
                        if probs and temp == 1.0:
                            if abs(float(np.max(np.abs(p_pre - np.asarray(probs))))) > 1e-6:
                                softmax_mismatch += 1
                        vals[key]["ess_pre"].append(e_pre)
                        vals[key]["ess_post"].append(e_post)
                        vals[key]["temp"].append(temp)
                        vals[key]["anneal_fired"].append(1.0 if temp > 1.0 + 1e-9 else 0.0)
                        vals[key]["ess_lt06"].append(1.0 if e_pre < 0.6 else 0.0)
                        vals[key]["ess_lt05"].append(1.0 if e_pre < 0.5 else 0.0)
                        vals[key]["w_std"].append(float(np.std(lw)))
                        vals[key]["w_gap"].append(float(np.max(lw) - np.min(lw)))
                        vals[key]["max_p"].append(float(np.max(p_pre)))
                        vals[key]["exp_max_copies"].append(float(np.max(p_pre)) * n)
                    if snaps:
                        vals[key]["frac_stopped"].append(
                            sum(1 for s in snaps if s.get("stopped")) / len(snaps))
                    par = res.get("parents")
                    if par:
                        anc = [anc[p] for p in par]
                        vals[key]["uniq_anc_ratio"].append(len(set(anc)) / n)

            # ---- per-step raw signals from final lineages, deduped by prefix ----
            seen_prefix = set()
            for pt in (r.get("particles") or []):
                steps = pt.get("steps") or []
                sigs = pt.get("signals") or []
                for k in range(min(len(steps), len(sigs))):
                    h = hash(tuple(steps[: k + 1]))
                    if (k, h) in seen_prefix:
                        continue
                    seen_prefix.add((k, h))
                    s = sigs[k]
                    key = (sig, budget, k)
                    mlp = s.get("mean_logprob")
                    if mlp is not None:
                        vals[key]["mean_logprob"].append(float(mlp))
                    ent = s.get("entropy")
                    if ent is not None:
                        vals[key]["entropy"].append(float(ent))
                    nt = s.get("num_tokens")
                    if nt:
                        vals[key]["step_tokens"].append(float(nt))

    # compact: convert lists to numpy-summarizable form and return raw (small enough per file)
    out = {}
    for key, mv in vals.items():
        out[key] = {m: v for m, v in mv.items()}
    return {"bench": bench, "model": model, "rows": rows,
            "softmax_mismatch": softmax_mismatch, "vals": out}


def summarize(v):
    a = np.asarray(v, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = []
    keep = set(a.models.split(","))
    for spec in a.runs:
        bench, d = spec.split(":", 1)
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            if os.path.basename(p).split("__")[0] in keep:
                tasks.append((p, bench))
    print(f"{len(tasks)} files", flush=True)

    merged = defaultdict(lambda: defaultdict(list))  # (bench,model,sig,bud,step) -> metric -> vals
    mism = 0
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, tasks)):
            mism += res["softmax_mismatch"]
            for (sig, bud, step), mv in res["vals"].items():
                for m, v in mv.items():
                    merged[(res["bench"], res["model"], sig, bud, step)][m].extend(v)
            print(f"[{i+1}/{len(tasks)}] {res['bench']} {res['model']} rows={res['rows']}",
                  flush=True)
    print("softmax sanity mismatches:", mism, flush=True)

    out = []
    for (bench, model, sig, bud, step), mv in sorted(merged.items()):
        cell = {"bench": bench, "model": model, "signal": sig, "budget": bud, "step": step}
        for m, v in mv.items():
            if not v:
                continue
            if m in ("anneal_fired", "ess_lt06", "ess_lt05"):
                cell[m] = round(float(np.mean(v)), 4)
                cell[m + "_n"] = len(v)
            else:
                cell[m] = summarize(v)
        if "ess_pre" in mv and len(mv["ess_pre"]) >= 20:
            hist, _ = np.histogram(mv["ess_pre"], bins=10, range=(0, 1))
            cell["ess_pre_hist10"] = [int(x) for x in hist]
        out.append(cell)

    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} cells -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
