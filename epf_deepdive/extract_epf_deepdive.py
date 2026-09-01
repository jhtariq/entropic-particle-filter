"""Offline deep-dive over stored EPF sweep JSONLs (epf3 full-trace runs).

Replays alternative selection rules, computes self-certainty signal diagnostics,
step-quality stats, and resampling-collapse stats — all WITHOUT new inference.
Read-only over the run folders; writes only into --out.

Usage (on a compute node):
  python extract_epf_deepdive.py \
    --runs mmau-pro-d1k:/u/awaheed/mmau_pro_d1k_epf3_out/samples \
           mmar:/u/awaheed/mmar_epf3_out/samples \
    --out /work/hdd/bcey/awaheed/its-for-audio-reasoning/epf_deepdive/out \
    --workers 8
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

LETTERS = "ABCDEFGHIJK"


# ---------------------------------------------------------------- per-particle measures

def traj_measures(particle):
    """Confidence measures for one particle from its per-step signal summaries."""
    sigs = particle.get("signals") or []
    lps, ents, toks = [], [], []
    for s in sigs:
        nt = s.get("num_tokens") or 0
        if nt <= 0:
            continue
        toks.append(nt)
        lp = s.get("mean_logprob")
        lps.append(lp if lp is not None else 0.0)
        ents.append(s.get("entropy"))
    m = {}
    if toks:
        tot = sum(toks)
        m["traj_mean_lp"] = sum(lp * t for lp, t in zip(lps, toks)) / tot
        m["traj_sum_lp"] = sum(lp * t for lp, t in zip(lps, toks))
        m["traj_min_step_lp"] = min(lps)
        m["traj_last_lp"] = lps[-1]
        tail = list(zip(lps, toks))[-2:]
        m["traj_tail_lp"] = sum(lp * t for lp, t in tail) / sum(t for _, t in tail)
        m["n_tokens"] = tot
        m["n_steps_sig"] = len(toks)
    if any(e is not None for e in ents):
        es = [e for e in ents if e is not None]
        m["traj_mean_ent"] = sum(es) / len(es)
        m["traj_max_step_ent"] = max(es)
        m["traj_last_ent"] = es[-1]
    return m


_norm_digit = re.compile(r"\d+")
_norm_ws = re.compile(r"\s+")


def _norm_step(s):
    return _norm_ws.sub(" ", _norm_digit.sub("", s.lower())).strip()


def has_repeat(steps):
    seen = set()
    for s in steps:
        n = _norm_step(s)
        if len(n) < 15:
            continue
        if n in seen:
            return True
        seen.add(n)
    return False


# ---------------------------------------------------------------- vote helpers

def plurality(letters, weights=None, tiebreak=None):
    """Weighted plurality over non-empty letters. tiebreak: dict letter->score."""
    if weights is None:
        weights = [1.0] * len(letters)
    tally = defaultdict(float)
    for L, w in zip(letters, weights):
        if L and L != "?":
            tally[L] += w
    if not tally:
        return ""
    best = max(tally.values())
    tied = sorted([L for L, v in tally.items() if v >= best - 1e-12])
    if len(tied) == 1 or not tiebreak:
        return tied[0]
    return max(tied, key=lambda L: (tiebreak.get(L, -1e18), -LETTERS.index(L)))


def softmax(xs):
    if not xs:
        return []
    mx = max(xs)
    es = [math.exp(x - mx) for x in xs]
    s = sum(es)
    return [e / s for e in es]


# ---------------------------------------------------------------- per-row analysis

def analyze_row(r):
    """Return (key, selector_hits, diag_sums, auc_stats, dump_candidate)."""
    budget = int(r["budget"])
    signal = r["signal"]
    gold = r.get("gold_letter") or ""
    preds = (r.get("preds") or "").split(",") if r.get("preds") else []
    parts = r.get("particles") or []
    resp = r.get("responses") or []
    n = len(preds)
    if n == 0 or len(parts) != n:
        return None
    gradeable = bool(gold)

    meas = [traj_measures(p) for p in parts]
    fin_w = [(p["log_weights"][-1] if p.get("log_weights") else 0.0) for p in parts]

    # tie-break score per letter: summed exp(traj_mean_lp) of its particles
    conf = [math.exp(m["traj_mean_lp"]) if "traj_mean_lp" in m else 0.0 for m in meas]
    tb = defaultdict(float)
    for L, c in zip(preds, conf):
        if L and L != "?":
            tb[L] += c

    sel = {}
    # original argmax-final-weight selection (replay)
    i_sel = max(range(n), key=lambda i: fin_w[i])
    sel["orig_argmax_w"] = preds[i_sel]
    sel["majority"] = plurality(preds, tiebreak=tb)
    sw = softmax(fin_w)
    sel["weighted_vote_finw"] = plurality(preds, weights=sw, tiebreak=tb)
    sel["conf_vote_meanlp"] = plurality(preds, weights=conf, tiebreak=tb)

    def argmax_pred(key, flip=False):
        idxs = [i for i in range(n) if key in meas[i]]
        if not idxs:
            return ""
        f = (lambda i: -meas[i][key]) if flip else (lambda i: meas[i][key])
        return preds[max(idxs, key=f)]

    sel["argmax_mean_lp"] = argmax_pred("traj_mean_lp")
    sel["argmax_sum_lp"] = argmax_pred("traj_sum_lp")
    sel["argmax_min_step_lp"] = argmax_pred("traj_min_step_lp")
    sel["argmax_last_lp"] = argmax_pred("traj_last_lp")
    sel["argmax_tail_lp"] = argmax_pred("traj_tail_lp")
    sel["argmin_mean_ent"] = argmax_pred("traj_mean_ent", flip=True)
    sel["argmin_max_step_ent"] = argmax_pred("traj_max_step_ent", flip=True)

    # DeepConf-style filtered majority: keep top q by traj_mean_lp
    order = sorted(range(n), key=lambda i: meas[i].get("traj_mean_lp", -1e18), reverse=True)
    for q, tag in ((0.25, "top25"), (0.5, "top50"), (0.75, "top75")):
        k = max(1, int(round(n * q)))
        keep = order[:k]
        sel[f"filt_majority_{tag}"] = plurality([preds[i] for i in keep], tiebreak=tb)

    # majority among particles that actually answered (final text has Answer:)
    answered = [i for i in range(n) if "Answer:" in (resp[i] if i < len(resp) else "")]
    sel["answered_majority"] = (
        plurality([preds[i] for i in answered], tiebreak=tb) if answered else sel["majority"]
    )

    # dedup by exact final text (collapse resampling duplicates), then majority
    seen, uniq = set(), []
    for i in range(n):
        t = resp[i] if i < len(resp) else str(i)
        if t not in seen:
            seen.add(t)
            uniq.append(i)
    sel["dedup_majority"] = plurality([preds[i] for i in uniq], tiebreak=tb)
    sel["dedup_conf_vote"] = plurality(
        [preds[i] for i in uniq], weights=[conf[i] for i in uniq], tiebreak=tb
    )

    hits = {k: (1 if (v == gold) else 0) for k, v in sel.items()} if gradeable else None
    parsed = [p for p in preds if p and p != "?"]
    oracle = (1 if any(p == gold for p in parsed) else 0) if gradeable else None

    # ---------- diagnostics ----------
    d = Counter()
    d["items"] = 1
    d["n_particles"] = n
    d["n_unique_text"] = len(uniq)
    d["n_answered"] = len(answered)
    d["n_parsed"] = len(parsed)
    d["n_distinct_answers"] = len(set(parsed))
    d["n_repeat_traj"] = sum(1 for p in parts if has_repeat(p.get("steps") or []))
    d["n_steps_total"] = sum(len(p.get("steps") or []) for p in parts)
    d["n_maxstep_truncated"] = sum(
        1
        for i, p in enumerate(parts)
        if len(p.get("steps") or []) >= 6 and "Answer:" not in (resp[i] if i < len(resp) else "")
    )
    # steps whose token count hit the 300 cap (cut mid-step)
    cap = tot_steps = 0
    for p in parts:
        for s in p.get("signals") or []:
            nt = s.get("num_tokens") or 0
            if nt > 0:
                tot_steps += 1
                if nt >= 300:
                    cap += 1
    d["n_steps_capped"] = cap
    d["n_steps_sig"] = tot_steps
    d["n_tokens_total"] = sum(m.get("n_tokens", 0) for m in meas)
    if gradeable:
        d["gradeable"] = 1
        d["oracle"] = oracle
        d["sel_match_stored"] = 1 if sel["orig_argmax_w"] == (r.get("selected_letter") or "") else 0

    # trace-derived: lineage roots + annealing
    tr = r.get("trace")
    if tr and tr.get("iterations"):
        anc = list(range(n))
        n_temp_gt1 = 0
        for it in tr["iterations"]:
            par = it["resample"]["parents"]
            anc = [anc[p] for p in par]
            if it["resample"].get("temperature", 1.0) > 1.0 + 1e-9:
                n_temp_gt1 += 1
        d["n_roots_final"] = len(set(anc))
        d["n_iters"] = len(tr["iterations"])
        d["n_iters_tempered"] = n_temp_gt1
        d["has_trace"] = 1

    # ---------- pairwise AUC stats (within item: correct vs incorrect particles) ----------
    auc = {}
    if gradeable:
        cor = [i for i in range(n) if preds[i] == gold]
        inc = [i for i in range(n) if preds[i] and preds[i] != "?" and preds[i] != gold]
        if cor and inc:
            for key, flip in (
                ("traj_mean_lp", False), ("traj_sum_lp", False), ("traj_min_step_lp", False),
                ("traj_last_lp", False), ("traj_tail_lp", False),
                ("traj_mean_ent", True), ("traj_max_step_ent", True),
            ):
                wins = ties = tot = 0
                for i in cor:
                    if key not in meas[i]:
                        continue
                    a = meas[i][key] * (-1 if flip else 1)
                    for j in inc:
                        if key not in meas[j]:
                            continue
                        b = meas[j][key] * (-1 if flip else 1)
                        tot += 1
                        if a > b:
                            wins += 1
                        elif a == b:
                            ties += 1
                if tot:
                    auc[key] = ((wins + 0.5 * ties) / tot, 1)
            # the ACTUAL final weight EPF used
            wins = ties = tot = 0
            for i in cor:
                for j in inc:
                    tot += 1
                    if fin_w[i] > fin_w[j]:
                        wins += 1
                    elif fin_w[i] == fin_w[j]:
                        ties += 1
            if tot:
                auc["final_weight"] = ((wins + 0.5 * ties) / tot, 1)

    # ---------- dump candidate ----------
    dump = None
    if budget == 8 and gradeable:
        maj = sel["majority"]
        selL = sel["orig_argmax_w"]
        if selL != gold and maj == gold:
            cat = "sel_wrong_maj_right"
        elif oracle == 0:
            cat = "oracle_miss"
        elif d["n_maxstep_truncated"] > 0:
            cat = "truncated"
        elif selL != gold:
            cat = "sel_wrong"
        else:
            cat = None
        if cat:
            dump = {
                "cat": cat, "unique_id": r["unique_id"], "gold": gold,
                "selected": selL, "majority": maj, "preds": r.get("preds"),
                "final_weights": fin_w,
                "particles": [
                    {"pred": preds[i], "steps": parts[i].get("steps"),
                     "signals": parts[i].get("signals"),
                     "log_weights": parts[i].get("log_weights")}
                    for i in range(min(n, 8))
                ],
            }

    key = (signal, budget)
    return key, hits, d, auc, dump


# ---------------------------------------------------------------- per-file worker

def process_file(args):
    path, bench = args
    model = os.path.basename(path).split("__")[0]
    sel_hits = defaultdict(Counter)   # (sig,bud) -> Counter(selector -> hits, _n)
    diags = defaultdict(Counter)      # (sig,bud) -> Counter
    aucs = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # (sig,bud) -> measure -> [sum, n]
    dumps = defaultdict(list)         # cat -> rows
    n_err = n_rows = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_err += 1
                continue
            if r.get("error"):
                n_err += 1
                continue
            if int(r.get("method", -1)) != 4 or r.get("signal") not in ("mean_logprob", "entropy"):
                continue
            out = analyze_row(r)
            if out is None:
                continue
            n_rows += 1
            key, hits, d, auc, dump = out
            if hits:
                for k, v in hits.items():
                    sel_hits[key][k] += v
                sel_hits[key]["_n"] += 1
            diags[key].update(d)
            for m, (v, c) in auc.items():
                aucs[key][m][0] += v
                aucs[key][m][1] += c
            if dump and len(dumps[dump["cat"]]) < 2:
                dumps[dump["cat"]].append(dump)
    return {
        "bench": bench, "model": model, "path": path, "n_rows": n_rows, "n_err": n_err,
        "sel_hits": {k: dict(v) for k, v in sel_hits.items()},
        "diags": {k: dict(v) for k, v in diags.items()},
        "aucs": {k: {m: list(v) for m, v in mv.items()} for k, mv in aucs.items()},
        "dumps": dict(dumps),
    }


# ---------------------------------------------------------------- merge + write

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="bench:dir pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=None, help="comma list to restrict model tags")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tasks = []
    for spec in a.runs:
        bench, d = spec.split(":", 1)
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            tag = os.path.basename(p).split("__")[0]
            if a.models and tag not in a.models.split(","):
                continue
            tasks.append((p, bench))
    print(f"{len(tasks)} files", flush=True)

    with Pool(a.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(process_file, tasks)):
            results.append(res)
            print(f"[{i+1}/{len(tasks)}] {res['bench']} {res['model']} "
                  f"rows={res['n_rows']} err={res['n_err']} ({os.path.basename(res['path'])})",
                  flush=True)

    # merge per (bench, model)
    S = defaultdict(Counter)
    D = defaultdict(Counter)
    A = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    DU = defaultdict(list)
    for res in results:
        bm = (res["bench"], res["model"])
        for key, c in res["sel_hits"].items():
            S[(bm, tuple(key))].update(c)
        for key, c in res["diags"].items():
            D[(bm, tuple(key))].update(c)
        for key, mv in res["aucs"].items():
            for m, (v, c) in mv.items():
                A[(bm, tuple(key))][m][0] += v
                A[(bm, tuple(key))][m][1] += c
        for cat, rows in res["dumps"].items():
            DU[(bm, cat)].extend(rows)

    with open(os.path.join(a.out, "selectors.csv"), "w") as f:
        f.write("bench,model,signal,budget,selector,acc,n\n")
        for ((bench, model), (sig, bud)), c in sorted(S.items()):
            nn = c.get("_n", 0)
            for k, v in sorted(c.items()):
                if k == "_n" or not nn:
                    continue
                f.write(f"{bench},{model},{sig},{bud},{k},{v/nn:.4f},{nn}\n")

    with open(os.path.join(a.out, "diagnostics.csv"), "w") as f:
        f.write("bench,model,signal,budget,metric,value\n")
        for ((bench, model), (sig, bud)), c in sorted(D.items()):
            it = c.get("items", 0) or 1
            npart = c.get("n_particles", 0) or 1
            nst = c.get("n_steps_sig", 0) or 1
            g = c.get("gradeable", 0) or 1
            rows = {
                "items": it,
                "oracle_acc": c.get("oracle", 0) / g,
                "sel_replay_match": c.get("sel_match_stored", 0) / g,
                "parsed_ratio": c.get("n_parsed", 0) / npart,
                "answered_ratio": c.get("n_answered", 0) / npart,
                "truncated_ratio": c.get("n_maxstep_truncated", 0) / npart,
                "repeat_traj_ratio": c.get("n_repeat_traj", 0) / npart,
                "unique_text_ratio": c.get("n_unique_text", 0) / npart,
                "distinct_answers_per_item": c.get("n_distinct_answers", 0) / it,
                "steps_per_particle": c.get("n_steps_total", 0) / npart,
                "tokens_per_step": c.get("n_tokens_total", 0) / nst,
                "capped_step_ratio": c.get("n_steps_capped", 0) / nst,
            }
            if c.get("has_trace"):
                ht = c["has_trace"]
                rows["roots_final_mean"] = c.get("n_roots_final", 0) / ht
                rows["iters_mean"] = c.get("n_iters", 0) / ht
                rows["iters_tempered_mean"] = c.get("n_iters_tempered", 0) / ht
            for k, v in rows.items():
                f.write(f"{bench},{model},{sig},{bud},{k},{v:.4f}\n")

    with open(os.path.join(a.out, "pairwise_auc.csv"), "w") as f:
        f.write("bench,model,signal,budget,measure,auc,n_items\n")
        for ((bench, model), (sig, bud)), mv in sorted(A.items()):
            for m, (v, c) in sorted(mv.items()):
                if c:
                    f.write(f"{bench},{model},{sig},{bud},{m},{v/c:.4f},{c}\n")

    ddir = os.path.join(a.out, "dumps")
    os.makedirs(ddir, exist_ok=True)
    for ((bench, model), cat), rows in DU.items():
        with open(os.path.join(ddir, f"{bench}__{model}__{cat}.json"), "w") as f:
            json.dump(rows[:6], f, indent=1)

    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
