"""Score MMAR arms and build the self-consistency budget curve from saved samples.

Two accuracies are reported at every budget:

  majority  the plurality letter over the K drawn samples — what self-consistency
            actually achieves.
  oracle    pass@K: correct if ANY of the K drawn samples is right. The upper bound on
            every selection method at that budget (majority, reranking, PF/EPF), so
            `oracle - majority` is the headroom a better selector could still capture.

Reads the JSONL written by `sample_runner.py` (one row per generated sample) and emits a
RESULTS.md carrying every diagnostic needed to decide whether a number is trustworthy:

  * greedy arms (n=1)      : accuracy, parse rate, truncation, length, answer distribution
  * self-consistency arms  : MAJORITY accuracy and ORACLE accuracy vs. particle budget K,
                             over a random K-subset of the N saved samples, R reps per K
  * per-cell health        : errors, unparsed, truncation, tie rate, vote concentration,
                             prompt/response token stats, latency, throughput
  * breakdowns             : by MMAR modality (sound/music/speech/mix-*) and duration
  * bias check             : predicted-letter distribution vs gold-letter distribution

Subsampling contract
--------------------
For each budget K < N we draw R independent subsets of size K WITHOUT replacement from
the N saved samples of that item, majority-vote each, and report mean +/- sd of accuracy
across the R replicates. This is what makes small budgets meaningful: a single K=8 draw
is a coin flip, the spread over R=50 draws is the actual quantity of interest.

At K = N there is exactly one possible subset, so it is one exact number with no error
bar (R collapses to 1) — reported as sd = 0 and flagged `(exact)`.

Voting rules, stated explicitly because they move the numbers:
  * A sample whose answer could not be parsed contributes NO vote. An item where every
    sampled vote is unparseable counts as WRONG (it is not dropped from the denominator).
  * Ties are broken uniformly at random from among the tied letters, using the seeded
    RNG — not by option order, which would bias toward 'A'.
  * Ungradeable items (MMAR ships 4 whose gold text matches no single choice) are
    excluded from all denominators, matching verify_data.py's 996-gradeable pin.

Usage:
    python -m benchmarking.mmar.sc_report --jsonl 'results/*.jsonl' --out RESULTS.md
"""

import collections
import glob
import json
import math
import os
import random
import statistics

import click

BUDGETS = (1, 8, 16, 32, 64, 128)
DEFAULT_REPLICATES = 50


# Requests carry the SERVED name, which for Phi-4-MM is its speech-LoRA adapter name
# ("speech") rather than anything recognisable. Map it back for display only — the raw
# rows keep whatever the server was actually asked for.
DISPLAY_NAME = {"speech": "phi4mm"}


def display(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def load_rows(patterns: list[str]) -> tuple[list[dict], list[str]]:
    rows, seen = [], []
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            seen.append(path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:  # truncated tail from a hard kill
                        continue
    return rows, seen


def group(rows: list[dict]) -> dict:
    """(model, arm) -> {'items': uid -> {...}, 'errors': [...], 'cfg': {...}}.

    De-dupe is by (unique_id, sample_idx) keeping the LAST occurrence, so a resumed run
    that re-generated an item does not double-count its samples in the vote.
    """
    samples: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    errors: dict = collections.defaultdict(list)
    first: dict = collections.defaultdict(dict)
    cfg: dict = {}
    for r in rows:
        key = (r.get("model", "?"), r.get("arm", "?"))
        cfg.setdefault(key, {
            "method": r.get("method"), "temperature": r.get("temperature"),
            "n_requested": r.get("n_requested"),
        })
        uid = r.get("unique_id")
        if uid is None:
            continue
        if r.get("error") or r.get("sample_idx", -1) < 0:
            errors[key].append(r)
            continue
        samples[key][uid][r["sample_idx"]] = r
        first[key].setdefault(uid, r)
    out = {}
    for key, items in samples.items():
        out[key] = {
            "cfg": cfg.get(key, {}),
            "errors": errors.get(key, []),
            "items": {
                uid: {
                    "gold": first[key][uid].get("gold_letter", ""),
                    "category": first[key][uid].get("category", "?"),
                    "length_type": first[key][uid].get("length_type", "?"),
                    "n_choices": first[key][uid].get("n_choices"),
                    "samples": [by_idx[i] for i in sorted(by_idx)],
                }
                for uid, by_idx in items.items()
            },
        }
    return out


def majority(votes: list[str], rng: random.Random) -> tuple[str, bool]:
    """(plurality letter over non-empty votes, was_tied). Ties broken uniformly at random."""
    valid = [v for v in votes if v]
    if not valid:
        return "", False
    counts = collections.Counter(valid)
    top = max(counts.values())
    best = sorted(k for k, v in counts.items() if v == top)
    if len(best) == 1:
        return best[0], False
    return rng.choice(best), True


def budget_curve(items: dict, budgets, replicates: int, rng: random.Random,
                 uid_filter=None) -> list[dict]:
    uids = sorted(u for u, d in items.items()
                  if d["gold"] and (uid_filter is None or u in uid_filter))
    if not uids:
        return []
    n_avail = min(len(items[u]["samples"]) for u in uids)
    out = []
    for k in budgets:
        if k > n_avail:
            continue
        reps = 1 if k >= n_avail else replicates
        accs, oras, ties = [], [], []
        for _ in range(reps):
            corr = orac = tie = 0
            for u in uids:
                d = items[u]
                letters = [s.get("pred_letter", "") for s in d["samples"]]
                pick = letters if k >= len(letters) else rng.sample(letters, k)
                v, t = majority(pick, rng)
                corr += v == d["gold"]
                # oracle / pass@K: a perfect selector picks the right sample whenever ANY
                # of the K drawn samples is right. Upper bound on every selection method
                # (majority, reranking, PF/EPF) at this budget.
                orac += any(x == d["gold"] for x in pick)
                tie += t
            accs.append(corr / len(uids))
            oras.append(orac / len(uids))
            ties.append(tie / len(uids))
        out.append({
            "budget": k, "replicates": reps, "n": len(uids),
            "mean": statistics.fmean(accs),
            "sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "lo": min(accs), "hi": max(accs),
            "oracle": statistics.fmean(oras),
            "oracle_sd": statistics.stdev(oras) if len(oras) > 1 else 0.0,
            "tie_rate": statistics.fmean(ties),
            "exact": k >= n_avail,
        })
    return out


def cell_health(cell: dict) -> dict:
    """Diagnostics that catch a silently-broken arm before its accuracy is believed."""
    items = cell["items"]
    n_s = n_parsed = n_trunc = 0
    words, ptoks, lats, finish = [], [], [], collections.Counter()
    pred_dist, gold_dist, nchoices = collections.Counter(), collections.Counter(), collections.Counter()
    per_item_counts, no_vote_items, concentration = [], 0, []
    for d in items.values():
        gold_dist[d["gold"] or "-"] += 1
        nchoices[d["n_choices"]] += 1
        per_item_counts.append(len(d["samples"]))
        letters = []
        for s in d["samples"]:
            n_s += 1
            pl = s.get("pred_letter", "")
            letters.append(pl)
            n_parsed += bool(pl)
            pred_dist[pl or "-"] += 1
            fr = s.get("finish_reason") or "?"
            finish[fr] += 1
            n_trunc += fr == "length"
            words.append(len((s.get("response") or "").split()))
            if s.get("prompt_tokens") is not None:
                ptoks.append(s["prompt_tokens"])
            if s.get("latency_s") is not None:
                lats.append(s["latency_s"])
        valid = [x for x in letters if x]
        if not valid:
            no_vote_items += 1
        else:
            concentration.append(collections.Counter(valid).most_common(1)[0][1] / len(valid))
    return {
        "items": len(items),
        "gradeable": sum(1 for d in items.values() if d["gold"]),
        "ungradeable": sum(1 for d in items.values() if not d["gold"]),
        "samples": n_s,
        "samples_per_item_min": min(per_item_counts) if per_item_counts else 0,
        "samples_per_item_max": max(per_item_counts) if per_item_counts else 0,
        "error_rows": len(cell["errors"]),
        "error_items": len({e.get("unique_id") for e in cell["errors"]}),
        "parse_rate": n_parsed / max(n_s, 1),
        "unparsed": n_s - n_parsed,
        "no_vote_items": no_vote_items,
        "trunc_rate": n_trunc / max(n_s, 1),
        "finish": finish,
        "words_mean": statistics.fmean(words) if words else 0.0,
        "words_med": statistics.median(words) if words else 0.0,
        "words_p90": quantile([float(w) for w in words], 0.90),
        "words_max": max(words) if words else 0,
        "ptok_mean": statistics.fmean(ptoks) if ptoks else 0.0,
        "ptok_max": max(ptoks) if ptoks else 0,
        "lat_med": statistics.median(lats) if lats else 0.0,
        "pred_dist": pred_dist, "gold_dist": gold_dist, "nchoices": nchoices,
        "concentration": statistics.fmean(concentration) if concentration else 0.0,
    }


def breakdown(items: dict, rng: random.Random, budget: int, key: str) -> list[tuple]:
    """Majority-vote accuracy at one budget, split by `key` (category / length_type)."""
    groups: dict = collections.defaultdict(list)
    for u, d in items.items():
        if d["gold"]:
            groups[d[key]].append(u)
    out = []
    for g, uids in sorted(groups.items()):
        corr = 0
        for u in uids:
            letters = [s.get("pred_letter", "") for s in items[u]["samples"]]
            pick = letters if budget >= len(letters) else rng.sample(letters, budget)
            v, _ = majority(pick, rng)
            corr += v == items[u]["gold"]
        out.append((g, corr, len(uids), corr / max(len(uids), 1)))
    return out


def mcnemar(items_a: dict, items_b: dict) -> dict | None:
    """Paired test over the common gradeable items (arms share every item)."""
    common = sorted(set(items_a) & set(items_b))
    common = [u for u in common if items_a[u]["gold"] and items_b[u]["gold"]]
    if not common:
        return None

    def hit(d):
        return d["samples"][0].get("pred_letter", "") == d["gold"]

    b = sum(1 for u in common if hit(items_a[u]) and not hit(items_b[u]))
    c = sum(1 for u in common if not hit(items_a[u]) and hit(items_b[u]))
    if b + c == 0:
        return {"n": len(common), "b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    return {"n": len(common), "b": b, "c": c, "chi2": chi2,
            "p": math.erfc(math.sqrt(chi2 / 2))}


@click.command()
@click.option("--jsonl", "patterns", multiple=True, required=True,
              help="glob(s) of sample_runner output; repeatable")
@click.option("--replicates", default=DEFAULT_REPLICATES, type=int)
@click.option("--seed", default=0, type=int)
@click.option("--budgets", default=",".join(str(b) for b in BUDGETS))
@click.option("--out", "out_path", default=None, help="write the report as markdown")
def main(patterns, replicates, seed, budgets, out_path):
    rows, files = load_rows(list(patterns))
    if not rows:
        raise SystemExit("no rows matched")
    cells = group(rows)
    buds = [int(x) for x in budgets.split(",") if x.strip()]
    rng = random.Random(seed)
    lines: list[str] = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # The same scorer serves both sweeps (identical row schema), so label the report
    # after whichever tree the samples came from rather than hardcoding MMAR.
    joined = " ".join(str(p) for p in files) + " " + str(out_path or "")
    bench = "MMAU" if "mmau" in joined.lower() else "MMAR"

    emit(f"# {bench} — greedy (± CoT) and self-consistency")
    emit()
    emit(f"Read **{len(rows):,} sample rows** from {len(files)} file(s). "
         f"Subsampling: R={replicates} replicates without replacement, seed={seed}.")
    emit()
    emit("Conventions: ungradeable items (gold text matching no single choice) are excluded "
         "from every denominator; an unparsed sample casts **no vote**, and an item whose "
         "every vote is unparsed counts as **wrong**; ties are broken uniformly at random, "
         "not by option order.")
    emit()
    emit("Source files:")
    for p in files:
        emit(f"- `{p}`")
    emit()

    # ---- summary table across all cells --------------------------------------------
    emit("## Summary")
    emit()
    emit("| model | arm | method | temp | n | items | gradeable | majority @ max K "
         "| oracle @ max K | gap | parse rate | unparsed | truncated | errors "
         "| words (mean/p90) |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary_rows = []
    for key in sorted(cells):
        model, arm = key
        cell = cells[key]
        h = cell_health(cell)
        curve = budget_curve(cell["items"], buds, replicates, rng)
        top = curve[-1] if curve else None
        cfg = cell["cfg"]
        acc = f"{top['mean']:.4f}" if top else "—"
        if top and not top["exact"]:
            acc += f" ± {top['sd']:.4f}"
        # at n=1 the oracle IS the accuracy (one sample, nothing to select between) — show
        # a dash rather than a duplicate number so the greedy rows cannot be misread
        multi = top is not None and cfg.get("n_requested", 1) > 1
        ora = f"**{top['oracle']:.4f}**" if multi else "—"
        gap = f"{top['oracle'] - top['mean']:+.4f}" if multi else "—"
        emit(f"| {display(model)} | `{arm}` | P{cfg.get('method')} | {cfg.get('temperature')} "
             f"| {cfg.get('n_requested')} | {h['items']} | {h['gradeable']} | **{acc}** "
             f"| {ora} | {gap} "
             f"| {pct(h['parse_rate'])} | {h['unparsed']} | {pct(h['trunc_rate'])} "
             f"| {h['error_rows']} | {h['words_mean']:.1f} / {h['words_p90']:.0f} |")
        summary_rows.append((key, cell, h, curve))
    emit()

    # ---- per-cell detail -------------------------------------------------------------
    for (model, arm), cell, h, curve in summary_rows:
        items = cell["items"]
        cfg = cell["cfg"]
        emit(f"## {display(model)} — `{arm}`")
        emit()
        emit(f"prompt method **P{cfg.get('method')}**, temperature **{cfg.get('temperature')}**, "
             f"n **{cfg.get('n_requested')}** per item")
        emit()
        emit("### Coverage and integrity")
        emit()
        emit(f"- items **{h['items']}** — {h['gradeable']} gradeable, "
             f"{h['ungradeable']} ungradeable (excluded)")
        emit(f"- samples **{h['samples']:,}** "
             f"(per item min {h['samples_per_item_min']}, max {h['samples_per_item_max']})")
        emit(f"- error rows **{h['error_rows']}** across {h['error_items']} item(s)")
        if h["samples_per_item_min"] != h["samples_per_item_max"]:
            emit("- ⚠ ragged sample counts — some items have fewer samples than others")
        emit()
        emit("### Generation quality")
        emit()
        emit(f"- parse rate **{pct(h['parse_rate'])}** ({h['unparsed']:,} unparsed of {h['samples']:,})")
        emit(f"- items with **zero** parseable votes: **{h['no_vote_items']}** (scored wrong)")
        emit(f"- truncated (`finish_reason=length`): **{pct(h['trunc_rate'])}**")
        emit(f"- finish reasons: {dict(h['finish'].most_common())}")
        emit(f"- response words: mean **{h['words_mean']:.1f}**, median {h['words_med']:.0f}, "
             f"p90 {h['words_p90']:.0f}, max {h['words_max']}")
        emit(f"- prompt tokens: mean {h['ptok_mean']:.0f}, max {h['ptok_max']}")
        emit(f"- median per-item latency: {h['lat_med']:.2f}s")
        if cfg.get("n_requested", 1) > 1:
            emit(f"- mean vote concentration (share on modal letter): "
                 f"**{pct(h['concentration'])}** — 100% would mean no diversity at all")
        emit()
        emit("### Answer distribution (position-bias check)")
        emit()
        gd = h["gold_dist"]
        pd = h["pred_dist"]
        tot_p = sum(pd.values()) or 1
        tot_g = sum(gd.values()) or 1
        keys = sorted(set(gd) | set(pd), key=lambda x: (x == "-", x))
        emit("| letter | gold items | predicted samples |")
        emit("|---|---|---|")
        for k in keys:
            emit(f"| {k} | {gd.get(k, 0)} ({100 * gd.get(k, 0) / tot_g:.1f}%) "
                 f"| {pd.get(k, 0)} ({100 * pd.get(k, 0) / tot_p:.1f}%) |")
        emit()
        emit(f"- choices per item: {dict(sorted(h['nchoices'].items(), key=lambda kv: (kv[0] is None, kv[0])))}")
        emit()

        if not curve:
            emit("_no gradeable items_")
            emit()
            continue

        if cfg.get("n_requested", 1) == 1:
            emit(f"### Accuracy: **{curve[0]['mean']:.4f}** ({curve[0]['n']} gradeable)")
            emit()
        else:
            emit("### Accuracy vs. self-consistency budget")
            emit()
            emit("| budget K | majority acc | sd | min | max | **oracle acc** | oracle sd "
                 "| gap | tie rate | R |")
            emit("|---|---|---|---|---|---|---|---|---|---|")
            for c in curve:
                tag = " *(exact)*" if c["exact"] else ""
                emit(f"| {c['budget']} | **{c['mean']:.4f}**{tag} | {c['sd']:.4f} "
                     f"| {c['lo']:.4f} | {c['hi']:.4f} | **{c['oracle']:.4f}** "
                     f"| {c['oracle_sd']:.4f} | {c['oracle'] - c['mean']:+.4f} "
                     f"| {pct(c['tie_rate'])} | {c['replicates']} |")
            emit()
            g, b = curve[0]["mean"], curve[-1]["mean"]
            og, ob = curve[0]["oracle"], curve[-1]["oracle"]
            emit(f"- majority gain K={curve[0]['budget']}→{curve[-1]['budget']}: "
                 f"**{b - g:+.4f}** ({100 * (b - g):+.2f} pts)")
            emit(f"- oracle gain K={curve[0]['budget']}→{curve[-1]['budget']}: "
                 f"**{ob - og:+.4f}** ({100 * (ob - og):+.2f} pts)")
            emit(f"- **selection gap at K={curve[-1]['budget']}: {ob - b:+.4f}** "
                 f"({100 * (ob - b):+.2f} pts) — headroom a perfect selector would capture "
                 f"over plain majority vote; this is what PF/EPF competes for")
            emit()

        maxk = curve[-1]["budget"]
        for field, title in ((("category", f"{bench} modality")), ("length_type", "clip duration")):
            bd = breakdown(items, rng, maxk, field)
            if len(bd) > 1:
                emit(f"### By {title} (at K={maxk})")
                emit()
                emit(f"| {field} | correct | n | accuracy |")
                emit("|---|---|---|---|")
                for g_, corr, n_, a in bd:
                    emit(f"| {g_} | {corr} | {n_} | {a:.4f} |")
                emit()

        le30 = {u for u, d in items.items() if d["length_type"] in ("le10s", "10to30s")}
        if 0 < len(le30) < len(items):
            sub = budget_curve(items, [maxk], replicates, rng, uid_filter=le30)
            if sub:
                emit(f"- **le30s cut** ({len(le30)} items — what Qwen2-Audio's 30 s encoder "
                     f"window can fully hear): K={maxk} accuracy **{sub[0]['mean']:.4f}**")
                emit()

    # ---- paired comparisons ----------------------------------------------------------
    emit("## Paired comparisons (McNemar, continuity-corrected)")
    emit()
    any_mc = False
    for model in sorted({m for m, _ in cells}):
        arms = {a: cells[(m, a)]["items"] for m, a in cells if m == model}
        g_no = arms.get("greedy_nocot")
        g_cot = next((v for k, v in sorted(arms.items()) if k.startswith("greedy_p")), None)
        if g_no and g_cot:
            mc = mcnemar(g_cot, g_no)
            if mc:
                any_mc = True
                emit(f"- **{display(model)}** greedy CoT vs greedy no-CoT: n={mc['n']}, "
                     f"CoT-only-correct b={mc['b']}, noCoT-only-correct c={mc['c']}, "
                     f"chi2={mc['chi2']:.2f}, **p={mc['p']:.4g}**")
    if not any_mc:
        emit("_(needs both greedy arms for a model)_")
    emit()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
