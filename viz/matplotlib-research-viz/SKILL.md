---
name: matplotlib-research-viz
description: >
  Apply a consistent, publication-quality matplotlib style to research visualizations.
  Use this skill whenever the user wants to create or style matplotlib charts, plots,
  bar charts, line charts, win/loss charts, or any scientific/academic figure — especially
  when they mention BERTScore, BLEU, VQA, win-rate comparisons, rating distributions,
  dual-axis plots, or any paper/research visualization. Also trigger when they ask to
  "make a plot like the others", "match the style of my figures", or "create a chart
  with hatching/colorblind palette". Always use this skill before writing any matplotlib code.
---

# Matplotlib Research Visualization Style Guide

This skill encodes the visual style used across a research paper's evaluation figures.
Apply it consistently across all chart types.

---

## Color Palette (CUD Colorblind-Friendly)

```python
BERT_COLOR  = "#0072B2"   # Blue      — BERTScore, Win bars, primary metric
VQA_COLOR   = "#D55E00"   # Vermillion — VQA, Loss bars, secondary metric  
BLEU_COLOR  = "#009E73"   # Green     — BLEU, Merged/combined metrics
LOSS_COLOR  = "#D55E00"   # Vermillion (same as VQA_COLOR for win/loss charts)
WIN_COLOR   = "#0072B2"   # Blue      (same as BERT_COLOR for win/loss charts)
```

Assign colors **by semantic role**, not arbitrary mapping. BERTScore is always blue,
BLEU is always green, VQA/Loss is always vermillion.

---

## Global Style Rules

### Spines: solid left + bottom axis lines; hide top + right
Keep the left and bottom spines visible as **solid black axis lines** (they define the
x and y axes); hide only the top and right.
```python
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color("black")
    ax.spines[side].set_linewidth(1.0)
```
For **twin-axis** charts, also keep the **right** spine visible (it carries the second
y-axis); still hide the top on both axes.

### No in-plot title or caption
Do **not** annotate the axes with a title/caption identifying the model, prompt, dataset,
etc. Identify the figure via its **filename** and the **legend** instead — keep the plot
area clean.

### Gridlines: y-axis only (or x-axis only for horizontal bar charts)
```python
# Vertical charts (bar, line):
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)

# Horizontal bar charts:
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)
```

Set `zorder=3` on all bars/lines so they render above the grid.

### Legend: always above the plot, no frame
```python
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05),
           ncol=2, frameon=False, fontsize=11)
```
Use `ncol=1` for single-series legends. Use `fig.legend` (not `ax.legend`) when there
are twin axes.

### Hatching on bars
All bars use hatching for print/greyscale legibility:
- Primary / Win / BERTScore bars: `hatch="///"`
- Secondary / Loss / BLEU / VQA bars: `hatch="..."`
- Always pair with `edgecolor="black"`

### Error bars: always show the per-point spread (std)
Every data point carries a **±1 standard-deviation** error bar. For the EPF benchmark
figures this is the **std100** value — the per-cell 100-item bootstrap SD reported in each
run's `*_bootstrap.html` (the `s1` ±value inside every table cell). Match the bar color to
its series; use small caps.
```python
ax.errorbar(x, values, yerr=std,
            marker="o", linewidth=2.0, markersize=6,
            color=SERIES_COLOR, markeredgecolor="black", markeredgewidth=1.0,
            capsize=3, capthick=1.0, elinewidth=1.0, zorder=3, label="...")
# bar charts: ax.bar(..., yerr=std, capsize=3, ecolor="black")
```

### Y-axis range: auto-fit to the data (± error bars), snapped to multiples of 0.1
Do **not** hard-code the y-limits — different models have different baselines (a strong
model sits ~0.55, a weaker one ~0.42), so any fixed window either clips one or wastes space
on another. Instead fit the range to the drawn data **including its error bars**, then round
**outward** to the nearest 0.1 so the limits are always clean multiples of 10% and the bars
never clip:
```python
import math
def auto_ylim(series):  # series: list of (values, errors)
    lo = min(v - e for vals, errs in series for v, e in zip(vals, errs))
    hi = max(v + e for vals, errs in series for v, e in zip(vals, errs))
    ylo, yhi = math.floor(lo * 10) / 10, math.ceil(hi * 10) / 10
    if lo - ylo < 1e-9: ylo -= 0.1   # a tip landed on a gridline -> keep a 0.1 gap
    if yhi - hi < 1e-9: yhi += 0.1
    return max(0.0, ylo), min(1.0, yhi)

ax.set_ylim(*auto_ylim([(selected, sel_err), (oracle, orc_err)]))
```
This yields `(min − k, max + k)` where `k` is whatever it takes to reach the next 0.1
gridline (0–0.1), clamped to [0, 1] — no manual tuning per model. **Trade-off:** auto-ranged
plots then have *different* y-windows, so don't compare two of them by eye height — read the
axis. (If a set of figures must be visually compared, hard-code a shared range instead.)

### Save settings
```python
fig.savefig("assets/output.pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
```
Always PDF, 600 dpi, tight bbox.

---

## Chart Templates

### 1. Dual-Axis Bar Chart (two metrics, different scales)

Use when: two metrics with different y-ranges need to appear side-by-side
(e.g., BERTScore ~86–92 vs. Relative VQAScore ~0–1).

```python
x = np.arange(len(LABELS))
bar_width = 0.35
fig, ax1 = plt.subplots(figsize=(5, 3.5))

# Left axis — primary metric (e.g., BERTScore)
ax1.bar(x - bar_width/2, primary_values, width=bar_width,
        color=BERT_COLOR, hatch="///", edgecolor="black", zorder=3, label="BERTScore")
ax1.set_ylabel("BERTScore (↑)", fontsize=13, color=BERT_COLOR)
ax1.tick_params(axis="y", labelcolor=BERT_COLOR, labelsize=11)
ax1.set_ylim(85, 92)  # tight range to highlight variation

# Right axis — secondary metric
ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, secondary_values, width=bar_width,
        color=VQA_COLOR, hatch="...", edgecolor="black", zorder=3, label="Relative VQAScore")
ax2.set_ylabel("Relative VQAScore (↑)", fontsize=13, color=VQA_COLOR)
ax2.tick_params(axis="y", labelcolor=VQA_COLOR, labelsize=11)
ax2.set_ylim(0, max(secondary_values) * 1.1)

ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=12)
ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)

# ax1 owns the left axis, ax2 owns the right; bottom is shared. Keep those solid, hide the rest.
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax1.spines[side].set_visible(True); ax1.spines[side].set_color("black"); ax1.spines[side].set_linewidth(1.0)
for side in ("top", "left", "bottom"):
    ax2.spines[side].set_visible(False)
ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color("black"); ax2.spines["right"].set_linewidth(1.0)

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=11)
fig.tight_layout()
```

### 2. Dual-Axis Line Chart (same as above but lines)

Swap bars for lines with consistent markers:

```python
ax1.plot(x, primary_values, marker='o', linewidth=2.0, markersize=6,
         color=BERT_COLOR, markeredgecolor="black", markeredgewidth=1.0, zorder=3, label="BERTScore")

ax2.plot(x, secondary_values, marker='s', linewidth=2.0, markersize=6,
         color=VQA_COLOR, markeredgecolor="black", markeredgewidth=1.0, zorder=3, label="Relative VQAScore")
```

Marker guide: `'o'` circle (primary), `'s'` square (secondary), `'d'` diamond (tertiary/BLEU).

### 3. Single-Metric Bar Chart

```python
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(x, values, width=0.6,
       color=BLEU_COLOR, hatch="...", edgecolor="black", zorder=3, label="BLEU")
ax.set_ylabel("BLEU (↑)", fontsize=13, color=BLEU_COLOR)
ax.tick_params(axis="y", labelcolor=BLEU_COLOR, labelsize=11)
ax.set_ylim(0, max(values) * 1.1)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=12)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_visible(True); ax.spines[side].set_color("black"); ax.spines[side].set_linewidth(1.0)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False, fontsize=11)
fig.tight_layout()
```

### 4. Single-Metric Line Chart

Same as #3 but use `ax.plot(...)` with appropriate marker. Use `width=0.6` for single bar charts.

### 5. Horizontal Stacked Win/Loss Bar Chart

Use when: comparing win rates against multiple opponents.

```python
fig, ax = plt.subplots(figsize=(6, 3.8))
y = np.arange(len(opponents))
bar_height = 0.55

ax.barh(y, wins, height=bar_height, color=WIN_COLOR, hatch="///",
        edgecolor="black", label="Win", zorder=3)
ax.barh(y, losses, height=bar_height, left=wins,
        color=LOSS_COLOR, hatch="xx", edgecolor="black", label="Loss", zorder=3)

# Inline labels: wins inside (white), losses outside (black)
for i, (w, l) in enumerate(zip(wins, losses)):
    if w > 1:
        ax.text(w/2, i, f"{w:.1f}%", ha="center", va="center",
                fontsize=11, color="white", weight="bold")
    if l > 1:
        ax.text(w + l + 1, i, f"{l:.1f}%", ha="left", va="center",
                fontsize=11, color="black", weight="bold")

ax.set_yticks(y)
ax.set_yticklabels([f"vs {opp}" for opp in opponents], fontsize=11)
ax.set_xlim(0, 105)  # extra room for outside labels
ax.set_xlabel("Model Win Rate (%)", fontsize=12)
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_visible(True); ax.spines[side].set_color("black"); ax.spines[side].set_linewidth(1.0)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=12)
fig.tight_layout()
```

Sort opponents by win rate descending before plotting:
```python
sorted_idx = np.argsort(wins)[::-1]
opponents = [opponents[i] for i in sorted_idx]
wins = [wins[i] for i in sorted_idx]
losses = [losses[i] for i in sorted_idx]
```

---

## Figure Size Reference

| Chart type                  | figsize      |
|-----------------------------|-------------|
| Single-metric bar/line      | `(5, 3.5)`  |
| Dual-axis bar/line          | `(5, 3.5)`  |
| Horizontal win/loss bars    | `(6, 3.8)`  |
| Larger win/loss (many rows) | `(6.5, 4.0)`|

---

## Font Sizes

| Element          | fontsize |
|------------------|----------|
| Axis labels      | 12–13    |
| Tick labels      | 11–12    |
| Legend           | 11–12    |
| Inline bar text  | 11–12    |
| Bold labels      | 14 (win/loss x-axis only) |

---

## Rating Pair Labels

When x-axis shows rating comparisons, use the format `"5-4"`, `"5-3"`, `"5-2"`, `"5-1"`
(reference rating minus comparison rating). Label the axis as `"Comparison (5 - r)"` or
omit the x-axis label if self-explanatory.

---

## Checklist Before Saving

- [ ] Left + bottom spines solid black; top + right hidden (twin axis: keep its right spine too)
- [ ] No in-plot title/caption — identify via filename + legend
- [ ] Every data point has a ±1 std (std100) error bar, colored to its series
- [ ] Y-axis auto-fit to data ± error bars, snapped outward to multiples of 0.1 (not hard-coded)
- [ ] Grid on the correct axis only, `zorder=0`
- [ ] Bars/lines at `zorder=3`
- [ ] Hatching on every bar (`///` or `...` or `xx`)
- [ ] Legend above the plot via `fig.legend(..., bbox_to_anchor=(0.5, 1.05))`
- [ ] `fig.tight_layout()` called before `fig.savefig`
- [ ] Saved as PDF, `dpi=600`, `bbox_inches="tight"`, `pad_inches=0.05`
- [ ] Colors match semantic role (blue=BERT/Win, vermillion=VQA/Loss, green=BLEU)
