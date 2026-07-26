"""
plot_genome_freq.py
====================
For each unique complete genome structure:
  - Count how many times it appeared across all generations (frequency)
  - Show its fitness score

Usage:
    python plot_genome_freq.py --all genome_all.csv --outdir ./plots
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DIM_COLORS = {
    "K": "#E63946",
    "C": "#457B9D",
    "Y": "#2A9D8F",
    "X": "#E9C46A",
    "R": "#F4A261",
    "S": "#A8DADC",
}


def parse_genome_str(genome_str):
    """
    Parse genome_str like:
      'L1:[sp=Y,511] S=3,X=55,Y=4,R=3,K=33,C=34 || L2:[sp=K,4] S=1,C=1,Y=1,X=1,R=1,K=1'
    Returns list of levels: [{'sp_dim','sp_sz','order':[(dim,tile),...]}]
    """
    levels = []
    for part in genome_str.split(" || "):
        part = part.strip()
        bracket_end = part.index("]")
        sp_part = part[part.index("[")+1: bracket_end]
        _, sp_rest = sp_part.split("=", 1)
        sp_dim, sp_sz = sp_rest.split(",")
        rest = part[bracket_end+1:].strip()
        order = []
        for token in rest.split(","):
            token = token.strip()
            if "=" in token:
                d, sz = token.split("=")
                order.append((d.strip(), int(sz.strip())))
        levels.append({
            "sp_dim": sp_dim.strip(),
            "sp_sz":  int(sp_sz.strip()),
            "order":  order
        })
    return levels


def short_label(genome_str, idx):
    """
    Make a short readable label for each genome:
    e.g.  #3  Y(511)|K(4)
    """
    try:
        levels = parse_genome_str(genome_str)
        parts = [f"{l['sp_dim']}({l['sp_sz']})" for l in levels]
        return f"#{idx+1}  " + "|".join(parts)
    except Exception:
        return f"#{idx+1}"


def plot_freq_vs_fitness(df_all, outdir):
    """
    Main plot:
    - Each unique genome_str is one entry
    - Bar height    = fitness1 (latency reward, closer to 0 = better)
    - Bar width     = proportional to frequency (how many times it appeared)
    - Color         = genome_pattern (sp-dim combo like Y|K)
    - Annotated     = count on top of each bar
    Also prints a clear table to terminal.
    """
    df = df_all.copy()
    df["fitness1"] = pd.to_numeric(df["fitness1"], errors="coerce")
    df["valid"]    = df["valid"].astype(int)

    # Count frequency of each unique genome_str
    freq = df.groupby("genome_str").size().rename("count")
    # Best fitness per genome_str
    best_fit = df.groupby("genome_str")["fitness1"].max().rename("best_fitness1")
    # Pattern label
    pattern  = df.groupby("genome_str")["genome_pattern"].first()

    summary = pd.concat([freq, best_fit, pattern], axis=1).reset_index()
    summary = summary.sort_values("best_fitness1", ascending=False).reset_index(drop=True)

    # ── Print table ────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'#':<4} {'Pattern':<8} {'Count':>6}  {'Best Fitness1':>15}  {'Genome Structure'}")
    print("="*90)
    for idx, row in summary.iterrows():
        label = short_label(row["genome_str"], idx)
        f1 = row["best_fitness1"]
        f1_str = f"{f1:.0f}" if f1 > -1e17 else "INVALID"
        print(f"{idx+1:<4} {row['genome_pattern']:<8} {int(row['count']):>6}  {f1_str:>15}  {row['genome_str'][:60]}")
    print("="*90)
    print(f"Total unique genome structures: {len(summary)}")
    print(f"Total evaluations:              {len(df)}\n")

    # ── Plot ───────────────────────────────────────────────────────────────
    n = len(summary)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(14, n * 0.55), 10),
        facecolor="#0d0d1a",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    for ax in [ax_top, ax_bot]:
        ax.set_facecolor("#0d0d1a")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="#aaa")

    # Assign a color per unique pattern
    unique_patterns = summary["genome_pattern"].unique()
    cmap = plt.cm.get_cmap("tab10", len(unique_patterns))
    pat_color = {p: cmap(i) for i, p in enumerate(unique_patterns)}

    bar_colors = [pat_color[p] for p in summary["genome_pattern"]]
    valid_mask = summary["best_fitness1"] > -1e17
    # make invalid bars clearly different
    bar_colors_final = [
        c if v else "#555"
        for c, v in zip(bar_colors, valid_mask)
    ]

    x = np.arange(n)
    counts = summary["count"].values
    fitness_vals = summary["best_fitness1"].values
    fitness_plot = np.where(valid_mask, fitness_vals, 0)

    # ── TOP: fitness bars ──────────────────────────────────────────────────
    bars = ax_top.bar(x, fitness_plot, color=bar_colors_final,
                      edgecolor="#222", linewidth=0.6)

    # annotate count on each bar
    for xi, (bar, cnt, fv, valid) in enumerate(zip(bars, counts, fitness_vals, valid_mask)):
        ypos = bar.get_height() if fv < 0 else 0
        ax_top.text(xi, ypos - abs(ypos)*0.03,
                    f"×{cnt}", ha="center", va="top",
                    fontsize=6.5, color="white", fontweight="bold")
        if not valid:
            ax_top.text(xi, -abs(ax_top.get_ylim()[0])*0.05,
                        "INV", ha="center", va="top",
                        fontsize=6, color="#E63946")

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(
        [short_label(row["genome_str"], idx) for idx, row in summary.iterrows()],
        rotation=60, ha="right", fontsize=6.5, color="#ccc", fontfamily="monospace"
    )
    ax_top.set_ylabel("Best fitness1 (latency reward)\ncloser to 0 = faster / better",
                       color="#aaa", fontsize=9)
    ax_top.axhline(0, color="#555", linewidth=0.8)
    ax_top.set_title(
        "Unique Genome Structures — Fitness & Frequency\n"
        "Bar height = best fitness  |  ×N = how many times this genome appeared across all generations  |  color = sp-dim pattern",
        color="white", fontsize=10, pad=10
    )

    # ── BOTTOM: frequency bars ─────────────────────────────────────────────
    ax_bot.bar(x, counts, color=bar_colors_final,
               edgecolor="#222", linewidth=0.6, alpha=0.85)
    for xi, cnt in enumerate(counts):
        ax_bot.text(xi, cnt + 0.1, str(cnt),
                    ha="center", va="bottom", fontsize=6, color="white")

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([])
    ax_bot.set_ylabel("Frequency\n(# appearances)", color="#aaa", fontsize=8)
    ax_bot.set_xlabel("Each bar = one unique complete genome structure", color="#aaa", fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=pat_color[p], label=f"pattern: {p}")
        for p in unique_patterns
    ]
    legend_patches.append(mpatches.Patch(color="#555", label="invalid (MAESTRO rejected)"))
    ax_top.legend(handles=legend_patches, fontsize=7,
                  facecolor="#1a1a2e", edgecolor="#444",
                  labelcolor="white", loc="lower right")

    out_path = os.path.join(outdir, "genome_freq_fitness.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"[plot] Saved → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    default="genome_all.csv",  help="Path to genome_all.csv")
    parser.add_argument("--outdir", default=".",               help="Where to save the plot")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {args.all} ...")
    df_all = pd.read_csv(args.all)
    plot_freq_vs_fitness(df_all, args.outdir)


if __name__ == "__main__":
    main()
