#!/usr/bin/env python3
"""
Comprehensive analysis of VotIE incremental LOMO experiments.
Generates presentation-ready tables (LaTeX + Markdown) and descriptive graphics.
Handles municipalities with different numbers of completed steps.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
EVAL_DIR = BASE_DIR / "evaluation"
OUT_DIR = BASE_DIR / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

MUNICIPALITIES = ["M01", "M02", "M03", "M04", "M05", "M06"]
MUNICIPALITY_NAMES = {
    "M01": "Alandroal",
    "M02": "Campomaior",
    "M03": "Porto",
    "M04": "Fundão",
    "M05": "Covilhã",
    "M06": "Guimarães",
}
MAX_STEPS = 6  # steps 0-5

ENTITY_TYPES = [
    "VOTING",
    "COUNTING-UNANIMITY",
    "COUNTING-MAJORITY",
    "VOTER-FAVOR",
    "VOTER-ABSTENTION",
    "VOTER-AGAINST",
    "VOTER-ABSENT",
    "SUBJECT",
]

# Presentation colors
MUNICIPALITY_COLORS = {
    "M01": "#2196F3",  # blue
    "M02": "#FF9800",  # orange
    "M03": "#4CAF50",  # green
    "M04": "#E91E63",  # pink
    "M05": "#9C27B0",  # purple
    "M06": "#009688",  # teal
}

MUNICIPALITY_MARKERS = {
    "M01": "o",
    "M02": "s",
    "M03": "^",
    "M04": "D",
    "M05": "v",
    "M06": "P",
}

# ── Style setup ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ──────────────────────────────────────────────────────────────────
def load_all_results():
    """Load all evaluation JSON files into a structured dict."""
    data = {}
    for muni in MUNICIPALITIES:
        data[muni] = {}
        for step in range(MAX_STEPS):
            fpath = EVAL_DIR / f"{muni}_step_{step:02d}_results.json"
            if fpath.exists():
                with open(fpath) as f:
                    data[muni][step] = json.load(f)
    return data


def get_available_steps(data, muni):
    """Return sorted list of available steps for a municipality."""
    return sorted(data[muni].keys())


def get_max_step(data, muni):
    """Return highest available step for a municipality."""
    steps = get_available_steps(data, muni)
    return steps[-1] if steps else -1


def extract_overall_metrics(data):
    """Extract overall F1/P/R into dicts with step keys."""
    metrics = {}
    for muni in MUNICIPALITIES:
        metrics[muni] = {"f1": {}, "precision": {}, "recall": {}, "n_examples": {}}
        for step in get_available_steps(data, muni):
            res = data[muni][step]
            em = res["entity_level_metrics"]
            metrics[muni]["f1"][step] = em["entity_f1"]
            metrics[muni]["precision"][step] = em["entity_precision"]
            metrics[muni]["recall"][step] = em["entity_recall"]
            metrics[muni]["n_examples"][step] = res["metadata"]["num_examples"]
    return metrics


def extract_per_entity_metrics(data):
    """Extract per-entity-type F1 for each municipality/step."""
    per_entity = {}
    for muni in MUNICIPALITIES:
        per_entity[muni] = {}
        for step in get_available_steps(data, muni):
            em = data[muni][step]["entity_level_metrics"]["per_type_metrics"]
            per_entity[muni][step] = {}
            for etype in ENTITY_TYPES:
                if etype in em:
                    per_entity[muni][step][etype] = em[etype]
                else:
                    per_entity[muni][step][etype] = {
                        "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0
                    }
    return per_entity


# ── TABLE GENERATION ───────────────────────────────────────────────────────────

def generate_main_results_table_latex(data, metrics):
    """Generate LaTeX table: Overall F1 learning curve by municipality."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Incremental document addition: Entity F1 scores (\%) across municipalities. Step 0 is the pure LOMO baseline (zero in-domain documents). Each subsequent step adds one document from the target municipality to the training set. ``--'' indicates experiment not yet completed.}")
    lines.append(r"\label{tab:incremental_f1}")
    lines.append(r"\small")
    ncols = len(MUNICIPALITIES)
    lines.append(r"\begin{tabular}{c" + "c" * ncols + r"c}")
    lines.append(r"\toprule")

    header = r"\textbf{Step} & " + " & ".join(
        [rf"\textbf{{{MUNICIPALITY_NAMES[m]}}}" for m in MUNICIPALITIES]
    ) + r" & \textbf{Avg.} \\"
    lines.append(header)
    lines.append(r"\midrule")

    for step in range(MAX_STEPS):
        vals = []
        row_parts = [str(step)]
        for m in MUNICIPALITIES:
            if step in metrics[m]["f1"]:
                v = metrics[m]["f1"][step]
                vals.append(v)
                row_parts.append(f"{v*100:.1f}")
            else:
                row_parts.append("--")
        if vals:
            avg = np.mean(vals)
            row_parts.append(f"{avg*100:.1f}")
        else:
            row_parts.append("--")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\midrule")
    # Delta row (last step - step 0)
    delta_parts = [r"$\Delta$ (0$\to$last)"]
    deltas = []
    for m in MUNICIPALITIES:
        max_s = get_max_step(data, m)
        if max_s > 0 and 0 in metrics[m]["f1"]:
            d = metrics[m]["f1"][max_s] - metrics[m]["f1"][0]
            deltas.append(d)
            sign = "+" if d >= 0 else ""
            delta_parts.append(f"{sign}{d*100:.1f}")
        else:
            delta_parts.append("--")
    if deltas:
        avg_d = np.mean(deltas)
        sign = "+" if avg_d >= 0 else ""
        delta_parts.append(f"{sign}{avg_d*100:.1f}")
    else:
        delta_parts.append("--")
    lines.append(" & ".join(delta_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_precision_recall_table_latex(data, metrics):
    """Generate LaTeX table: Precision and Recall at step 0 vs last step."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Precision and Recall comparison: LOMO baseline (Step 0) vs.\ after adding in-domain documents (last completed step).}")
    lines.append(r"\label{tab:incremental_pr}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c cc cc c}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{\textbf{Step 0 (LOMO)}} & \multicolumn{2}{c}{\textbf{Last Step}} & \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    lines.append(r"\textbf{Municipality} & \textbf{Steps} & \textbf{Prec.} & \textbf{Rec.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{$\Delta$ F1} \\")
    lines.append(r"\midrule")

    for m in MUNICIPALITIES:
        max_s = get_max_step(data, m)
        p0 = metrics[m]["precision"].get(0, 0) * 100
        r0 = metrics[m]["recall"].get(0, 0) * 100
        pL = metrics[m]["precision"].get(max_s, 0) * 100
        rL = metrics[m]["recall"].get(max_s, 0) * 100
        f0 = metrics[m]["f1"].get(0, 0)
        fL = metrics[m]["f1"].get(max_s, 0)
        delta_f1 = (fL - f0) * 100
        sign = "+" if delta_f1 >= 0 else ""
        lines.append(
            f"{MUNICIPALITY_NAMES[m]} & 0$\\to${max_s} & {p0:.1f} & {r0:.1f} & {pL:.1f} & {rL:.1f} & {sign}{delta_f1:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_per_entity_table_latex(data, per_entity):
    """Generate LaTeX table: Per-entity F1 at Step 0 vs last step for each municipality."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-entity F1 scores (\%) at Step 0 (LOMO baseline) and last completed step. Entities with support $<$ 5 marked with $\dagger$.}")
    lines.append(r"\label{tab:per_entity_f1}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l" + " cc" * len(MUNICIPALITIES) + r"}")
    lines.append(r"\toprule")

    # Multi-column header
    header1 = r"\textbf{Entity Type}"
    for m in MUNICIPALITIES:
        max_s = get_max_step(data, m)
        header1 += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{MUNICIPALITY_NAMES[m]}}}}}"
    header1 += r" \\"
    lines.append(header1)

    # Cmidrules
    cmidrules = ""
    for i in range(len(MUNICIPALITIES)):
        col_start = 2 + i * 2
        col_end = col_start + 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
    lines.append(cmidrules)

    # Sub-header with step numbers
    subheader = ""
    for m in MUNICIPALITIES:
        max_s = get_max_step(data, m)
        subheader += rf" & \textbf{{S0}} & \textbf{{S{max_s}}}"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    for etype in ENTITY_TYPES:
        short_name = etype.replace("COUNTING-", "C-").replace("VOTER-", "V-")
        row = short_name
        for m in MUNICIPALITIES:
            max_s = get_max_step(data, m)
            e0 = per_entity[m].get(0, {}).get(etype, {})
            eL = per_entity[m].get(max_s, {}).get(etype, {})

            f1_0 = e0.get("f1_score", 0.0) * 100
            f1_L = eL.get("f1_score", 0.0) * 100
            sup_0 = e0.get("support", 0)
            sup_L = eL.get("support", 0)

            s0_str = f"{f1_0:.1f}"
            sL_str = f"{f1_L:.1f}"
            if sup_0 < 5:
                s0_str += r"$^\dagger$"
            if sup_L < 5:
                sL_str += r"$^\dagger$"

            row += f" & {s0_str} & {sL_str}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_markdown_summary(data, metrics, per_entity):
    """Generate a complete markdown summary."""
    lines = ["# Incremental LOMO Experiment Results\n"]
    lines.append("**Model:** XLM-R + CRF (xlm-roberta-base)  ")
    lines.append("**Setup:** Leave-One-Municipality-Out with incremental document addition  ")
    lines.append("**Seed:** 42  ")
    lines.append(f"**Municipalities:** {len(MUNICIPALITIES)} ({', '.join(MUNICIPALITY_NAMES[m] for m in MUNICIPALITIES)})\n")

    # Summary of available data
    lines.append("## Data Availability\n")
    lines.append("| Municipality | Code | Steps Completed |")
    lines.append("|-------------|------|-----------------|")
    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        lines.append(f"| {MUNICIPALITY_NAMES[m]} | {m} | 0-{max(steps)} ({len(steps)} steps) |")

    lines.append(f"\n## Overall Entity F1 (%) by Municipality and Step\n")
    header = "| Step | " + " | ".join([f"{MUNICIPALITY_NAMES[m]}" for m in MUNICIPALITIES]) + " | Avg |"
    lines.append(header)
    lines.append("|------|" + "|".join(["------"] * (len(MUNICIPALITIES) + 1)) + "|")
    for step in range(MAX_STEPS):
        vals = []
        parts = [f"| {step}"]
        for m in MUNICIPALITIES:
            if step in metrics[m]["f1"]:
                v = metrics[m]["f1"][step]
                vals.append(v)
                parts.append(f"{v*100:.1f}%")
            else:
                parts.append("--")
        if vals:
            parts.append(f"{np.mean(vals)*100:.1f}%")
        else:
            parts.append("--")
        lines.append(" | ".join(parts) + " |")

    lines.append(f"\n## Improvement (Step 0 → Last Step)\n")
    lines.append("| Municipality | Steps | F1 Start | F1 End | Delta (pp) | Relative Gain |")
    lines.append("|-------------|-------|----------|--------|------------|---------------|")
    for m in MUNICIPALITIES:
        max_s = get_max_step(data, m)
        f0 = metrics[m]["f1"].get(0, 0)
        fL = metrics[m]["f1"].get(max_s, 0)
        delta = fL - f0
        rel = delta / f0 * 100 if f0 > 0 else 0
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {MUNICIPALITY_NAMES[m]} | 0→{max_s} | {f0*100:.1f}% | {fL*100:.1f}% | {sign}{delta*100:.1f} | {sign}{rel:.1f}% |")

    lines.append(f"\n## Per-Entity F1 (Step 0 vs Last Step)\n")
    for m in MUNICIPALITIES:
        max_s = get_max_step(data, m)
        lines.append(f"\n### {MUNICIPALITY_NAMES[m]} ({m}) — Steps 0→{max_s}\n")
        lines.append("| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |")
        lines.append("|--------|---------|-----------|-------|-------------|---------------|")
        for etype in ENTITY_TYPES:
            e0 = per_entity[m].get(0, {}).get(etype, {})
            eL = per_entity[m].get(max_s, {}).get(etype, {})
            f0 = e0.get("f1_score", 0.0)
            fL = eL.get("f1_score", 0.0)
            s0 = e0.get("support", 0)
            sL = eL.get("support", 0)
            d = fL - f0
            sign = "+" if d >= 0 else ""
            lines.append(f"| {etype} | {f0*100:.1f}% | {fL*100:.1f}% | {sign}{d*100:.1f} | {s0} | {sL} |")

    return "\n".join(lines)


# ── GRAPHICS GENERATION ────────────────────────────────────────────────────────

def plot_learning_curves(data, metrics):
    """Main figure: F1 learning curves for all municipalities."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        f1_vals = [metrics[m]["f1"][s] * 100 for s in steps]
        ax.plot(steps, f1_vals, marker=MUNICIPALITY_MARKERS[m], linewidth=2.5,
                markersize=8, color=MUNICIPALITY_COLORS[m],
                label=f"{MUNICIPALITY_NAMES[m]} ({m})", zorder=3)

    # Average line (only over steps where all 6 munis have data)
    common_steps = sorted(set.intersection(*[set(get_available_steps(data, m)) for m in MUNICIPALITIES]))
    if common_steps:
        avg_vals = [np.mean([metrics[m]["f1"][s] for m in MUNICIPALITIES]) * 100 for s in common_steps]
        ax.plot(common_steps, avg_vals, marker="*", linewidth=2, markersize=8,
                color="black", linestyle="--", alpha=0.7, label="Average (all 6)", zorder=2)

    ax.set_xlabel("Number of In-Domain Documents Added")
    ax.set_ylabel("Entity F1 (%)")
    ax.set_title("Incremental Document Addition: Learning Curves")
    ax.set_xticks(range(MAX_STEPS))
    ax.set_xticklabels([str(s) for s in range(MAX_STEPS)])
    ax.set_ylim(20, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.savefig(OUT_DIR / "learning_curves_f1.pdf")
    fig.savefig(OUT_DIR / "learning_curves_f1.png")
    plt.close(fig)
    print("  Saved: learning_curves_f1.pdf / .png")


def plot_precision_recall_curves(data, metrics):
    """Precision and Recall learning curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for ax, metric_name, title in zip(axes, ["precision", "recall"], ["Precision", "Recall"]):
        for m in MUNICIPALITIES:
            steps = get_available_steps(data, m)
            vals = [metrics[m][metric_name][s] * 100 for s in steps]
            ax.plot(steps, vals, marker=MUNICIPALITY_MARKERS[m], linewidth=2.5,
                    markersize=7, color=MUNICIPALITY_COLORS[m],
                    label=f"{MUNICIPALITY_NAMES[m]}")
        ax.set_xlabel("In-Domain Documents Added")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(f"Entity {title}")
        ax.set_xticks(range(MAX_STEPS))
        ax.set_ylim(20, 100)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig.suptitle("Incremental Document Addition: Precision & Recall", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "precision_recall_curves.pdf")
    fig.savefig(OUT_DIR / "precision_recall_curves.png")
    plt.close(fig)
    print("  Saved: precision_recall_curves.pdf / .png")


def plot_improvement_bars(data, metrics):
    """Bar chart showing F1 improvement from step 0 to last available step."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(MUNICIPALITIES))
    width = 0.35

    f1_step0 = [metrics[m]["f1"].get(0, 0) * 100 for m in MUNICIPALITIES]
    f1_last = [metrics[m]["f1"].get(get_max_step(data, m), 0) * 100 for m in MUNICIPALITIES]
    max_steps_labels = [get_max_step(data, m) for m in MUNICIPALITIES]

    bars0 = ax.bar(x - width / 2, f1_step0, width, label="Step 0 (LOMO)",
                   color=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES], alpha=0.4,
                   edgecolor=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES], linewidth=1.5)
    bars_last = ax.bar(x + width / 2, f1_last, width,
                       label="Last Step",
                       color=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES],
                       edgecolor=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES], linewidth=1.5)

    # Value labels
    for bar in bars0:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9)
    for i, bar in enumerate(bars_last):
        h = bar.get_height()
        step_lbl = f"S{max_steps_labels[i]}"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}\n({step_lbl})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Delta annotations
    for i, m in enumerate(MUNICIPALITIES):
        delta = f1_last[i] - f1_step0[i]
        sign = "+" if delta >= 0 else ""
        color = "darkgreen" if delta > 0 else "red"
        ax.annotate(f"{sign}{delta:.1f}pp",
                    xy=(x[i], max(f1_last[i], f1_step0[i]) + 10),
                    ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_ylabel("Entity F1 (%)")
    ax.set_title("Impact of In-Domain Document Addition on Entity F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MUNICIPALITY_NAMES[m]}\n({m})" for m in MUNICIPALITIES])
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "improvement_bars.pdf")
    fig.savefig(OUT_DIR / "improvement_bars.png")
    plt.close(fig)
    print("  Saved: improvement_bars.pdf / .png")


def plot_per_entity_heatmap(data, per_entity):
    """Heatmap: Per-entity F1 at last step across municipalities."""
    matrix = []
    row_labels = []
    for etype in ENTITY_TYPES:
        row = []
        for m in MUNICIPALITIES:
            max_s = get_max_step(data, m)
            f1 = per_entity[m].get(max_s, {}).get(etype, {}).get("f1_score", 0.0) * 100
            sup = per_entity[m].get(max_s, {}).get(etype, {}).get("support", 0)
            row.append(f1 if sup >= 3 else np.nan)
        matrix.append(row)
        row_labels.append(etype)

    matrix = np.array(matrix)
    step_labels = [f"{MUNICIPALITY_NAMES[m]}\n(S{get_max_step(data,m)})" for m in MUNICIPALITIES]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=step_labels,
                yticklabels=row_labels,
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "F1 (%)"},
                ax=ax, mask=np.isnan(matrix))

    ax.set_title("Per-Entity F1 at Last Completed Step", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_entity_heatmap_last.pdf")
    fig.savefig(OUT_DIR / "per_entity_heatmap_last.png")
    plt.close(fig)
    print("  Saved: per_entity_heatmap_last.pdf / .png")


def plot_per_entity_improvement_heatmap(data, per_entity):
    """Heatmap: F1 delta (last step - step 0) per entity per municipality."""
    matrix = []
    row_labels = []
    for etype in ENTITY_TYPES:
        row = []
        for m in MUNICIPALITIES:
            max_s = get_max_step(data, m)
            f1_0 = per_entity[m].get(0, {}).get(etype, {}).get("f1_score", 0.0) * 100
            f1_L = per_entity[m].get(max_s, {}).get(etype, {}).get("f1_score", 0.0) * 100
            sup_0 = per_entity[m].get(0, {}).get(etype, {}).get("support", 0)
            sup_L = per_entity[m].get(max_s, {}).get(etype, {}).get("support", 0)
            if sup_0 >= 3 or sup_L >= 3:
                row.append(f1_L - f1_0)
            else:
                row.append(np.nan)
        matrix.append(row)
        row_labels.append(etype)

    matrix = np.array(matrix)
    step_labels = [f"{MUNICIPALITY_NAMES[m]}\n(0→{get_max_step(data,m)})" for m in MUNICIPALITIES]

    fig, ax = plt.subplots(figsize=(10, 6))
    max_abs = np.nanmax(np.abs(matrix))
    sns.heatmap(matrix, annot=True, fmt="+.1f", cmap="RdBu",
                xticklabels=step_labels,
                yticklabels=row_labels,
                vmin=-max_abs, vmax=max_abs,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "F1 Change (pp)"},
                ax=ax, mask=np.isnan(matrix),
                center=0)

    ax.set_title("F1 Change: Step 0 → Last Step (percentage points)", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_entity_improvement_heatmap.pdf")
    fig.savefig(OUT_DIR / "per_entity_improvement_heatmap.png")
    plt.close(fig)
    print("  Saved: per_entity_improvement_heatmap.pdf / .png")


def plot_subject_learning_curves(data, per_entity):
    """Focused plot: SUBJECT entity F1 across steps (the hardest entity)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        vals = [per_entity[m][s].get("SUBJECT", {}).get("f1_score", 0.0) * 100 for s in steps]
        ax.plot(steps, vals, marker=MUNICIPALITY_MARKERS[m], linewidth=2.5,
                markersize=8, color=MUNICIPALITY_COLORS[m],
                label=f"{MUNICIPALITY_NAMES[m]}")

    ax.set_xlabel("Number of In-Domain Documents Added")
    ax.set_ylabel("SUBJECT Entity F1 (%)")
    ax.set_title("SUBJECT Entity: Learning Curves\n(Most Challenging Entity Type)")
    ax.set_xticks(range(MAX_STEPS))
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "subject_learning_curves.pdf")
    fig.savefig(OUT_DIR / "subject_learning_curves.png")
    plt.close(fig)
    print("  Saved: subject_learning_curves.pdf / .png")


def plot_entity_grouped_bars(data, per_entity):
    """Grouped bar chart: per-entity F1 at step 0 vs last step, averaged across municipalities."""
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(ENTITY_TYPES))
    width = 0.35

    f1_s0 = []
    f1_sL = []
    for etype in ENTITY_TYPES:
        vals_0 = []
        vals_L = []
        for m in MUNICIPALITIES:
            max_s = get_max_step(data, m)
            e0 = per_entity[m].get(0, {}).get(etype, {})
            eL = per_entity[m].get(max_s, {}).get(etype, {})
            if e0.get("support", 0) >= 3:
                vals_0.append(e0.get("f1_score", 0.0) * 100)
            if eL.get("support", 0) >= 3:
                vals_L.append(eL.get("f1_score", 0.0) * 100)
        f1_s0.append(np.mean(vals_0) if vals_0 else 0)
        f1_sL.append(np.mean(vals_L) if vals_L else 0)

    bars0 = ax.bar(x - width / 2, f1_s0, width, label="Step 0 (LOMO)",
                   color="#90CAF9", edgecolor="#1565C0", linewidth=1)
    barsL = ax.bar(x + width / 2, f1_sL, width, label="Last Step (+ docs)",
                   color="#1565C0", edgecolor="#0D47A1", linewidth=1)

    for bar in bars0:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9, color="gray")
    for bar in barsL:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Entity F1 (%)")
    ax.set_title("Average Per-Entity F1: LOMO Baseline vs. After In-Domain Addition (All 6 Municipalities)")
    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_TYPES, rotation=30, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "entity_grouped_bars.pdf")
    fig.savefig(OUT_DIR / "entity_grouped_bars.png")
    plt.close(fig)
    print("  Saved: entity_grouped_bars.pdf / .png")


def plot_test_set_size(data, metrics):
    """Line chart showing decreasing test set size as docs are added."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        ax.plot(steps, [metrics[m]["n_examples"][s] for s in steps],
                marker=MUNICIPALITY_MARKERS[m], linewidth=2,
                color=MUNICIPALITY_COLORS[m],
                label=f"{MUNICIPALITY_NAMES[m]}")

    ax.set_xlabel("Number of In-Domain Documents Added")
    ax.set_ylabel("Test Set Size (segments)")
    ax.set_title("Test Set Size Reduction with Incremental Addition")
    ax.set_xticks(range(MAX_STEPS))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_set_size.pdf")
    fig.savefig(OUT_DIR / "test_set_size.png")
    plt.close(fig)
    print("  Saved: test_set_size.pdf / .png")


def plot_first_doc_impact(data, metrics):
    """Bar chart: the impact of adding just the FIRST in-domain document (step 0 vs step 1)."""
    # Only include munis that have at least step 1
    munis_with_s1 = [m for m in MUNICIPALITIES if 1 in metrics[m]["f1"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(munis_with_s1))
    width = 0.35

    f1_s0 = [metrics[m]["f1"][0] * 100 for m in munis_with_s1]
    f1_s1 = [metrics[m]["f1"][1] * 100 for m in munis_with_s1]

    ax.bar(x - width / 2, f1_s0, width, label="Step 0 (Zero in-domain)",
           color=[MUNICIPALITY_COLORS[m] for m in munis_with_s1], alpha=0.4,
           edgecolor=[MUNICIPALITY_COLORS[m] for m in munis_with_s1], linewidth=1.5)
    bars1 = ax.bar(x + width / 2, f1_s1, width, label="Step 1 (1 document added)",
                   color=[MUNICIPALITY_COLORS[m] for m in munis_with_s1],
                   edgecolor=[MUNICIPALITY_COLORS[m] for m in munis_with_s1], linewidth=1.5)

    for i, m in enumerate(munis_with_s1):
        delta = f1_s1[i] - f1_s0[i]
        sign = "+" if delta >= 0 else ""
        color = "darkgreen" if delta > 0 else "red"
        ax.annotate(f"{sign}{delta:.1f}pp",
                    xy=(x[i], max(f1_s1[i], f1_s0[i]) + 3),
                    ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_ylabel("Entity F1 (%)")
    ax.set_title("Impact of Adding a Single In-Domain Document")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MUNICIPALITY_NAMES[m]}\n({m})" for m in munis_with_s1])
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "first_doc_impact.pdf")
    fig.savefig(OUT_DIR / "first_doc_impact.png")
    plt.close(fig)
    print("  Saved: first_doc_impact.pdf / .png")


def plot_combined_dashboard(data, metrics, per_entity):
    """4-panel dashboard figure for presentations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A: Learning curves
    ax = axes[0, 0]
    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        f1_vals = [metrics[m]["f1"][s] * 100 for s in steps]
        ax.plot(steps, f1_vals, marker=MUNICIPALITY_MARKERS[m], linewidth=2.5,
                markersize=7, color=MUNICIPALITY_COLORS[m],
                label=f"{MUNICIPALITY_NAMES[m]}")
    ax.set_xlabel("In-Domain Docs Added")
    ax.set_ylabel("Entity F1 (%)")
    ax.set_title("(a) F1 Learning Curves")
    ax.set_xticks(range(MAX_STEPS))
    ax.set_ylim(20, 100)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel B: Improvement bars (step 0 vs last)
    ax = axes[0, 1]
    x = np.arange(len(MUNICIPALITIES))
    width = 0.35
    f1_s0 = [metrics[m]["f1"].get(0, 0) * 100 for m in MUNICIPALITIES]
    f1_sL = [metrics[m]["f1"].get(get_max_step(data, m), 0) * 100 for m in MUNICIPALITIES]
    ax.bar(x - width / 2, f1_s0, width, label="Step 0",
           color=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES], alpha=0.4,
           edgecolor=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES])
    ax.bar(x + width / 2, f1_sL, width, label="Last Step",
           color=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES],
           edgecolor=[MUNICIPALITY_COLORS[m] for m in MUNICIPALITIES])
    for i, m in enumerate(MUNICIPALITIES):
        delta = f1_sL[i] - f1_s0[i]
        sign = "+" if delta >= 0 else ""
        color = "darkgreen" if delta > 0 else "red"
        ax.annotate(f"{sign}{delta:.1f}",
                    xy=(x[i], max(f1_sL[i], f1_s0[i]) + 3),
                    ha="center", fontsize=8, color=color, fontweight="bold")
    ax.set_ylabel("Entity F1 (%)")
    ax.set_title("(b) Step 0 vs Last Step")
    ax.set_xticks(x)
    ax.set_xticklabels([MUNICIPALITY_NAMES[m] for m in MUNICIPALITIES], fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")

    # Panel C: SUBJECT learning curve
    ax = axes[1, 0]
    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        vals = [per_entity[m][s].get("SUBJECT", {}).get("f1_score", 0.0) * 100 for s in steps]
        ax.plot(steps, vals, marker=MUNICIPALITY_MARKERS[m], linewidth=2.5,
                markersize=7, color=MUNICIPALITY_COLORS[m],
                label=f"{MUNICIPALITY_NAMES[m]}")
    ax.set_xlabel("In-Domain Docs Added")
    ax.set_ylabel("SUBJECT F1 (%)")
    ax.set_title("(c) SUBJECT Entity (Hardest)")
    ax.set_xticks(range(MAX_STEPS))
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel D: Per-entity heatmap at last step
    ax = axes[1, 1]
    matrix = []
    row_labels = []
    for etype in ENTITY_TYPES:
        row = []
        for m in MUNICIPALITIES:
            max_s = get_max_step(data, m)
            f1 = per_entity[m].get(max_s, {}).get(etype, {}).get("f1_score", 0.0) * 100
            sup = per_entity[m].get(max_s, {}).get(etype, {}).get("support", 0)
            row.append(f1 if sup >= 3 else np.nan)
        matrix.append(row)
        short = etype.replace("COUNTING-", "C-").replace("VOTER-", "V-")
        row_labels.append(short)
    matrix = np.array(matrix)
    col_labels = [f"{MUNICIPALITY_NAMES[m]}\n(S{get_max_step(data,m)})" for m in MUNICIPALITIES]
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="RdYlGn",
                xticklabels=col_labels,
                yticklabels=row_labels,
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "F1 (%)"},
                ax=ax, mask=np.isnan(matrix))
    ax.set_title("(d) Per-Entity F1 at Last Step")
    ax.tick_params(axis='x', labelsize=8)

    fig.suptitle("Incremental Document Addition Experiments — All 6 Municipalities", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dashboard.pdf")
    fig.savefig(OUT_DIR / "dashboard.png")
    plt.close(fig)
    print("  Saved: dashboard.pdf / .png")


def plot_marginal_gain(data, metrics):
    """Line chart: marginal F1 gain per document added."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        if len(steps) < 2:
            continue
        f1_vals = [metrics[m]["f1"][s] * 100 for s in steps]
        marginal = [f1_vals[i] - f1_vals[i - 1] for i in range(1, len(f1_vals))]
        ax.plot(steps[1:], marginal, marker=MUNICIPALITY_MARKERS[m], linewidth=2,
                markersize=7, color=MUNICIPALITY_COLORS[m],
                label=f"{MUNICIPALITY_NAMES[m]}")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Step (Documents Added)")
    ax.set_ylabel("Marginal F1 Gain (pp)")
    ax.set_title("Marginal F1 Gain per Additional Document")
    ax.set_xticks(range(1, MAX_STEPS))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "marginal_gain.pdf")
    fig.savefig(OUT_DIR / "marginal_gain.png")
    plt.close(fig)
    print("  Saved: marginal_gain.pdf / .png")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading evaluation data...")
    data = load_all_results()
    metrics = extract_overall_metrics(data)
    per_entity = extract_per_entity_metrics(data)

    print("\nData availability:")
    for m in MUNICIPALITIES:
        steps = get_available_steps(data, m)
        print(f"  {MUNICIPALITY_NAMES[m]} ({m}): steps {steps}")

    print("\n=== Generating Tables ===")

    latex_f1 = generate_main_results_table_latex(data, metrics)
    with open(OUT_DIR / "table_incremental_f1.tex", "w") as f:
        f.write(latex_f1)
    print("  Saved: table_incremental_f1.tex")

    latex_pr = generate_precision_recall_table_latex(data, metrics)
    with open(OUT_DIR / "table_precision_recall.tex", "w") as f:
        f.write(latex_pr)
    print("  Saved: table_precision_recall.tex")

    latex_per_entity = generate_per_entity_table_latex(data, per_entity)
    with open(OUT_DIR / "table_per_entity_f1.tex", "w") as f:
        f.write(latex_per_entity)
    print("  Saved: table_per_entity_f1.tex")

    md = generate_markdown_summary(data, metrics, per_entity)
    with open(OUT_DIR / "results_summary.md", "w") as f:
        f.write(md)
    print("  Saved: results_summary.md")

    print("\n=== Generating Graphics ===")
    plot_learning_curves(data, metrics)
    plot_precision_recall_curves(data, metrics)
    plot_improvement_bars(data, metrics)
    plot_per_entity_heatmap(data, per_entity)
    plot_per_entity_improvement_heatmap(data, per_entity)
    plot_subject_learning_curves(data, per_entity)
    plot_entity_grouped_bars(data, per_entity)
    plot_test_set_size(data, metrics)
    plot_first_doc_impact(data, metrics)
    plot_marginal_gain(data, metrics)
    plot_combined_dashboard(data, metrics, per_entity)

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"  - 3 LaTeX tables (.tex)")
    print(f"  - 1 Markdown summary (.md)")
    print(f"  - 11 figures (PDF + PNG pairs)")


if __name__ == "__main__":
    main()
