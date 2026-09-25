"""Generate the main matched-HNHN depth-study figure from completed results.

The figure facets by dataset. Within each panel, colour denotes the method
family and line style denotes baseline vs II variant.

Usage:
  python scripts/plot_classification.py \
    --csv results/classification/final_table.csv \
    --out outputs/figures/depth_trajectory.png
"""

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dhgbench"))
from experiments.classification_results import collect, write_table
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FAMILIES = [
    ("HGNN", "HGNN", "HGNNII"),
    ("HNHN", "HNHN", "HNHNII"),
    ("HyperGCN", "HyperGCN", "HyperGCNII"),
    ("AllSetTransformer", "AllSetformer", "AllSetformerII"),
    ("UniGCN", "UniGCN", "UniGCNII"),
]

COLORS = {
    "HGNN": "#E69F00",
    "HNHN": "#0072B2",
    "HyperGCN": "#009E73",
    "AllSetTransformer": "#D55E00",
    "UniGCN": "#CC79A7",
}

DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "20newsW100": "20News100",
    "house-committees-100": "House-Committee",
    "pokec": "Pokec",
    "NTU2012": "NTU2012",
}

DATASET_ORDER = [
    "cora",
    "citeseer",
    "20newsW100",
    "house-committees-100",
    "pokec",
    "NTU2012",
]

INK = "#222222"
INK_2 = "#555555"
GRID = "#e7e7e7"


def load_rows(path, families):
    rows = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") == "dataset":
                continue
            if row["method"] not in {m for _, base, ii in families for m in (base, ii)}:
                continue
            if row['method'] in ('HNHN', 'HNHNII') and row.get('architecture') != 'MatchedHNHN':
                raise ValueError('HNHN rows must come from the matched main study; regenerate the classification results')
            dataset = row["dataset"]
            method = row["method"]
            depth = int(row["depth"])
            rows[dataset][method][depth] = float(row["test_acc"])
    return rows


def ordered_datasets(rows, requested):
    if requested:
        missing = [d for d in requested if d not in rows]
        if missing:
            raise SystemExit(f"datasets not found in CSV: {', '.join(missing)}")
        return requested

    known = [d for d in DATASET_ORDER if d in rows]
    extra = sorted(d for d in rows if d not in DATASET_ORDER)
    return known + extra


def panel_shape(n_panels):
    if n_panels <= 3:
        return 1, n_panels
    return (2, 2) if n_panels == 4 else (2, 3)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--csv', type=Path, help='Plot an existing table')
    source.add_argument('--root', type=Path, help='Collect completed cells from this run directory')
    parser.add_argument('--kind', choices=['selected', 'evaluated'], default='evaluated')
    parser.add_argument("--out", default=str(ROOT / "outputs/figures/depth_trajectory.png"))
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    if args.csv is None:
        records = collect(args.root or ROOT / 'results/classification', args.kind)
        args.csv = Path(args.out).with_suffix('.csv')
        write_table(args.csv, records)
    rows = load_rows(args.csv, FAMILIES)
    datasets = ordered_datasets(rows, args.datasets)
    if not datasets:
        raise SystemExit("no data rows found")

    nrows, ncols = panel_shape(len(datasets))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.5, 8.0 if nrows == 2 else 4.2),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    letters = "abcdefghijklmnopqrstuvwxyz"

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        dataset_rows = rows[dataset]
        depths = sorted(
            {
                depth
                for _, base, ii in FAMILIES
                for method in (base, ii)
                for depth in dataset_rows.get(method, {})
            }
        )
        for family, base_method, ii_method in FAMILIES:
            color = COLORS[family]
            for method, linestyle in ((base_method, "--"), (ii_method, "-")):
                points = sorted(dataset_rows.get(method, {}).items())
                if not points:
                    continue
                xs = [depth for depth, _ in points]
                ys = [acc for _, acc in points]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    linewidth=2.0,
                    markersize=4.8,
                    zorder=3,
                )

        ax.set_xscale("log", base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(depth) for depth in depths], fontsize=14)
        ax.set_title(
            f"({letters[idx]}) {DATASET_LABELS.get(dataset, dataset)}",
            loc="left",
            fontsize=16,
            fontweight="bold",
            color=INK,
            pad=8,
        )
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9a9a9a")
        ax.tick_params(colors=INK_2, labelsize=14)

    for ax in axes_flat[len(datasets) :]:
        ax.axis("off")

    fig.supylabel("Test accuracy (%)", fontsize=16, fontweight="bold", color=INK, x=0.035)
    fig.supxlabel("Network depth (L)", fontsize=16, fontweight="bold", color=INK, y=0.11)

    method_label = Line2D([], [], color="none", label="Method family")
    method_handles = [
        Line2D([], [], color=COLORS[family], marker="o", linewidth=2.6, label=family)
        for family, _, _ in FAMILIES
    ]
    variant_label = Line2D([], [], color="none", label="Variant")
    variant_handles = [
        Line2D([], [], color="black", linestyle="--", linewidth=2.2, label="Baseline"),
        Line2D([], [], color="black", linestyle="-", linewidth=2.2, label="II variant"),
    ]
    handles = [method_label, *method_handles, variant_label, *variant_handles]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(handles),
        frameon=False,
        fontsize=14,
        handlelength=2.3,
        columnspacing=1.25,
    )
    for text in legend.get_texts():
        if text.get_text() in {"Method family", "Variant"}:
            text.set_fontweight("bold")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=(0.035, 0.125, 1.0, 0.98), w_pad=2.8, h_pad=2.6)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out} and {pdf_out}")


if __name__ == "__main__":
    main()
