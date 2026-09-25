"""Plot saved energy curves for all six datasets into energy_trajectory.pdf."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [('cora', 'Cora'), ('citeseer', 'Citeseer'), ('20newsW100', '20News100'),
            ('house-committees-100', 'House-Committee'), ('pokec', 'Pokec'), ('NTU2012', 'NTU2012')]
FIGURE_HEIGHT = 2.1 * len(DATASETS) + 2.10
FAMILIES = ['HGNN', 'HNHN', 'UniGCN']
SKY, ORANGE, GREEN, RED, GREY = '#56B4E9', '#E69F00', '#009E73', '#D55E00', '#666666'
PURPLE = '#CC79A7'
INK, TICK, GRID, SPINE = '#222222', '#444444', '#e1e1e1', '#999999'
TITLE_PT, LABEL_PT, TICK_PT = [pt * 18 / 15.01 for pt in (18, 16, 14)]
BLOCKS = [
    ('diagnostics', '(a) Measure varies, propagation fixed',
     r'$R(X^{(\ell)})\,/\,R(X^{(0)})$', 1.05, 7.6,
     [('op', r'Operator-relative $R_P$', GREEN, 'solid'),
      ('un', r'Classical $R_{\mathrm{un}}$', SKY, (0, (5, 2))),
      ('sym', r'Classical $R_{\mathrm{n}}$', ORANGE, (0, (5, 2)))]),
    ('restart', '(b) Propagation varies, measure fixed',
     r'$\mathcal{E}_P(X^{(\ell)})\,/\,\mathcal{E}_P(X^{(0)})$', 8.88, 15.43,
     [('restart', r'II', RED, 'solid'),
      ('base', 'Baseline', PURPLE, (0, (5, 2)))]),
]


def load_data(energy_root, features):
    frames = []
    for name, _ in DATASETS:
        path = energy_root / name / 'figures' / f'track_a_{name}_{features}_plotdata.csv'
        frames.append(pd.read_csv(path).assign(dataset=name))
    data = pd.concat(frames, ignore_index=True)
    values = data[['mean', 'lower', 'upper']].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Log-scale curves require finite, positive values.')
    for name, _ in DATASETS:
        for section, _, _, _, _, series in BLOCKS:
            for family in FAMILIES:
                for key, *_ in series:
                    curve = data[(data.dataset == name) & (data.section == section)
                                 & (data.family == family) & (data.series == key)]
                    if sorted(curve.step) != list(range(257)):
                        raise ValueError(f'Incomplete curve: {name}/{family}/{key}')
    return data


def draw_block(fig, data, spec):
    section, title, ylabel, left, right, series = spec
    left, right = left / 16, right / 16
    top, bottom = 1 - .75 / FIGURE_HEIGHT, 1.35 / FIGURE_HEIGHT
    grid = fig.add_gridspec(len(DATASETS), 3, left=left, right=right, top=top,
                            bottom=bottom, hspace=.22, wspace=.22)
    for row, (name, label) in enumerate(DATASETS):
        subset = data[(data.dataset == name) & (data.section == section)]
        exponents = range(-5, 2)
        for col, family in enumerate(FAMILIES):
            ax = fig.add_subplot(grid[row, col])
            if section == 'restart':
                ax.axhline(.01, color=GREY, linestyle=':', linewidth=2.2, zorder=2)
            for index, (key, _, color, style) in enumerate(series):
                curve = subset[(subset.family == family) & (subset.series == key)].sort_values('step')
                ax.plot(curve.step, curve['mean'], color=color,
                        linestyle='--' if key == 'base' else '-',
                        linewidth=2.2, zorder=3, solid_capstyle='round')
                markers = curve.iloc[4 + index * 10::32]
                ax.plot(markers.step, markers['mean'], linestyle='none', color=color,
                        marker=['o', 's', '^'][index], markersize=5, markerfacecolor=color,
                        markeredgewidth=1.3, zorder=4)
            ax.set_yscale('log')
            ax.set_ylim(1e-5, 1e1)
            ax.set_yticks([10. ** e for e in exponents],
                          labels=[rf'$10^{{{e}}}$' for e in exponents])
            ax.set_xlim(0, 256)
            ax.set_xticks([0, 64, 128, 192, 256])
            if row < len(DATASETS) - 1:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(family, fontsize=LABEL_PT, fontweight='bold', color=INK, pad=5)
            if col > 0:
                ax.set_yticklabels([])
            ax.grid(True, which='major', color=GRID, linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color(SPINE)
            ax.tick_params(labelsize=TICK_PT, colors=TICK)
            if col == 2 and section == 'restart':
                pos = ax.get_position()
                fig.text(15.73 / 16, pos.y0 + pos.height / 2, label, fontsize=TICK_PT,
                         fontweight='bold', color=INK, ha='center', va='center', rotation=270)
    centre = (left + right) / 2
    fig.text(left, 1 - .30 / FIGURE_HEIGHT, title, fontsize=TITLE_PT, fontweight='bold', color=INK)
    fig.text(centre, .65 / FIGURE_HEIGHT, r'Propagation step $\ell$', fontsize=LABEL_PT, color=INK, ha='center')
    label_x = .24 / 16 if section == 'diagnostics' else left - .80 / 16
    fig.text(label_x, (top + bottom) / 2, ylabel, fontsize=LABEL_PT,
             color=INK, ha='center', va='center', rotation=90)
    handles = [Line2D([], [], color=color, linewidth=2.2, label=label,
                      linestyle='--' if key == 'base' else '-',
                      marker=['o', 's', '^'][index],
                      markersize=5, markerfacecolor=color, markeredgewidth=1.3)
               for index, (key, label, color, style) in enumerate(series)]
    if section == 'restart':
        handles.append(Line2D([], [], color=GREY, linestyle=':', linewidth=2.2, label='Lower Bound'))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(centre, .3 / FIGURE_HEIGHT),
               ncol=len(handles), frameon=False, fontsize=TICK_PT, handlelength=2.5, columnspacing=1.8)


def main():
    global DATASETS, FIGURE_HEIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy-root', type=Path, default=ROOT / 'results/energy')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/figures', help='Output directory')
    parser.add_argument('--datasets', nargs='+', choices=[name for name, _ in DATASETS])
    parser.add_argument('--features', choices=['real', 'random'], default='real')
    args = parser.parse_args()
    if args.datasets:
        labels = dict(DATASETS)
        DATASETS = [(name, labels[name]) for name in args.datasets]
    FIGURE_HEIGHT = 2.1 * len(DATASETS) + 2.10
    data = load_data(args.energy_root, args.features)
    fig = plt.figure(figsize=(18, FIGURE_HEIGHT))
    for block in BLOCKS:
        draw_block(fig, data, block)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ('energy_trajectory.pdf' if args.features == 'real' else 'energy_trajectory_random.pdf')
    with matplotlib.rc_context({'pdf.fonttype': 42}):
        fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(f"wrote {path} and {path.with_suffix('.png')}")


if __name__ == '__main__':
    main()
