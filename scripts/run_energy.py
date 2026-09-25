"""Run controlled linear propagation on the frozen study datasets."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dhgbench'))
from experiments.linear_propagation import DATASETS, FAMILIES, prepare, worker, export_plotdata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--features', nargs='+', choices=['real', 'random'], default=['real', 'random'])
    parser.add_argument('--out', type=Path, default=ROOT / 'results/energy')
    parser.add_argument('--overwrite', action='store_true', help='Recompute completed trajectories in the output directory')
    args = parser.parse_args()
    for dataset in args.datasets:
        run = args.out.resolve() / dataset
        prepare(run, dataset=dataset)
        for kind in args.features:
            for family in FAMILIES:
                worker(run, family, kind, overwrite=args.overwrite)
            export_plotdata(run, dataset, kind)


if __name__ == '__main__':
    main()
