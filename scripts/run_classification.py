"""Run the main depth study: tune and evaluate baseline/II models, using matched HNHN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dhgbench'))
DATASETS = ['cora', 'citeseer', '20newsW100', 'house-committees-100', 'pokec', 'NTU2012']
METHODS = ['HGNN', 'HGNNII', 'HNHN', 'HNHNII', 'HyperGCN', 'HyperGCNII',
           'AllSetformer', 'AllSetformerII', 'UniGCN', 'UniGCNII']
DEPTHS = [2, 4, 8, 16, 32, 64]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='run', choices=['run', 'selected', 'tune', 'evaluate'])
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=None)
    p.add_argument('--depths', nargs='+', type=int, choices=DEPTHS, default=DEPTHS)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seeds', type=int, default=None, help='Default: 3 tuning, 10 evaluation')
    p.add_argument('--trials', type=int, default=30, help='Target number of successfully completed trials per cell')
    p.add_argument('--sampler-seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, default=ROOT / 'results/classification')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.methods is None:
        a.methods = [m for m in METHODS if m not in ('HNHN', 'HNHNII', 'UniGCN')] if a.action == 'selected' else METHODS
    seed_override = a.seeds
    a.seeds = a.seeds if a.seeds is not None else (3 if a.action == 'tune' else 10)
    if min(a.epochs, a.seeds, a.trials, a.threads) < 1:
        p.error('epochs, seeds, trials and threads must be positive')
    out = a.out.resolve()
    configs = json.loads((ROOT / 'configs/selected.json').read_text())
    if a.action == 'selected':
        if any(m in ('HNHN', 'HNHNII', 'UniGCN') for m in a.methods):
            p.error('Saved HNHN/HNHNII and UniGCN settings need architecture/normalization compatibility checks before replay; use run or tune/evaluate with the current models')
        indexed = {(c['dataset'], c['method'], c['depth']): c for c in configs}
        requested = [(d, m, depth) for d in a.datasets for m in a.methods for depth in a.depths]
        missing = [key for key in requested if key not in indexed]
        if missing:
            p.error(f'No saved configuration for {missing[0]}; use tune then evaluate.')
        selected = [indexed[key] for key in requested]
    else:
        selected = [dict(dataset=d, method=m, depth=depth)
                    for d in a.datasets for m in a.methods for depth in a.depths]
    seed_label = '3 tuning / 10 evaluation' if a.action == 'run' and seed_override is None else str(a.seeds)
    print(f'{a.action}: {len(selected)} cells; {a.epochs} epochs; {seed_label} seeds; {a.device}', flush=True)
    if a.dry_run:
        for c in selected:
            print(c['dataset'], c['method'], c['depth'])
        return
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['MKL_NUM_THREADS'] = str(a.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(a.threads)
    from lib_dataset.prepare_data import prepare_data
    prepare_data()
    out.mkdir(parents=True, exist_ok=True)
    if a.action != 'selected':
        from experiments.classification_results import collect, write_table
        for c in selected:
            study = out / 'studies/main-depth-matched-hnhn-v1' / c['dataset']
            study.mkdir(parents=True, exist_ok=True)
            actions = ['tune', 'evaluate'] if a.action == 'run' else [a.action]
            for action in actions:
                seeds = seed_override if seed_override is not None else (3 if action == 'tune' else 10)
                common = ['--dname', c['dataset'], '--epochs', str(a.epochs), '--num_seeds', str(seeds),
                          '--device', a.device, '--study_dir', str(study)]
                if action == 'tune':
                    command = [sys.executable, 'tune_optuna.py', *common, '--method', c['method'],
                               '--All_num_layers', str(c['depth']), '--n_trials', str(a.trials),
                               '--seed', str(a.sampler_seed)]
                else:
                    folder = out / 'evaluated' / c['dataset']
                    folder.mkdir(parents=True, exist_ok=True)
                    journal = study / f"{c['dataset']}_{c['method']}_L{c['depth']}.log"
                    if not journal.exists():
                        raise FileNotFoundError(f'Tune this cell first: {journal}')
                    command = [sys.executable, 'eval_best.py', *common, '--only_method', c['method'],
                               '--only_depth', str(c['depth']), '--out', str(folder / f"{c['method']}_L{c['depth']}.csv")]
                subprocess.run(command, cwd=ROOT / 'dhgbench', check=True)
                if action == 'evaluate':
                    write_table(out / 'final_table.csv', collect(out, 'evaluated'))
        if a.action != 'tune':
            print(f"Figure-ready table: {out / 'final_table.csv'}")
        return

    sys.path.insert(0, str(ROOT / 'dhgbench'))
    import torch
    from tune_optuna import build_base_args, run_config
    from lib_dataset.data_base import HyperDataset
    from lib_dataset.preprocessing import data_processing
    torch.set_num_threads(a.threads)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
    os.chdir(ROOT / 'dhgbench')
    rows = []
    for config in selected:
        dataset, method, depth = config['dataset'], config['method'], config['depth']
        folder = out / dataset
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{method}_L{depth}.json'
        source_hash = hashlib.sha256(b''.join(q.read_bytes() for q in sorted((ROOT/'dhgbench').rglob('*.py')))).hexdigest()
        identity = dict(config=config, epochs=a.epochs, seeds=a.seeds, device=a.device,
                        threads=a.threads, source_sha256=source_hash)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] != identity:
                raise ValueError(f'Different configuration exists: {path}. Choose a new --out.')
            row = saved['summary']
            print(f'Reusing {path}', flush=True)
        else:
            args = build_base_args(method, dataset, a.device, a.epochs, a.seeds)
            data = data_processing(args, HyperDataset(args))
            data._initialization_()
            started = time.time()
            val, acc, std, seeds = run_config(args, data, config['params'], depth, return_per_seed=True)
            row = dict(dataset=dataset, method=method, depth=depth,
                       protocol='main-depth-matched-hnhn-v1', architecture=method, test_acc=acc,
                       test_std=std, val_acc=val, num_seeds=a.seeds, epochs=a.epochs,
                       params=str(config['params']))
            resolved = vars(args).copy()
            resolved.update(config['params'], All_num_layers=depth)
            saved = dict(identity=identity, summary=row, per_seed=seeds,
                         per_seed_columns=['train', 'validation', 'test'], seed_ids=list(range(a.seeds)),
                         resolved_args=resolved, seconds=time.time()-started,
                         torch=torch.__version__, cuda=torch.version.cuda)
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(saved, indent=2, default=str)+'\n')
            temp.replace(path)
            print(f'{dataset} {method} L={depth}: {acc:.4f} +/- {std:.4f}', flush=True)
        rows.append(row)
    from experiments.classification_results import collect, write_table
    write_table(out / 'selected_table.csv', collect(out, 'selected'))
    # Unique subset filename prevents parallel workers overwriting one another.
    key = hashlib.sha256(json.dumps([(c['dataset'], c['method'], c['depth']) for c in selected]).encode()).hexdigest()[:12]
    write_table(out / f'summary_{key}.csv', rows)


if __name__ == '__main__':
    main()
