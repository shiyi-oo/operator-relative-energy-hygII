"""Controlled linear propagation from the frozen study inputs."""
import csv
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from lib_utils.operator_energy import Incidence, OperatorEnergy, array_hash

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('HGNN', 'HNHN', 'UniGCN')
KINDS = ('real', 'random')
DATASETS = {'cora': 'Cora', 'citeseer': 'Citeseer', '20newsW100': '20News100',
            'house-committees-100': 'House-Committee', 'pokec': 'Pokec', 'NTU2012': 'NTU2012'}
FIELDS = ('energy_un', 'energy_sym', 'energy_op', 'norm_f2', 'norm_pi2',
          'R_un', 'R_sym', 'R_op')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def trajectory(op, x0, steps, alpha, progress=False):
    """Return both recurrences and verify the PSD linear energy bounds."""
    initial = op.metrics(x0)
    if any(initial[k] is None or initial[k] <= 0 for k in FIELDS):
        raise ValueError('Initial state must have nonzero energy and norm for all ratios')
    if op.lambda1 > 1 + 1e-10 or op.lambda2 is None or op.lambda2 < 0:
        raise ValueError('The restart floor requires a PSD operator with spectrum in [0, 1]')
    resolution = max(100 * np.finfo(np.float64).eps ** 2,
                     100 * op.residual ** 2, 100 * op.metadata()['kernel_control_R_op'], 1e-24)
    base, restart = x0.copy(), x0.copy()
    rows = []
    for step in range(steps + 1):
        for variant, state in (('base', base), ('restart', restart)):
            values = op.metrics(state)
            row = dict(step=step, variant=variant, **{k: values[k] for k in FIELDS})
            for key in FIELDS:
                row[key + '_relative'] = values[key] / initial[key] if values[key] is not None else None
            row.update(zero_state=values['zero_state'], resolution_R_op=resolution,
                       censored_R_op=values['R_op'] is not None and values['R_op'] <= resolution)
            ratio = row['energy_op_relative']
            if variant == 'base':
                if ratio > op.lambda2 ** (2 * step) + 1e-10:
                    raise AssertionError('Linear base energy exceeds its spectral bound')
            elif step and not alpha ** 2 - 1e-10 <= ratio <= 1 + 1e-10:
                raise AssertionError('Linear restart energy violates the floor/upper bound')
            if op.family == 'HGNN' and not np.isclose(values['energy_op'], values['energy_sym'],
                                                     rtol=1e-10, atol=1e-12 * initial['energy_op']):
                raise AssertionError('HGNN operator energy disagrees with normalized-Laplacian energy')
            rows.append(row)
        if progress and step % 32 == 0:
            print(json.dumps(dict(step=step, base_e=rows[-2]['energy_op_relative'],
                                  restart_e=rows[-1]['energy_op_relative'])), flush=True)
        if step != steps:
            base = op.apply(base)
            restart = (1 - alpha) * op.apply(restart) + alpha * x0
    return rows


def worker(run, family, kind, overwrite=False):
    cfg = json.loads((run / 'suite.json').read_text())
    target = run / f'{family}_{kind}'
    target.mkdir(exist_ok=True)
    status = target / 'status.json'
    if not overwrite and status.exists() and json.loads(status.read_text()).get('status') == 'complete':
        print(f'Already complete: {target}')
        return
    start = time.time()
    atomic_json(status, dict(status='running', started_unix=start, job_id=os.getenv('SLURM_JOB_ID')))
    try:
        with np.load(run / cfg.get('input_file', 'cora_input.npz')) as saved:
            features, index, nodes = saved['features'], saved['index'], saved['nodes']
        inc = Incidence.largest_component(index, len(features))
        if inc.source_hash != cfg['incidence_hash'] or not np.array_equal(nodes, inc.nodes):
            raise ValueError('Frozen incidence/component mismatch')
        if array_hash(features) != cfg['full_feature_hash']:
            raise ValueError('Frozen feature hash mismatch')
        op = OperatorEnergy(inc, family, cfg['hnhn_alpha'], cfg['hnhn_beta'], chunk_size=2048)
        atomic_json(target / 'operator.json', op.metadata())
        np.savez_compressed(target / 'operator_vectors.npz', nodes=nodes, pi=op.pi, psi=op.psi)
        seeds = [None] if kind == 'real' else cfg['random_seeds']
        rows, inputs = [], []
        for seed in seeds:
            if kind == 'real':
                x0 = features[nodes].copy()
            else:
                x0 = np.random.default_rng(seed).normal(size=(len(nodes), cfg['random_width']))
                x0 *= np.sqrt(x0.size / np.sum(x0 * x0))
            inputs.append(dict(seed=seed, feature_hash=array_hash(x0), width=x0.shape[1]))
            print(json.dumps(dict(event='input', family=family, kind=kind, seed=seed)), flush=True)
            measured = trajectory(op, x0, cfg['steps'], cfg['alpha'], progress=True)
            rows.extend(dict(family=family, features=kind, seed=seed, **row) for row in measured)
        write_csv(target / 'trajectories.csv', rows)
        atomic_json(target / 'inputs.json', inputs)
        atomic_json(status, dict(status='complete', seconds=time.time() - start,
                                 job_id=os.getenv('SLURM_JOB_ID'), inputs=len(seeds),
                                 trajectories=2 * len(seeds), rows=len(rows), checks='passed'))
    except Exception:
        atomic_json(status, dict(status='failed', seconds=time.time() - start,
                                 job_id=os.getenv('SLURM_JOB_ID'), error=traceback.format_exc()))
        raise


def summarize(rows, field, variant, steps, expected):
    means, stds = [], []
    for step in range(steps + 1):
        values = [r[field] for r in rows if r['variant'] == variant and int(r['step']) == step]
        if len(values) != expected:
            raise ValueError('A plotted point must contain every expected feature seed')
        vals = np.array([float(v) if v not in ('', None) else np.nan for v in values])
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)) if expected > 1 else 0.)
    return np.array(means), np.array(stds)



def prepare(run, steps=256, dataset='cora'):
    """Install frozen inputs and reject incompatible resumed runs."""
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    target = run / 'suite.json'
    if target.exists():
        cfg = json.loads(target.read_text())
        if cfg['steps'] != steps or cfg['dataset'] != dataset:
            raise ValueError('Use a new output directory to change the dataset or propagation depth')
        input_path = run / cfg.get('input_file', f'{dataset}_input.npz')
        if not input_path.exists():
            shutil.copy2(ROOT / 'data/energy_inputs' / f'{dataset}.npz', input_path)
        return cfg
    cfg = json.loads((ROOT / 'results/energy' / dataset / 'suite.json').read_text())
    cfg['steps'] = steps
    shutil.copy2(ROOT / 'data/energy_inputs' / f'{dataset}.npz', run / cfg['input_file'])
    atomic_json(target, cfg)
    return cfg
def export_plotdata(run, dataset, kind):
    """Produce the exact ratio summaries consumed by the combined figure script."""
    import numpy as np
    records = []
    mapping = [('un', 'base', 'R_un_relative'), ('sym', 'base', 'R_sym_relative'),
               ('op', 'base', 'R_op_relative'), ('base', 'base', 'energy_op_relative'),
               ('restart', 'restart', 'energy_op_relative')]
    for family in FAMILIES:
        with (run / f'{family}_{kind}/trajectories.csv').open() as handle:
            raw = list(csv.DictReader(handle))
        for series, variant, field in mapping:
            for step in sorted({int(row['step']) for row in raw}):
                values = np.array([float(row[field]) for row in raw
                                   if row['variant'] == variant and int(row['step']) == step])
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else 0.
                records.append(dict(family=family, section='diagnostics' if series in ('un', 'sym', 'op') else 'restart', series=series, step=step, mean=mean,
                                    sd=sd, lower=mean-sd, upper=mean+sd, seeds=len(values)))
    folder = run / 'figures'
    folder.mkdir(exist_ok=True)
    with (folder / f'track_a_{dataset}_{kind}_plotdata.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

