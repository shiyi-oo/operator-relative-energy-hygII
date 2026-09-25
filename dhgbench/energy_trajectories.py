"""Train/tune/extract Track B energy trajectories with saved provenance.

Run from any directory. See README.md for the reproduction workflow.
Native configurations come from the existing validation-selected CSV winners.
Plain UniGCN and constrained UniGCNII are tuned separately on explicit seeds.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dhgbench'))
from tune_optuna import build_base_args
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.exp_agent import parse_model
from lib_utils.operator_energy import Incidence, OperatorEnergy, array_hash
from lib_utils.utils import fix_seed


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def simple_args(args):
    return {k: v for k, v in vars(args).items()
            if isinstance(v, (str, bool, int, float, list, dict, type(None))) and not k.startswith('_')}


def source_hashes():
    paths = list((ROOT / 'dhgbench/lib_models/HNN').glob('*.py'))
    paths += [Path(__file__), ROOT / 'dhgbench/lib_utils/operator_energy.py',
              ROOT / 'dhgbench/parameter_parser.py', ROOT / 'dhgbench/tune_optuna.py',
              ROOT / 'dhgbench/lib_dataset/preprocessing.py',
              ROOT / 'dhgbench/lib_dataset/data_base.py',
              ROOT / 'dhgbench/lib_models/__init__.py',
              ROOT / 'dhgbench/lib_utils/utils.py', ROOT / 'dhgbench/lib_utils/exp_agent.py']
    paths += list((ROOT / 'dhgbench/lib_yamls/node_yamls').glob('*.yaml'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def native_config(dataset, method, depth):
    path = ROOT / 'logs' / f'optuna_{dataset}' / 'final_table.csv'
    with path.open() as f:
        matches = [r for r in csv.DictReader(f) if r['method'] == method and int(r['depth']) == depth]
    if len(matches) != 1:
        raise ValueError(f'Expected one saved winner for {dataset}/{method}/L{depth}')
    row = matches[0]
    return ast.literal_eval(row['params']), dict(source=str(path), row=row,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def resolved_args(cli, params):
    args = build_base_args(cli.method, cli.dataset, cli.device, cli.epochs, 1)
    for k, v in params.items():
        setattr(args, k, v)
    args.All_num_layers = cli.depth
    args.train_prop, args.valid_prop = 0.5, 0.25
    if cli.cohort in ('plain', 'constrained'):
        args.use_norm = False
        args.activation = 'relu'
        args.input_drop = 0.
    if cli.method == 'UniGCN':
        args.restart_alpha = 0.
    return args


def load_data(args):
    data = data_processing(args, HyperDataset(args))
    data._initialization_()
    return algo_preprocessing(data, args)


def make_optimizer(model, args):
    if args.method in ('UniGCN', 'UniGCNII'):
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)], lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    meta = [dict(lr=g['lr'], weight_decay=g['weight_decay'],
                 parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups]
    return optimizer, meta


def fit(args, data, seed, report_test=True, progress=True):
    fix_seed(seed)
    masks = data.generate_random_split(args.train_prop, args.valid_prop, seed=seed)
    model = parse_model(args, data).to(args.device)
    # Match the original trainer's reset semantics, for the study models.
    model.reset_parameters()
    optimizer, optimizer_meta = make_optimizer(model, args)
    best_val, best_epoch, best_state = -1., 0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = F.nll_loss(F.log_softmax(logits[masks['train']], dim=1), data.y[masks['train']])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_thresh)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(data)
            val = float((logits[masks['valid']].argmax(1) == data.y[masks['valid']]).float().mean())
        if val >= best_val:
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress and (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs):
            print(json.dumps(dict(event='epoch', epoch=epoch, seed=seed, loss=float(loss.detach()),
                                  val=val, best_val=best_val, seconds=time.time() - start)), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        metrics = {name + '_accuracy': float((logits[mask].argmax(1) == data.y[mask]).float().mean())
                   for name, mask in masks.items() if report_test or name != 'test'}
    metrics.update(best_epoch=best_epoch, train_seconds=time.time() - start,
                   optimizer=optimizer_meta, seed=seed)
    return model, masks, metrics


def family(method):
    return {'HGNNII': 'HGNN', 'HNHNII': 'HNHN', 'UniGCNII': 'UniGCN'}.get(method, method)


def measure(model, data, args, run_dir, run_meta):
    index = data.hyperedge_index.detach().cpu().numpy()
    inc = Incidence.largest_component(index, data.num_nodes)
    operator = OperatorEnergy(inc, family(args.method), args.HNHN_alpha,
                              args.HNHN_beta)
    metadata = operator.metadata()
    metadata.update(diagnostic_dtype='float64', forward_dtype=str(data.x.dtype))
    json_write(run_dir / 'operator.json', metadata)
    np.savez_compressed(run_dir / 'operator_vectors.npz', nodes=inc.nodes,
                        pi=operator.pi, psi=operator.psi)
    rows, initial, verification = [], None, {}
    node_index = torch.as_tensor(inc.nodes, device=data.x.device)
    eps = torch.finfo(data.x.dtype).eps
    resolution_op = max(100 * eps ** 2, 100 * operator.residual ** 2,
                        100 * metadata['kernel_control_R_op'], 1e-24)

    def trace(step, role, x, **extra):
        nonlocal initial
        state = x.detach().index_select(0, node_index).cpu().double().numpy()
        values = operator.metrics(state)
        row = dict(dataset=args.dname, method=args.method, cohort=run_meta['cohort'],
                   seed=run_meta['seed'], configured_depth=args.All_num_layers,
                   step=step, role=role, feature_width=state.shape[1],
                   use_norm=bool(getattr(args, 'use_norm', False)), **values)
        if initial is None:
            if role != 'initial':
                raise ValueError('Trace must begin with the diagnostic initial state')
            initial = values
        for key in ('energy_un', 'energy_sym', 'energy_op', 'R_un', 'R_sym', 'R_op',
                    'norm_f2', 'norm_pi2'):
            denom, value = initial[key], values[key]
            row[key + '_relative'] = value / denom if denom and value is not None else None
        row['resolution_R_op'] = resolution_op
        row['censored_R_op'] = values['R_op'] is not None and values['R_op'] <= resolution_op
        row['censored_R_un'] = values['R_un'] is not None and values['R_un'] <= resolution_op * operator.d.max()
        row['censored_R_sym'] = values['R_sym'] is not None and values['R_sym'] <= resolution_op
        rows.append(row)
        if step in (0, 1, 2) and role in ('initial', 'hidden_post_activation'):
            verification[f'{step}_{role}'] = state
        if role == 'hidden_post_activation' and (step % 16 == 0 or step == 1):
            print(json.dumps(dict(event='trace', step=step, R_op=values['R_op'],
                                  delta=values['delta_op2'])), flush=True)

    model.eval()
    with torch.no_grad():
        expected, _ = model(data)
        repeated, _ = model(data)
        actual, _ = model(data, trace=trace)
    # CUDA scatter reductions can vary by a few ulps between two ordinary
    # forwards. Pointwise relative tolerance near zero logits incorrectly rejects
    # those native repeats. Check whole-output error and save the repeat baseline.
    denominator = max(float(expected.norm()), torch.finfo(expected.dtype).tiny)
    repeat_rel = float((repeated - expected).norm()) / denominator
    trace_rel = float((actual - expected).norm()) / denominator
    repeat_max = float((repeated - expected).abs().max())
    trace_max = float((actual - expected).abs().max())
    rel_tolerance = max(1e-6, 10 * repeat_rel)
    abs_tolerance = max(1e-6 + 1e-5 * float(expected.abs().max()), 10 * repeat_max)
    trace_check = dict(repeated_forward_relative_l2=repeat_rel,
                       traced_forward_relative_l2=trace_rel,
                       repeated_forward_maxabs=repeat_max, traced_forward_maxabs=trace_max,
                       relative_l2_tolerance=rel_tolerance, maxabs_tolerance=abs_tolerance,
                       prediction_changes=int((actual.argmax(1) != expected.argmax(1)).sum()))
    json_write(run_dir / 'trace_check.json', trace_check)
    if trace_rel > rel_tolerance or trace_max > abs_tolerance:
        raise AssertionError(f'Tracing exceeds forward numerical tolerance: {trace_check}')
    expected_hidden = args.All_num_layers - 1 if args.method == 'HGNN' else args.All_num_layers
    if sum(r['role'] == 'hidden_post_activation' for r in rows) != expected_hidden:
        raise AssertionError('Unexpected hidden-state count')
    np.savez_compressed(run_dir / 'verification_states.npz', **verification)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temp = run_dir / 'trajectories.csv.tmp'
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(run_dir / 'trajectories.csv')
    return dict(states=len(rows), hidden_states=expected_hidden, component_nodes=len(inc.nodes),
                trace_logits_match=True, trace_check=trace_check)


def train_run(cli):
    run_dir = cli.output / f'{cli.dataset}_{cli.method}_{cli.cohort}_L{cli.depth}_s{cli.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('status') == 'complete':
        print(f'Already complete: {run_dir}', flush=True)
        return
    start = time.time()
    json_write(status, dict(status='running', started_unix=start, command=sys.argv))
    try:
        # Resume interrupted measurement from the already selected checkpoint.
        if cli.mode == 'train' and (run_dir / 'checkpoint.pt').exists():
            cli.mode = 'extract'
        if cli.mode == 'extract':
            checkpoint = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
            params, provenance = checkpoint['params'], checkpoint['provenance']
        elif cli.cohort == 'native':
            params, provenance = native_config(cli.dataset, cli.method, cli.depth)
        else:
            if cli.config is None:
                raise ValueError('Non-native reporting requires --config from validation-only tuning')
            cfg = json.loads(cli.config.read_text())
            if (cfg['dataset'], cfg['method'], cfg['depth']) != (cli.dataset, cli.method, cli.depth):
                raise ValueError('Tuned configuration does not match this reporting cell')
            params, provenance = cfg['params'], dict(source=str(cli.config), tuning=cfg)
        args = resolved_args(cli, params)
        fix_seed(cli.seed)
        data = load_data(args)
        meta = dict(dataset=cli.dataset, method=cli.method, cohort=cli.cohort,
                    seed=cli.seed, depth=cli.depth, epochs=cli.epochs, args=simple_args(args),
                    params=params, provenance=provenance, source_hashes=source_hashes(),
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    feature_hash=array_hash(data.x.detach().cpu().numpy()),
                    incidence_hash=array_hash(data.hyperedge_index.detach().cpu().numpy()),
                    label_hash=array_hash(data.y.detach().cpu().numpy()))
        if cli.mode == 'extract':
            if meta['args'] != checkpoint['manifest']['args']:
                raise ValueError('Resolved configuration differs from saved checkpoint')
            for key in ('feature_hash', 'incidence_hash', 'label_hash'):
                if meta[key] != checkpoint['manifest'][key]:
                    raise ValueError(f'Checkpoint provenance mismatch: {key}')
            original = checkpoint['manifest']['source_hashes']
            changed = {k for k in set(original) | set(meta['source_hashes'])
                       if original.get(k) != meta['source_hashes'].get(k)}
            if changed - {'dhgbench/energy_trajectories.py'}:
                raise ValueError(f'Model/operator source changed since training: {changed}')
            # Training provenance stays immutable. A diagnostic-only runner
            # revision is recorded separately when recovering a saved checkpoint.
            json_write(run_dir / 'extraction_manifest.json', dict(
                source_hashes=meta['source_hashes'], changed_files=sorted(changed),
                training_source_hashes=original, extraction_unix=time.time()))
            model = parse_model(args, data).to(args.device)
            model.load_state_dict(checkpoint['state_dict'])
            metrics = checkpoint['metrics']
        else:
            json_write(run_dir / 'manifest.json', meta)
            model, masks, metrics = fit(args, data, cli.seed)
            checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              params=params, provenance=provenance, manifest=meta, metrics=metrics,
                              masks={k: v.cpu() for k, v in masks.items()})
            temp = run_dir / 'checkpoint.pt.tmp'
            torch.save(checkpoint, temp)
            temp.replace(run_dir / 'checkpoint.pt')
            json_write(run_dir / 'metrics.json', metrics)
            json_write(status, dict(status='measuring', training=metrics))
        measurement = measure(model, data, args, run_dir, meta)
        json_write(status, dict(status='complete', seconds=time.time() - start,
                                metrics=metrics, measurement=measurement))
        print(f'COMPLETE {run_dir}', flush=True)
    except Exception:
        json_write(status, dict(status='failed', seconds=time.time() - start,
                                error=traceback.format_exc(), checkpoint_saved=(run_dir / 'checkpoint.pt').exists()))
        raise


def tune(cli):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    if cli.method not in ('UniGCN', 'UniGCNII') or cli.cohort == 'native':
        raise ValueError('This search is only for plain UniGCN / constrained UniGCNII')
    out = cli.output / 'tuning' / f'{cli.dataset}_{cli.method}_{cli.cohort}_L{cli.depth}'
    out.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(out / 'study.log')))
    study = optuna.create_study(study_name=out.name, storage=storage, direction='maximize',
                               load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=2026))
    seeds = [100, 101, 102]
    json_write(out / 'manifest.json', dict(dataset=cli.dataset, method=cli.method, depth=cli.depth,
                 seeds=seeds, trials=cli.trials, epochs=cli.epochs, source_hashes=source_hashes()))

    def objective(trial):
        params = dict(MLP_hidden=trial.suggest_categorical('MLP_hidden', [64, 128, 256]),
                      dropout=trial.suggest_float('dropout', 0., 0.7, step=0.1))
        if cli.method == 'UniGCNII':
            params['restart_alpha'] = trial.suggest_float('restart_alpha', .05, .5, step=.05)
            params['lamda'] = trial.suggest_categorical('lamda', [.5, 1., 1.5, 2.])
        args = resolved_args(cli, params)
        scores = []
        data = load_data(args)
        try:
            for seed in seeds:
                model, _, metrics = fit(args, data, seed, report_test=False)
                scores.append(metrics['valid_accuracy'])
                del model
            trial.set_user_attr('seed_validation_accuracies', scores)
            return float(np.mean(scores))
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned('CUDA out of memory')
        finally:
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, cli.trials - complete), gc_after_trial=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError('No completed tuning trial')
    best = study.best_trial
    json_write(out / 'best_config.json', dict(dataset=cli.dataset, method=cli.method,
        cohort=cli.cohort, depth=cli.depth, epochs=cli.epochs, params=best.params,
        validation_accuracy=best.value, completed_trials=len(completed), requested_trials=cli.trials,
        tuning_seeds=seeds, best_trial=best.number, source_hashes=source_hashes()))
    print(f'BEST CONFIG {out / "best_config.json"}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['train', 'tune', 'extract'], default='train')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--method', required=True, choices=['HGNN', 'HGNNII', 'HNHN', 'HNHNII', 'UniGCN', 'UniGCNII'])
    ap.add_argument('--cohort', choices=['native', 'plain', 'constrained'], default='native')
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--depth', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--output', type=Path, default=ROOT / 'logs/operator_energy/20260919_track_b')
    cli = ap.parse_args()
    cli.output = cli.output.resolve()
    if cli.config:
        cli.config = cli.config.resolve()
    os.chdir(ROOT / 'dhgbench')
    torch.set_num_threads(cli.threads)
    if cli.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no implicit CPU fallback')
    if cli.depth < 2:
        raise ValueError('Use depth >= 2 for native hidden-state comparisons')
    if (cli.method == 'UniGCN') != (cli.cohort == 'plain'):
        raise ValueError('UniGCN must use the plain cohort, and only UniGCN may do so')
    if cli.cohort == 'constrained' and cli.method != 'UniGCNII':
        raise ValueError('The constrained cohort is UniGCNII')
    if cli.mode == 'tune':
        tune(cli)
    else:
        train_run(cli)


if __name__ == '__main__':
    main()
