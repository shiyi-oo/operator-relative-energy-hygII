"""
tune_optuna.py -- distributed Optuna hyperparameter search for the hypergraph models.

One process = one Optuna worker. Many workers share a study through a JournalStorage
file (safe on a shared/NFS filesystem, unlike SQLite), so a SLURM array job of N tasks
performs one N-way-parallel search.

The dataset is loaded ONCE per worker; each trial deep-copies it from a pristine base
and re-runs the method-specific preprocessing, so trials cannot leak state into each
other (this matters because e.g. HNHN_alpha/HNHN_beta change the preprocessing, and
hgnn2 caches a _DB attribute on the data object).

Objective = mean validation accuracy over `--num_seeds` seeds. The test split is never
used for model selection; the final numbers are produced afterwards by eval_best.py.

Example
-------
  python tune_optuna.py --method HGNNII --dname cora --All_num_layers 8 \
      --n_trials 30 --num_seeds 3 --study_dir ../logs/optuna_cora
"""
import argparse
import copy
import hashlib
from pathlib import Path
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from lib_models import _semi_methods_
from experiments.search import NumericalTrainingError, complete_search
from parameter_parser import parameter_parser, method_config, set_task_args
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_utils.exp_agent import ExpAgent, parse_model
from lib_utils.utils import fix_seed, mean_std_metrics


def source_hash():
    root = Path(__file__).resolve().parent
    paths = sorted(p for p in root.rglob('*') if p.suffix in ('.py', '.yaml'))
    return hashlib.sha256(b''.join(str(p.relative_to(root)).encode() + p.read_bytes() for p in paths)).hexdigest()


# --------------------------------------------------------------------------- spaces
# Shared knobs, then per-method additions. II variants additionally get the two
# knobs of Definition 1: the restart weight alpha and the identity-map schedule lamda.
II_METHODS = {"HGNNII", "HNHNII", "HyperGCNII", "AllSetformerII", "UniGCNII"}


def suggest(trial, method, depth):
    p = {}
    p["lr"] = trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    p["wd"] = trial.suggest_categorical("wd", [0.0, 1e-5, 1e-4, 1e-3])
    p["dropout"] = trial.suggest_float("dropout", 0.0, 0.7, step=0.1)
    # deep models need a smaller hidden width to stay trainable / fit in memory
    widths = [64, 128, 256, 512] if depth <= 16 else [64, 128, 256]
    p["MLP_hidden"] = trial.suggest_categorical("MLP_hidden", widths)

    if method in II_METHODS:
        # alpha: how much of X^(0) is restarted at every layer
        p["restart_alpha"] = trial.suggest_float("restart_alpha", 0.05, 0.5, step=0.05)
        # lamda: beta_l = log(lamda/l + 1); larger => weights stay further from identity
        p["lamda"] = trial.suggest_categorical("lamda", [0.5, 1.0, 1.5, 2.0])

    if method in ("HNHN", "HNHNII"):
        p["HNHN_alpha"] = trial.suggest_categorical("HNHN_alpha", [-1.5, -1.0, -0.5, 0.0])
        p["HNHN_beta"] = trial.suggest_categorical("HNHN_beta", [-1.0, -0.5, 0.0])

    if method in ("HyperGCN", "HyperGCNII"):
        p["HyperGCN_mediators"] = True
        p["HyperGCN_fast"] = True
        if method == "HyperGCNII":
            # the manuscript prescribes the lazy operator (spec(P) in [0,1]); tune it so
            # the cost/benefit at each depth is measured rather than assumed
            p["HyperGCN_lazy"] = trial.suggest_categorical("HyperGCN_lazy", [True, False])

    if method in ("AllSetformer", "AllSetformerII"):
        p["heads"] = trial.suggest_categorical("heads", [1, 4, 8])
        p["MLP_num_layers"] = trial.suggest_categorical("MLP_num_layers", [1, 2])
        p["decoder_num_layers"] = trial.suggest_categorical("decoder_num_layers", [1, 2])
        p["normalization"] = trial.suggest_categorical("normalization", ["ln", "bn"])

    if method in ("UniGCN", "UniGCNII"):
        p["input_drop"] = trial.suggest_float("input_drop", 0.0, 0.6, step=0.2)
        p["activation"] = trial.suggest_categorical("activation", ["relu", "prelu"])
        p["use_norm"] = trial.suggest_categorical("use_norm", [False] if method == "UniGCN" else [True, False])
    return p


# --------------------------------------------------------------------------- runner
def build_base_args(method, dname, device, epochs, num_seeds):
    """Resolve defaults + YAML for this method, without consuming the real sys.argv."""
    saved = sys.argv
    sys.argv = ["tune_optuna.py", f"--dname={dname}", f"--method={method}", "--use_yaml"]
    try:
        args = method_config(parameter_parser(), protected=set())
    finally:
        sys.argv = saved
    args = set_task_args(args)
    args.device = device
    args.epochs = epochs
    args.num_seeds = num_seeds
    args.eval_verbose = False
    args.display_step = 10 ** 9
    return args


def run_config(base_args, base_data, overrides, depth, metric="acc", return_per_seed=False):
    """Train+evaluate one configuration.

    Mirrors ExpAgent.node_cls_train_eval, but returns the VALIDATION score as well.
    ExpAgent keeps only the test column (test_dict[m] = [test_mean, test_std]) and the
    evaluator returns {metric: [train, val, test]}, so selecting on validation -- which
    is what a tuner must do -- needs this loop rather than agent.test_dict.

    Returns (val_mean, test_mean, test_std).
    """
    args = copy.deepcopy(base_args)
    args.All_num_layers = depth
    for k, v in overrides.items():
        setattr(args, k, v)

    # fresh data every trial: preprocessing depends on args (e.g. HNHN_alpha/beta) and
    # some models cache tensors on the data object (hgnn2 caches data._DB)
    data = copy.deepcopy(base_data)
    data = algo_preprocessing(data, args)

    agent = ExpAgent(args)
    per_seed = []
    for seed in range(args.num_seeds):
        fix_seed(seed)
        masks = data.generate_random_split(train_ratio=args.train_prop,
                                           val_ratio=args.valid_prop, seed=seed)
        model = parse_model(args, data).to(args.device)
        model = agent.trainer.training(model, data, args, seed_split=masks,
                                       task_type='node_cls')
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise NumericalTrainingError('Nonfinite trained parameters')
        result = agent.evaluator.evaluate(model, data, seed_split=masks,
                                          task_type='node_cls', verbose=False)
        key = metric if metric in result else list(result)[0]
        if not np.isfinite(result[key]).all():
            raise NumericalTrainingError('Nonfinite evaluation scores')
        per_seed.append(result[key])          # [train, val, test]
        del model
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()

    mean, std = mean_std_metrics(per_seed)
    summary = (float(mean[1]), float(mean[2]), float(std[2]))
    return (*summary, np.asarray(per_seed).tolist()) if return_per_seed else summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=_semi_methods_, required=True)
    ap.add_argument("--dname", default="cora")
    ap.add_argument("--All_num_layers", type=int, required=True)
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--study_dir", default="../logs/optuna_cora")
    ap.add_argument("--seed", type=int, default=0, help="sampler seed (vary per worker)")
    a = ap.parse_args()

    os.makedirs(a.study_dir, exist_ok=True)
    study_name = f"{a.dname}_{a.method}_L{a.All_num_layers}"
    storage = JournalStorage(JournalFileBackend(os.path.join(a.study_dir, f"{study_name}.log")))

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to cpu", flush=True)
        a.device = "cpu"

    base_args = build_base_args(a.method, a.dname, a.device, a.epochs, a.num_seeds)

    raw = HyperDataset(base_args)
    base_data = data_processing(base_args, raw)
    base_data._initialization_()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=8),
    )

    protocol = 'main-depth-matched-hnhn-v1'
    recorded = study.user_attrs.get('protocol')
    if recorded is None and study.trials:
        raise ValueError('Historical study has no current protocol identity; use a new study directory')
    if recorded not in (None, protocol):
        raise ValueError('Study protocol differs; use a new study directory')
    identity = dict(epochs=a.epochs, tuning_seeds=a.num_seeds, source_sha256=source_hash())
    previous = study.user_attrs.get('identity')
    if study.trials and previous != identity:
        raise ValueError('Study source or training settings changed; use a new output directory')
    study.set_user_attr('protocol', protocol)
    study.set_user_attr('identity', identity)

    def objective(trial):
        params = suggest(trial, a.method, a.All_num_layers)
        t0 = time.time()
        try:
            val, test, test_std = run_config(base_args, base_data, params, a.All_num_layers)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM")
        # recorded for reporting only -- never used for selection
        trial.set_user_attr("test_acc", test)
        trial.set_user_attr("test_std", test_std)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"[trial {trial.number}] val={val:.2f} test={test:.2f} "
              f"({time.time()-t0:.0f}s) {params}", flush=True)
        return val

    complete_search(study, objective, target=a.n_trials, max_attempts=max(10 * a.n_trials, 30))

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        b = study.best_trial
        print(f"\nBEST {study_name}: val={b.value:.2f} "
              f"test={b.user_attrs.get('test_acc', float('nan')):.2f}")
        print(f"BEST PARAMS {study_name}: {b.params}")
    print(f"completed trials in study: {len(done)}", flush=True)


if __name__ == "__main__":
    main()
