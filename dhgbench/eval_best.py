"""
eval_best.py -- read the finished Optuna studies and produce the final table.

For every study (dataset, method, depth) it takes the configuration with the best
VALIDATION accuracy and re-runs it with more seeds to get the reported test accuracy.
Validation selects; test only reports.

  python eval_best.py --study_dir ../logs/optuna_cora --num_seeds 10 \
      --out ../logs/optuna_cora/final_table.csv
"""
import argparse
import glob
import json
from pathlib import Path
import os
import re
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from lib_models import _semi_methods_
from experiments.classification_results import write_table
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from tune_optuna import build_base_args, run_config, source_hash

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_dir", default="../logs/optuna_cora")
    ap.add_argument("--dname", default="cora")
    ap.add_argument("--num_seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only_method", choices=_semi_methods_, default=None)
    ap.add_argument("--only_depth", type=int, default=None)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        a.device = "cpu"

    logs = sorted(glob.glob(os.path.join(a.study_dir, f"{a.dname}_*_L*.log")))
    if not logs:
        raise FileNotFoundError(f"no studies found in {a.study_dir}")

    rows = []
    cache = {}
    for path in logs:
        name = os.path.basename(path)[:-4]
        m = re.match(rf"{re.escape(a.dname)}_(.+)_L(\d+)$", name)
        if not m:
            continue
        method, depth = m.group(1), int(m.group(2))
        if method not in _semi_methods_:
            continue
        if a.only_method and method != a.only_method:
            continue
        if a.only_depth is not None and depth != a.only_depth:
            continue

        try:
            study = optuna.load_study(study_name=name,
                                      storage=JournalStorage(JournalFileBackend(path)))
        except Exception as e:
            raise RuntimeError(f"Cannot load study {name}") from e
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            raise RuntimeError(f"{name}: no completed trials")

        if study.user_attrs.get('protocol') != 'main-depth-matched-hnhn-v1':
            raise ValueError('This is not a current main-depth study; tune into a new study directory')
        identity = study.user_attrs.get('identity', {})
        if identity.get('source_sha256') != source_hash() or identity.get('epochs') != a.epochs:
            raise ValueError('Evaluation source or epoch count differs from tuning; use a new study')
        best = study.best_trial
        if method not in cache:
            ba = build_base_args(method, a.dname, a.device, a.epochs, a.num_seeds)
            raw = HyperDataset(ba)
            d = data_processing(ba, raw)
            d._initialization_()
            cache[method] = (ba, d)
        base_args, base_data = cache[method]

        # run_config, not ExpAgent.running: the evaluator returns {metric: [train, val,
        # test]} and ExpAgent.test_dict keeps only the test column under the key "acc",
        # so the validation number we also want to record is only reachable this way.
        val, test, test_std, per_seed = run_config(base_args, base_data, best.params, depth, return_per_seed=True)

        row = dict(
            dataset=a.dname, method=method, depth=depth,
            protocol="main-depth-matched-hnhn-v1",
            architecture="MatchedHNHN" if method in ("HNHN", "HNHNII") else method,
            n_trials=len(done),
            tune_val=float(best.value),
            test_acc=test,
            test_std=test_std,
            val_acc=val,
            num_seeds=a.num_seeds,
            params=str(best.params),
        )
        result_path = Path(a.out or os.path.join(a.study_dir, 'final_table.csv')).parent / f'{method}_L{depth}.json'
        temporary = result_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(dict(summary=row, selected_trial=best.number,
                    epochs=a.epochs, source_sha256=identity["source_sha256"], params=best.params, seed_ids=list(range(a.num_seeds)),
                    per_seed_columns=['train', 'validation', 'test'], per_seed=per_seed), indent=2) + '\n')
        temporary.replace(result_path)
        rows.append(row)
        print(f"  {method:<16} L={depth:<3} trials={len(done):<3} "
              f"test={row['test_acc']:.2f}+-{row['test_std']:.2f}", flush=True)

    if not rows:
        raise RuntimeError("No studies matched the requested method/depth")
    out = a.out or os.path.join(a.study_dir, "final_table.csv")
    write_table(out, rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")

    # accuracy-vs-depth, base vs II, which is the point of the exercise
    depths = sorted({r["depth"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    print("\ntest accuracy vs depth")
    print(f"  {'method':<18}" + "".join(f"{('L=' + str(d)):>9}" for d in depths))
    for meth in methods:
        line = f"  {meth:<18}"
        for d in depths:
            hit = [r for r in rows if r["method"] == meth and r["depth"] == d]
            line += f"{hit[0]['test_acc']:>9.2f}" if hit else f"{'-':>9}"
        print(line)


if __name__ == "__main__":
    main()
