# Measuring Hypergraph Over-Smoothing Through the Propagation Operator

The public workflow has four scripts: run controlled linear propagation, plot its energies, run the classification depth study, and plot its accuracies. Run the commands below from `Code/`, the Git repository root. Python 3.11 or 3.12 and Linux are recommended for training.

## 1. Installation

Create an environment and install PyTorch with matching compiled PyG extensions before installing the remaining dependencies from the single requirements file.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU installation.
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: torch_scatter==2.1.2 torch_sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
python -m pip install -r requirements.txt

python -c "import torch, torch_scatter, torch_sparse; print(torch.__version__, torch.version.cuda)"
python tests/verify_release.py
```

For a compatible CUDA 13.0 environment, substitute `cu130` for the Torch index suffix `cpu` and `+cu130.html` for the extension wheel suffix `+cpu.html` before installing `requirements.txt`. Follow the [PyG wheel compatibility instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) when choosing a different supported CUDA build. Use `--device cuda:0` in training commands to enable GPU execution. CPU/GPU and dependency differences can affect numerical results.

## 2. Controlled linear propagation

Run the operators, then generate the figure:

```bash
python scripts/run_energy.py
python scripts/plot_energy.py
```

The runner writes to `results/energy/`. It uses the frozen inputs in `data/energy_inputs/` and requires no Torch training. Defaults cover Cora, Citeseer, 20News100, House-Committee, Pokec, and NTU2012, with both original features and Gaussian controls. The figure defaults to original features and writes `outputs/figures/energy_trajectory.pdf` and `.png`.

```bash
# Run and plot one dataset.
python scripts/run_energy.py --datasets cora --features real
python scripts/plot_energy.py --datasets cora

# Plot Gaussian controls from the same result directory.
python scripts/plot_energy.py --features random
```

Completed energy tasks are reused. Add `--overwrite` to recompute their measurements in the output directory, or choose a new `--out` directory for an independent run and pass it as `--energy-root` to the plot command. Figures can be sent to the manuscript with `--out ../Manuscript/figures`.

The protocol applies HGNN, HNHN, and UniGCN for 256 float64 steps, with identity channel maps and activations, no training, and restart weight 0.1. HNHN exponents are -1.5 and -0.5. Gaussian controls use seeds 0–9, width 64, and initial squared Frobenius norm 64n. Operators share the largest connected component: 1,330/2,708 Cora nodes, 1,019/3,312 Citeseer nodes, and 64/14,998 Pokec nodes; the other datasets are connected. Numerical floor flags remain in the trajectory CSVs.

## 3. Main classification depth study

Run tuning and final evaluation, then generate the figure:

```bash
python scripts/run_classification.py --device cuda:0
python scripts/plot_classification.py
```

The runner installs and verifies the bundled processed data automatically. It writes Optuna studies, selected parameters, per-seed evaluation scores, cell summaries, and `final_table.csv` under `results/classification/`. The plot command collects completed evaluation cells from that directory and writes `outputs/figures/depth_trajectory.png`, `.pdf`, and its input `.csv`. A partial run produces a figure of the completed cells only.

Every classification CSV formats `tune_val`, `test_acc`, `test_std`, and `val_acc` with exactly two decimal places, including trailing zeros, using round-half-even. This applies to individual cells, combined tables, replay summaries, and figure-input exports. The shared specification is in `dhgbench/experiments/classification_results.py`. JSON results, raw seed scores, and hyperparameter configurations retain numeric precision; formatting happens only when exporting CSV.

The main grid contains six datasets × ten methods × six depths (360 cells). Models are HGNN, HNHN, HyperGCN, AllSetTransformer, UniGCN, and their II variants. Depths are 2, 4, 8, 16, 32, and 64. Each cell targets 30 completed tuning trials using three seeds, followed by ten-seed evaluation of the validation-selected configuration. The default is 200 epochs. Repeating tuning resumes the completed-trial target rather than adding another 30 trials. Source and training-setting identities are checked before resuming or evaluating a study.

```bash
# Inspect the grid without training or extracting data.
python scripts/run_classification.py --dry-run

# Run a subset of the main study.
python scripts/run_classification.py --datasets cora --methods HNHN HNHNII --depths 2 4
python scripts/plot_classification.py --datasets cora

# Tuning and evaluation can also be run separately.
python scripts/run_classification.py tune --datasets cora --methods UniGCN --depths 2
python scripts/run_classification.py evaluate --datasets cora --methods UniGCN --depths 2

# Short CPU execution check; output is kept separate from the reference results.
python scripts/run_classification.py --datasets cora --methods HNHN HNHNII \
  --depths 2 --trials 1 --epochs 2 --seeds 1 --out outputs/smoke
python scripts/plot_classification.py --root outputs/smoke --out outputs/smoke/depth.png
```

Omit `--device` for CPU. `--seeds` overrides both tuning and final seed counts for short checks. Independent cells can run concurrently, but do not launch duplicate workers for the same cell/output directory. Plot after workers finish to collect their completed cells.

Classification uses random 50/25/25 node splits, full-batch Adam and NLL loss, and validation-selected epochs, with later epochs winning ties. UniGCN and UniGCNII use their fixed optimizer (learning rate 0.01 and parameter-group weight decays 0.01/0.0005). Current UniGCN rejects row normalization, so its new search fixes `use_norm=False`.

The 360 configurations in `configs/selected.json` are extracted from `results/classification/all_six_datasets.csv`: six datasets × ten models × six depths. Parameters, seed counts, trial counts, and reference metrics match the corresponding result rows. The 200-epoch setting follows the documented protocol because the CSV does not contain an epoch column. Replay currently remains enabled for the other seven methods:

```bash
python scripts/run_classification.py selected --datasets cora --methods HGNN HGNNII --depths 2
python scripts/plot_classification.py --kind selected
```

The extraction preserves recorded settings rather than adapting them to the current code. In the local matched-HNHN partial table, 54 HNHN/HNHNII parameter dictionaries match the reference CSV; the remaining 18 HNHN rows have no architecture confirmation in that partial table. Also, 22 UniGCN entries use `use_norm=True`, which the current implementation rejects. The existing `selected` guards for HNHN/HNHNII and UniGCN remain until those compatibility issues are reconciled; fresh `run` or `tune/evaluate` uses the current models. Extraction checks are recorded in `provenance/validation/config-extraction.json`. The default figure command reads newly generated main-study results, not the archived `all_six_datasets.csv`; it requires explicit `MatchedHNHN` architecture metadata for HNHN rows.

## Attribution and data terms

The benchmark runner adapts DHG-bench and implementations of [AllSet](https://github.com/jianhao2016/AllSet), [UniGNN](https://github.com/OneForward/UniGNN), [HyperGCN](https://github.com/malllabiisc/HyperGCN), and HNHN. The AllSet MIT notice is preserved in [licenses/AllSet-MIT.txt](licenses/AllSet-MIT.txt). Existing inline notices remain in the source; that notice does not assign a blanket license to every file in this repository.

Benchmark sources follow the loaders in `dhgbench/lib_dataset/load_other_datasets.py` and the AllSet/UniGNN conventions. Pokec uses the heterogeneous-data loader. Complete original dataset download and license metadata were not stored with the caches. Dataset-specific terms still apply, and no new license is assigned to third-party data. The manuscript currently uses anonymous authors; no publication identifier or author identity is assumed here.
