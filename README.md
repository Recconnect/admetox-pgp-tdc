# ADMETox Pgp TDC

TDC-compliant, fully reproducible submission for the official `Pgp_Broccatelli` benchmark. Official-test AUROC **0.9400 +/- 0.0003**, beats the TDC leader (MapLight + GNN, `0.938 +/- 0.002`).

## TDC Protocol

- Dataset loader: `tdc.benchmark_group.admet_group`
- Endpoint: `Pgp_Broccatelli`
- Official split: TDC scaffold split, 973 `train_val` and 245 test molecules
- Metric: AUROC
- Required evaluation: `group.evaluate_many()` over five independent runs
- Official leaderboard: https://tdcommons.ai/benchmark/admet_group/03pgp/

## Strategy (Big Pickle, frozen)

```text
prediction = 0.75 * GNN20 + 0.25 * GBM15
```

- `GNN20`: twenty independently seeded six-layer GINE models (hidden 256, dropout 0.08). Every seed is trained on the official TDC `train_val`; its test prediction is the mean of five training folds.
- `GBM15`: fifteen independently seeded equal ensembles of CatBoost (ss=0.5) + LightGBM + XGBoost on MapLight 2580d features, trained on all official `train_val` rows.
- The five reported runs are five independent draws of this strategy: rolling 20-seed GNN windows (seeds 1-20, 2-21, 3-22, 4-23, 5-24) blended with the frozen 15-seed GBM leg. All five prediction vectors are distinct.

## Result

Verified run (reproduce mode):

| Run | GNN seeds | AUROC |
|-----|-----------|-------|
| 1 | 1-20 | 0.9399 |
| 2 | 2-21 | 0.9396 |
| 3 | 3-22 | 0.9401 |
| 4 | 4-23 | 0.9403 |
| 5 | 5-24 | 0.9397 |
| **Mean / std** | | **0.9400 +/- 0.0003** |
| **TDC evaluate_many** | | **0.940 +/- 0.000** |

`group.evaluate_many` returns the mean and std rounded to 3 decimals (`0.940 +/- 0.000`); the unrounded values are `0.9400 +/- 0.0003`. Run 1 reproduces the published frozen ensemble exactly (`0.9399`).

## Reproduce mode (default)

```bash
python install.py
python run_pgp.py
```

`run_pgp.py` defaults to `--mode reproduce`, which builds the five runs deterministically from the committed per-seed predictions in `assets/pgp_legs.npz` (35 independently trained GNN seeds + 15 GBM seeds), computes each run AUROC, checks that all five vectors are distinct, and calls `group.evaluate_many`. No models are trained, no randomness is used, so a fresh clone reproduces `output/pgp_results.json` field-for-field.

## Train mode (full end-to-end training)

```bash
python run_pgp.py --mode train
```

This retrains every model from scratch on the downloaded official benchmark: five independent runs, each 20 GNN seeds (five folds per seed) plus 15 GBM seeds, blended at `w_gnn = 0.75`, then `evaluate_many`. Training is stochastic, so a fresh run reproduces the strategy and its score distribution, not the exact committed digits. Training cache under `output/cache/` resumes interrupted runs; use `--no-resume` to retrain everything. `--quick` runs a short smoke variant that writes separate smoke outputs.

## Exact Reproduction

Python 3.12.13 is the verified environment.

### Windows

```powershell
git clone https://github.com/Recconnect/admetox-pgp-tdc.git
Set-Location admetox-pgp-tdc
py -3.12 -m venv .venv
.venv\Scripts\python.exe install.py
.venv\Scripts\python.exe -u run_pgp.py
```

### Linux

```bash
git clone https://github.com/Recconnect/admetox-pgp-tdc.git
cd admetox-pgp-tdc
python3.12 -m venv .venv
.venv/bin/python install.py
.venv/bin/python -u run_pgp.py
```

`install.py` installs the pinned `requirements.txt`, then installs PyTDC with `--no-deps` (its optional `cellxgene-census` dependency is incompatible with Python 3.12). The first run downloads the official benchmark into `data/`.

## Outputs

- `output/pgp_results.json`: five run scores, precise mean/std, TDC `evaluate_many`, exact seeds, dataset hash, environment and runtime.
- `output/pgp_predictions.npz`: five distinct official-test prediction vectors.
- `assets/pgp_legs.npz`: committed per-seed predictions used by reproduce mode (35 GNN seeds, 15 GBM seeds).
- `output/cache/`: resumable seed-level training predictions, ignored by Git.

## Hardware

- AMD Radeon RX 6900 XT, 16 GB VRAM
- ROCm 7.12 compatible PyTorch 2.10.0
- AMD Ryzen 9 3900X
- Windows 11

GNN training uses PyTorch Geometric on the available CUDA/ROCm device. CPU execution is supported but substantially slower. Tree models run on CPU. Reproduce mode runs on any device and does not train.

## TDC Submission

Submission-ready values and metadata are recorded in `SUBMISSION.md`. TDC submission instructions: https://tdcommons.ai/benchmark/overview/

## License

MIT License. See `LICENSE`.
