# ADMETox-Pgp-TDC

Reproducible P-glycoprotein substrate prediction for the [TDC ADMET Leaderboard](https://tdcommons.ai/benchmark/admet_group/06pgp/), using the `Pgp_Broccatelli` official split.

## Reproducibility Status

The initial experiment produced an averaged submission-vector AUROC of **0.9391**, above the published TDC SOTA of 0.9380. A subsequent independent full retraining from a public clean clone produced **0.9366** on the same official split and configuration.

The result is therefore a **candidate record, not a submission-ready SOTA claim**. This repository keeps both values visible until repeated independent full runs establish a stable result above 0.9380.

## Results

| Metric | Value |
|---|---:|
| Initial experiment, averaged submission-vector AUROC | 0.9391 |
| Independent clean-clone full retraining | 0.9366 |
| TDC SOTA (MapLight+GNN) | 0.9380 |
| Independent `evaluate_many` seed-level result | 0.9320 +/- 0.0060 |

The averaged submission-vector AUROC and TDC `evaluate_many` are distinct: `evaluate_many` summarizes seed-level blends, whereas the former scores one averaged prediction vector.

## Method

The final prediction is:

```text
0.80 * GNN ensemble + 0.20 * GBM ensemble
```

The blend weight was selected from out-of-fold predictions before final test evaluation.

| Branch | Training protocol | Test AUROC |
|---|---|---:|
| GNN, independent clean run | 6-layer GINE, hidden 256, dropout 0.08, 5-fold CV x 15 seeds | 0.9328 |
| GBM | 5 MapLight models x 15 seeds; every model trains on all `train_val` rows | 0.9287 |
| Final blend, independent clean run | 80% GNN, 20% GBM | 0.9366 |

### GNN

- Graph features: RDKit atom/bond encodings (127 node features, 14 edge features).
- Architecture: six `GINEConv` layers, hidden dimension 256, global-add pooling, MLP classification head.
- Training: five stratified folds per random seed, OneCycleLR, early stopping on each validation fold.

### MapLight GBM Ensemble

The 2580-dimensional feature vector contains Morgan count radius 2 (1024), Avalon count (1024), ErG (315), and RDKit 2D descriptors (217).

1. CatBoost default
2. CatBoost with `subsample=0.5`, `sampling_frequency=PerTree`
3. LightGBM balanced
4. XGBoost default
5. XGBoost HIA-style (`subsample=0.7`, `colsample_bytree=0.7`)

## Quick Start

The first run downloads the official benchmark to `data/`. The full 15-seed run is computationally expensive because it trains 75 tree models and 75 five-fold GNNs.

```bash
git clone https://github.com/Recconnect/admetox-pgp-tdc.git
cd admetox-pgp-tdc

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# PyTDC's optional dependency chain is incompatible with Python 3.12.
pip install PyTDC --no-deps

# Full submission reproduction
python run_pgp.py --seeds 1-15 --w-gnn 0.80
```

For a five-seed smoke run only:

```bash
python run_pgp.py --quick
```

`--quick` is not the reported leaderboard configuration.

## Outputs

| File | Contents |
|---|---|
| `output/pgp_results.json` | Metrics, protocol, seeds, TDC evaluation result |
| `output/pgp_submission.npz` | Averaged submission prediction vector and official test labels for local verification |

`pgp_submission.npz` is generated and intentionally ignored by Git. The committed JSON records the independently reproduced result.

## Reproducibility

- **Benchmark:** `Pgp_Broccatelli`, official TDC `admet_group` split.
- **Data:** train/validation 973 molecules; test 245 molecules.
- **Seeds:** 1 through 15.
- **Hardware used for the record:** AMD Radeon RX 6900 XT (16 GB), AMD Ryzen 9 3900X, Windows 11.
- **Python environment:** Python 3.12, PyTorch 2.10 ROCm, PyTorch Geometric, RDKit, CatBoost, LightGBM, XGBoost.

GPU-enabled PyTorch is required for practical GNN training. A CUDA PyTorch build also works; CPU execution is supported by the code but is substantially slower.

## Troubleshooting

| Issue | Resolution |
|---|---|
| `pip install PyTDC` fails on Python 3.12 | Run `pip install PyTDC --no-deps` after `pip install -r requirements.txt`. |
| TDC download fails | Check network access, remove the local `data/` cache, and rerun. |
| `torch_geometric` cannot import | Install a PyTorch build appropriate for ROCm or CUDA first, then reinstall `torch-geometric`. |
| Different AUROC | Use the exact 15 seeds and `--w-gnn 0.80`; current independent reproduction is 0.9366, so do not claim SOTA until repeated runs are stably above 0.9380. |
| `evaluate_many` is lower than averaged AUROC | Expected: it summarizes individual seed blends, not the averaged submission vector. |

## References

- TDC ADMET benchmark: https://tdcommons.ai/benchmark/admet_group/overview/
- MapLight: https://arxiv.org/abs/2310.00174
- Broccatelli et al., P-glycoprotein substrate dataset.

## Citation

```bibtex
@software{bykadorov2026admetoxpgp,
  author = {Bykadorov, Rodion V.},
  title = {{ADMETox-Pgp-TDC}: GINE and MapLight Gradient-Boosting Ensemble for P-glycoprotein Substrate Prediction},
  year = {2026},
  url = {https://github.com/Recconnect/admetox-pgp-tdc}
}
```

## License

MIT License. See [LICENSE](LICENSE).
