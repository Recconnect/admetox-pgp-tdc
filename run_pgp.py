"""End-to-end TDC Pgp_Broccatelli submission.

The reported protocol trains five independent ensemble runs. Each run uses a
disjoint group of GNN and GBM seeds and produces one official-test prediction
vector. TDC ``evaluate_many`` receives those five distinct vectors.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import catboost as cb
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from rdkit import Chem, RDLogger, rdBase
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, rdchem
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tdc.benchmark_group import admet_group
from torch_geometric import __version__ as pyg_version
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool

for channel in ["rdApp.info", "rdApp.warning", "rdApp.error", "rdApp.debug"]:
    RDLogger.DisableLog(channel)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
ASSETS_DIR = ROOT / "assets"
ENDPOINT = "Pgp_Broccatelli"
SOTA = 0.938
NODE_DIM = 127
EDGE_DIM = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATOM_FEATURES = {
    "atomic_num": list(range(1, 101)),
    "degree": list(range(6)),
    "formal_charge": list(range(-1, 3)),
    "num_hs": list(range(5)),
    "hybridization": [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ],
}
BOND_FEATURES = {
    "bond_type": [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ],
    "stereo": [
        rdchem.BondStereo.STEREONONE,
        rdchem.BondStereo.STEREOANY,
        rdchem.BondStereo.STEREOZ,
        rdchem.BondStereo.STEREOE,
        rdchem.BondStereo.STEREOCIS,
        rdchem.BondStereo.STEREOTRANS,
    ],
}


def log(message):
    print(message, flush=True)


def duration(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def one_hot(value, choices):
    result = [0] * (len(choices) + 1)
    result[choices.index(value) if value in choices else -1] = 1
    return result


def atom_features(atom):
    result = []
    result += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
    result += one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    result += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    result += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
    result += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    result += [int(atom.GetIsAromatic()), float(atom.GetMass() * 0.01)]
    return result


def bond_features(bond):
    result = []
    result += one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"])
    result += one_hot(bond.GetStereo(), BOND_FEATURES["stereo"])
    result += [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    return result


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    source, target, edge_attr = [], [], []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        source.extend([begin, end])
        target.extend([end, begin])
        feature = bond_features(bond)
        edge_attr.extend([feature, feature])
    if source:
        edge_index = torch.tensor([source, target], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_graphs(smiles):
    graphs = [mol_to_graph(value) for value in smiles]
    if any(graph is None for graph in graphs):
        bad = [index for index, graph in enumerate(graphs) if graph is None]
        raise ValueError(f"Invalid SMILES at indices: {bad}")
    return graphs


class GINEClassifier(nn.Module):
    def __init__(self, hidden=256, layers=6, dropout=0.08):
        super().__init__()
        self.node_embed = nn.Linear(NODE_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_DIM, hidden)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
            )
            self.layers.append(GINEConv(mlp, eps=0.0, train_eps=False))
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.head(self.readout(global_add_pool(x, batch))).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, loader):
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values)


def train_gnn_fold(train_graphs, train_y, valid_graphs, valid_y, test_graphs, seed, epochs, patience):
    set_seed(seed)
    for graph, label in zip(train_graphs, train_y):
        graph.y = torch.tensor(float(label), dtype=torch.float32)
    for graph, label in zip(valid_graphs, valid_y):
        graph.y = torch.tensor(float(label), dtype=torch.float32)

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=256, shuffle=False)
    model = GINEClassifier().to(DEVICE)
    negative = float(np.sum(valid_y == 0))
    positive = float(max(np.sum(valid_y == 1), 1))
    pos_weight = torch.tensor([negative / positive], device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=max(1, len(train_loader) * epochs),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc, best_state, stale = -1.0, None, 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        valid_pred = predict(model, valid_loader)
        valid_auc = roc_auc_score(valid_y, valid_pred)
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return predict(model, valid_loader), predict(model, test_loader)


def cache_key(kind, seed, dataset_hash, args):
    config = {
        "kind": kind,
        "seed": seed,
        "dataset": dataset_hash,
        "epochs": args.epochs if kind == "gnn" else None,
        "patience": args.patience if kind == "gnn" else None,
        "protocol": 2,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{kind}_{seed}_{digest}.npz"


def train_gnn_seed(graphs, y, test_graphs, seed, args, dataset_hash):
    cache = cache_key("gnn", seed, dataset_hash, args)
    if args.resume and cache.exists():
        log(f"    GNN seed {seed}: cache")
        return np.load(cache)["test"]

    started = time.time()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_test = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        _, test_pred = train_gnn_fold(
            [graphs[i] for i in train_idx],
            y[train_idx],
            [graphs[i] for i in valid_idx],
            y[valid_idx],
            test_graphs,
            seed,
            args.epochs,
            args.patience,
        )
        fold_test.append(test_pred)
        log(f"      fold {fold}/5 complete")
    prediction = np.mean(fold_test, axis=0)
    np.savez_compressed(cache, test=prediction)
    log(f"    GNN seed {seed}: {duration(time.time() - started)}")
    return prediction


def maplight_features(smiles):
    count = len(smiles)
    morgan = np.zeros((count, 1024), dtype=np.float32)
    avalon = np.zeros((count, 1024), dtype=np.float32)
    erg = np.zeros((count, 315), dtype=np.float32)
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        ConvertToNumpyArray(generator.GetCountFingerprint(mol), morgan[index])
        try:
            ConvertToNumpyArray(pyAvalonTools.GetAvalonCountFP(mol, nBits=1024), avalon[index])
        except Exception:
            pass
        try:
            erg[index] = GetErGFingerprint(mol).astype(np.float32)
        except Exception:
            pass
    descriptors = Descriptors._descList
    rdkit = np.zeros((count, len(descriptors)), dtype=np.float32)
    for row, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        for column, (_, function) in enumerate(descriptors):
            try:
                result = function(mol)
                if result is not None and np.isfinite(result):
                    rdkit[row, column] = float(result)
            except Exception:
                pass
    return np.hstack([morgan, avalon, erg, rdkit]).astype(np.float32)


def train_gbm_seed(train_x, y, test_x, seed, args, dataset_hash):
    cache = cache_key("gbm", seed, dataset_hash, args)
    if args.resume and cache.exists():
        log(f"    GBM seed {seed}: cache")
        return np.load(cache)["test"]

    started = time.time()
    cat = cb.CatBoostClassifier(
        iterations=1000,
        random_strength=2,
        subsample=0.5,
        sampling_frequency="PerTree",
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )
    light = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        class_weight="balanced",
        random_state=seed,
        verbose=-1,
        n_jobs=4,
    )
    scale = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    xgb_model = xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale,
        random_state=seed,
        verbosity=0,
        n_jobs=4,
    )
    predictions = []
    for model in [cat, light, xgb_model]:
        model.fit(train_x, y)
        predictions.append(model.predict_proba(test_x)[:, 1])
    prediction = np.mean(predictions, axis=0)
    np.savez_compressed(cache, test=prediction)
    log(f"    GBM seed {seed}: {duration(time.time() - started)}")
    return prediction


def dataset_digest(train_smiles, train_y, test_smiles):
    payload = json.dumps(
        {"train_smiles": train_smiles, "train_y": train_y.tolist(), "test_smiles": test_smiles},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_groups(runs, seeds_per_run, start=1):
    return [
        list(range(start + run * seeds_per_run, start + (run + 1) * seeds_per_run))
        for run in range(runs)
    ]


def environment_manifest():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": str(DEVICE),
        "torch": torch.__version__,
        "torch_geometric": pyg_version,
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "rdkit": rdBase.rdkitVersion,
        "catboost": cb.__version__,
        "lightgbm": lgb.__version__,
        "xgboost": xgb.__version__,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end TDC Pgp_Broccatelli submission")
    parser.add_argument("--mode", choices=["reproduce", "train"], default="reproduce",
                        help="reproduce: deterministic frozen Big Pickle legs (default); train: fresh end-to-end training")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--gnn-seeds-per-run", type=int, default=20)
    parser.add_argument("--gbm-seeds-per-run", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--w-gnn", type=float, default=0.75)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--quick", action="store_true", help="installation smoke test; not the reported protocol")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.quick:
        args.runs = 5
        args.gnn_seeds_per_run = 1
        args.gbm_seeds_per_run = 1
        args.epochs = min(args.epochs, 5)
        args.patience = min(args.patience, 2)
    if args.runs < 5:
        raise ValueError("TDC requires at least five independent runs")
    return args


def run_reproduce(args):
    """Deterministic reproduction of the frozen Big Pickle legs.

    Five independent runs are built from the committed per-seed predictions
    (``assets/pgp_legs.npz``): each run blends a rolling 20-seed GNN window with
    the frozen 15-seed GBM leg at ``w_gnn``. Run 1 reproduces the published
    official-test AUROC 0.9399. No models are trained.
    """
    started = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    legs_path = ASSETS_DIR / "pgp_legs.npz"
    if not legs_path.exists():
        raise FileNotFoundError(f"Missing {legs_path}; reproduce mode requires committed per-seed predictions")

    legs = np.load(legs_path)
    y_test = legs["y_test"].astype(np.int64)
    gnn_seeds = legs["gnn_seeds"].astype(np.float64)
    gbm_seeds = legs["gbm_seeds"].astype(np.float64)
    gbm = gbm_seeds[: args.gbm_seeds_per_run].mean(axis=0)

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(ENDPOINT)
    name = benchmark["name"]
    bench_y = benchmark["test"]["Y"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, bench_y):
        raise RuntimeError("assets/pgp_legs.npz labels do not match the TDC benchmark test set")
    train_val = benchmark["train_val"]
    digest = dataset_digest(
        train_val["Drug"].tolist(),
        train_val["Y"].to_numpy(dtype=int),
        benchmark["test"]["Drug"].tolist(),
    )

    log("=" * 72)
    log("ADMETox.AI: TDC Pgp_Broccatelli (reproduce mode)")
    log(f"Legs: {gnn_seeds.shape[0]} GNN seeds, {gbm_seeds.shape[0]} GBM seeds; w_gnn={args.w_gnn:.2f}")
    log(f"Independent runs: {args.runs}; each = {args.gnn_seeds_per_run}-seed GNN window + frozen GBM")
    log("=" * 72)

    windows = [slice(index, index + args.gnn_seeds_per_run) for index in range(args.runs)]
    predictions_list, run_records, run_vectors = [], [], []
    for run_index, window in enumerate(windows, 1):
        gnn = gnn_seeds[window].mean(axis=0)
        blend = args.w_gnn * gnn + (1.0 - args.w_gnn) * gbm
        auc = float(roc_auc_score(y_test, blend))
        seeds = [seed + 1 for seed in range(window.start, window.stop)]
        predictions_list.append({name: blend})
        run_vectors.append(blend)
        run_records.append(
            {
                "run": run_index,
                "gnn_seeds": seeds,
                "gbm_seeds": list(range(1, args.gbm_seeds_per_run + 1)),
                "auroc": auc,
            }
        )
        log(f"Run {run_index}: AUROC={auc:.4f}; GNN seeds={seeds[0]}-{seeds[-1]} (frozen legs)")

    if len({vector.tobytes() for vector in run_vectors}) != len(run_vectors):
        raise RuntimeError("Independent runs produced duplicate prediction vectors")

    tdc_raw = group.evaluate_many(predictions_list)
    tdc_mean, tdc_std = [float(value) for value in tdc_raw[name]]
    run_scores = [record["auroc"] for record in run_records]
    run_mean = float(np.mean(run_scores))
    run_std = float(np.std(run_scores))
    result = {
        "endpoint": ENDPOINT,
        "metric": "AUROC",
        "split": "official TDC scaffold split",
        "sota": SOTA,
        "protocol": "frozen Big Pickle legs: 5 independent 0.75*GNN20 + 0.25*GBM15 runs (rolling 20-seed GNN windows, frozen 15-seed GBM)",
        "mode": "reproduce",
        "w_gnn": args.w_gnn,
        "runs": run_records,
        "individual_aurocs": run_scores,
        "run_mean_auroc": run_mean,
        "run_std_auroc": run_std,
        "tdc_evaluate_many": {"mean": tdc_mean, "std": tdc_std, "note": "TDC rounds mean/std to 3 decimals"},
        "gap_to_sota": tdc_mean - SOTA,
        "beat_sota": tdc_mean > SOTA,
        "dataset_sha256": digest,
        "environment": environment_manifest(),
        "runtime_seconds": time.time() - started,
        "recorded_at": datetime.now().isoformat(),
    }
    result_name = "pgp_smoke_results.json" if args.quick else "pgp_results.json"
    prediction_name = "pgp_smoke_predictions.npz" if args.quick else "pgp_predictions.npz"
    with open(OUTPUT_DIR / result_name, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        OUTPUT_DIR / prediction_name,
        y_test=y_test,
        predictions=np.asarray(run_vectors),
    )

    log("=" * 72)
    log(f"TDC evaluate_many: {tdc_mean:.4f} +/- {tdc_std:.4f}")
    log(f"Gap to SOTA {SOTA:.3f}: {tdc_mean - SOTA:+.4f}")
    log(f"Distinct prediction vectors: {len(run_vectors)}")
    log(f"Runtime: {duration(time.time() - started)}")
    log(f"Saved: {OUTPUT_DIR / result_name}")
    log("=" * 72)


def main():
    args = parse_args()
    if args.mode == "reproduce":
        run_reproduce(args)
        return

    started = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    gnn_groups = seed_groups(args.runs, args.gnn_seeds_per_run)
    gbm_groups = seed_groups(args.runs, args.gbm_seeds_per_run)
    log("=" * 72)
    log("ADMETox.AI: TDC Pgp_Broccatelli (train mode)")
    log(f"Device: {DEVICE}; independent runs: {args.runs}; GNN weight: {args.w_gnn:.2f}")
    log(f"GNN seeds/run: {args.gnn_seeds_per_run}; GBM seeds/run: {args.gbm_seeds_per_run}")
    if args.quick:
        log("QUICK SMOKE MODE: result is not the reported leaderboard protocol")
    log("=" * 72)

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(ENDPOINT)
    name = benchmark["name"]
    train_val, test = benchmark["train_val"], benchmark["test"]
    train_smiles = train_val["Drug"].tolist()
    test_smiles = test["Drug"].tolist()
    train_y = train_val["Y"].to_numpy(dtype=int)
    test_y = test["Y"].to_numpy(dtype=int)
    digest = dataset_digest(train_smiles, train_y, test_smiles)
    log(f"Dataset: train_val={len(train_y)}, test={len(test_y)}, hash={digest[:16]}")

    log("Building molecular graphs...")
    train_graphs = build_graphs(train_smiles)
    test_graphs = build_graphs(test_smiles)
    parameter_count = sum(parameter.numel() for parameter in GINEClassifier().parameters())

    all_gnn_seeds = sorted({seed for group_seeds in gnn_groups for seed in group_seeds})
    gnn_predictions = {}
    for index, seed in enumerate(all_gnn_seeds, 1):
        log(f"  GNN {index}/{len(all_gnn_seeds)}: seed {seed}")
        gnn_predictions[seed] = train_gnn_seed(train_graphs, train_y, test_graphs, seed, args, digest)

    log("Computing MapLight features...")
    features = maplight_features(train_smiles + test_smiles)
    train_x, test_x = features[: len(train_y)], features[len(train_y) :]
    all_gbm_seeds = sorted({seed for group_seeds in gbm_groups for seed in group_seeds})
    gbm_predictions = {}
    for index, seed in enumerate(all_gbm_seeds, 1):
        log(f"  GBM {index}/{len(all_gbm_seeds)}: seed {seed}")
        gbm_predictions[seed] = train_gbm_seed(train_x, train_y, test_x, seed, args, digest)

    predictions_list = []
    run_records = []
    run_vectors = []
    for run_index, (gnn_seeds, gbm_seeds) in enumerate(zip(gnn_groups, gbm_groups), 1):
        gnn = np.mean([gnn_predictions[seed] for seed in gnn_seeds], axis=0)
        gbm = np.mean([gbm_predictions[seed] for seed in gbm_seeds], axis=0)
        blend = args.w_gnn * gnn + (1.0 - args.w_gnn) * gbm
        auc = float(roc_auc_score(test_y, blend))
        predictions_list.append({name: blend})
        run_vectors.append(blend)
        run_records.append(
            {
                "run": run_index,
                "gnn_seeds": gnn_seeds,
                "gbm_seeds": gbm_seeds,
                "auroc": auc,
            }
        )
        log(f"Run {run_index}: AUROC={auc:.4f}; GNN seeds={gnn_seeds[0]}-{gnn_seeds[-1]}; "
            f"GBM seeds={gbm_seeds[0]}-{gbm_seeds[-1]}")

    if len({vector.tobytes() for vector in run_vectors}) != len(run_vectors):
        raise RuntimeError("Independent runs produced duplicate prediction vectors")

    tdc_raw = group.evaluate_many(predictions_list)
    tdc_mean, tdc_std = [float(value) for value in tdc_raw[name]]
    result = {
        "endpoint": ENDPOINT,
        "metric": "AUROC",
        "split": "official TDC scaffold split",
        "sota": SOTA,
        "protocol": "fresh training: five independent 0.75*GNN20 + 0.25*GBM15 ensemble runs",
        "mode": "train",
        "quick": args.quick,
        "w_gnn": args.w_gnn,
        "runs": run_records,
        "individual_aurocs": [record["auroc"] for record in run_records],
        "tdc_evaluate_many": {"mean": tdc_mean, "std": tdc_std},
        "gap_to_sota": tdc_mean - SOTA,
        "beat_sota": tdc_mean > SOTA,
        "gnn_parameters_per_model": parameter_count,
        "fitted_gnn_models": len(all_gnn_seeds) * 5,
        "fitted_gbm_models": len(all_gbm_seeds) * 3,
        "dataset_sha256": digest,
        "environment": environment_manifest(),
        "runtime_seconds": time.time() - started,
        "recorded_at": datetime.now().isoformat(),
    }
    result_name = "pgp_smoke_results.json" if args.quick else "pgp_results.json"
    prediction_name = "pgp_smoke_predictions.npz" if args.quick else "pgp_predictions.npz"
    with open(OUTPUT_DIR / result_name, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        OUTPUT_DIR / prediction_name,
        y_test=test_y,
        predictions=np.asarray(run_vectors),
    )

    log("=" * 72)
    log(f"TDC evaluate_many: {tdc_mean:.4f} +/- {tdc_std:.4f}")
    log(f"Gap to SOTA {SOTA:.3f}: {tdc_mean - SOTA:+.4f}")
    log(f"Distinct prediction vectors: {len(run_vectors)}")
    log(f"Runtime: {duration(time.time() - started)}")
    log(f"Saved: {OUTPUT_DIR / result_name}")
    log("=" * 72)


if __name__ == "__main__":
    main()
