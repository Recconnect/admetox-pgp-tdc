"""TDC Pgp_Broccatelli submission: full-data MapLight GBM + GINE ensemble.

The final submission is 0.80 * averaged GNN prediction + 0.20 * averaged
GBM prediction. The blend weight was selected with GNN/GBM OOF predictions.
"""
import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, rdchem
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from tdc.benchmark_group import admet_group

for channel in ["rdApp.info", "rdApp.warning", "rdApp.error", "rdApp.debug"]:
    RDLogger.DisableLog(channel)

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
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
        rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ],
}
BOND_FEATURES = {
    "bond_type": [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
                  rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC],
    "stereo": [rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOANY,
               rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE,
               rdchem.BondStereo.STEREOCIS, rdchem.BondStereo.STEREOTRANS],
}


def log(message):
    print(message, flush=True)


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
    graphs, indices = [], []
    for index, value in enumerate(smiles):
        graph = mol_to_graph(value)
        if graph is not None:
            graphs.append(graph)
            indices.append(index)
    return graphs, np.asarray(indices)


class GINEClassifier(nn.Module):
    """The 6-layer 256-hidden GINE used for the submitted Pgp ensemble."""
    def __init__(self, hidden=256, layers=6, dropout=0.08):
        super().__init__()
        self.node_embed = nn.Linear(NODE_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_DIM, hidden)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            )
            self.layers.append(GINEConv(mlp, eps=0.0, train_eps=False))
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

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
    pos_weight = torch.tensor([np.mean(valid_y == 0) / max(np.mean(valid_y == 1), 1e-6)], device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=max(1, len(train_loader) * epochs),
        pct_start=0.1, anneal_strategy="cos",
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
    return predict(model, valid_loader), predict(model, test_loader), best_auc


def run_gnn(graphs, y, test_graphs, seeds, epochs, patience):
    """Return seed-level OOF and test predictions from 5-fold CV GNNs."""
    test_predictions, oof_predictions = [], []
    for seed in seeds:
        log(f"  GNN seed {seed}: 5-fold CV")
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=np.float64)
        fold_test = []
        for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
            valid_pred, test_pred, _ = train_gnn_fold(
                [graphs[i] for i in train_idx], y[train_idx],
                [graphs[i] for i in valid_idx], y[valid_idx], test_graphs,
                seed, epochs, patience,
            )
            oof[valid_idx] = valid_pred
            fold_test.append(test_pred)
        test_predictions.append(np.mean(fold_test, axis=0))
        oof_predictions.append(oof)
    return np.asarray(oof_predictions), np.asarray(test_predictions)


def maplight_features(smiles):
    count = len(smiles)
    morgan = np.zeros((count, 1024), dtype=np.float32)
    avalon = np.zeros((count, 1024), dtype=np.float32)
    erg = np.zeros((count, 315), dtype=np.float32)
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
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
        if mol is None:
            continue
        for column, (_, function) in enumerate(descriptors):
            try:
                result = function(mol)
                if result is not None and np.isfinite(result):
                    rdkit[row, column] = float(result)
            except Exception:
                pass
    return np.hstack([morgan, avalon, erg, rdkit]).astype(np.float32)


def gbm_models(seed):
    return [
        cb.CatBoostClassifier(iterations=1000, random_strength=2, loss_function="Logloss", random_seed=seed,
                              verbose=0, thread_count=4, allow_writing_files=False),
        cb.CatBoostClassifier(iterations=1000, random_strength=2, subsample=0.5, sampling_frequency="PerTree",
                              loss_function="Logloss", random_seed=seed, verbose=0, thread_count=4,
                              allow_writing_files=False),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1,
                           class_weight="balanced", random_state=seed, verbose=-1, n_jobs=4),
        xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          random_state=seed, verbosity=0, n_jobs=4),
        xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, subsample=0.7, colsample_bytree=0.7,
                          random_state=seed, verbosity=0, n_jobs=4),
    ]


def run_gbm(train_x, y, test_x, seeds):
    predictions = []
    for seed in seeds:
        log(f"  GBM seed {seed}: 5 configs on all train_val")
        seed_predictions = []
        for model in gbm_models(seed):
            model.fit(train_x, y)
            seed_predictions.append(model.predict_proba(test_x)[:, 1])
        predictions.append(np.mean(seed_predictions, axis=0))
    return np.asarray(predictions)


def parse_seeds(value):
    if "-" in value:
        start, end = value.split("-", maxsplit=1)
        return list(range(int(start), int(end) + 1))
    return [int(seed) for seed in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description="TDC Pgp_Broccatelli final submission")
    parser.add_argument("--seeds", default="1-15", help="Seed range or comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--w-gnn", type=float, default=0.80, dest="w_gnn")
    parser.add_argument("--quick", action="store_true", help="Five-seed smoke run; not the reported result")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.quick:
        seeds = seeds[:5]
        args.epochs = min(args.epochs, 30)
    if len(seeds) < 5:
        raise ValueError("TDC evaluate_many requires at least five seeds.")

    start = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    log("=" * 70)
    log("ADMETox.AI: TDC Pgp_Broccatelli Final Submission")
    log("=" * 70)
    log(f"Device: {DEVICE}; seeds: {seeds}; GNN weight: {args.w_gnn:.2f}")

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(ENDPOINT)
    name = benchmark["name"]
    train_val, test = benchmark["train_val"], benchmark["test"]
    train_smiles, test_smiles = train_val["Drug"].tolist(), test["Drug"].tolist()
    train_y, test_y = train_val["Y"].to_numpy(dtype=int), test["Y"].to_numpy(dtype=int)
    log(f"Dataset: train_val={len(train_y)}, test={len(test_y)}, positives={train_y.mean():.1%}")

    log("Building molecular graphs...")
    train_graphs, train_valid = build_graphs(train_smiles)
    test_graphs, test_valid = build_graphs(test_smiles)
    if len(train_valid) != len(train_y) or len(test_valid) != len(test_y):
        raise ValueError("Invalid SMILES found; this reference benchmark is expected to contain only valid SMILES.")
    gnn_oof, gnn_seed_predictions = run_gnn(train_graphs, train_y, test_graphs, seeds, args.epochs, args.patience)
    gnn_prediction = gnn_seed_predictions.mean(axis=0)

    log("Computing MapLight 2580d features...")
    features = maplight_features(train_smiles + test_smiles)
    gbm_seed_predictions = run_gbm(features[:len(train_y)], train_y, features[len(train_y):], seeds)
    gbm_prediction = gbm_seed_predictions.mean(axis=0)

    blend_prediction = args.w_gnn * gnn_prediction + (1.0 - args.w_gnn) * gbm_prediction
    ensemble_auc = float(roc_auc_score(test_y, blend_prediction))
    gnn_auc = float(roc_auc_score(test_y, gnn_prediction))
    gbm_auc = float(roc_auc_score(test_y, gbm_prediction))
    seed_blends = [
        {name: args.w_gnn * gnn_seed_predictions[index] + (1.0 - args.w_gnn) * gbm_prediction}
        for index in range(len(seeds))
    ]
    tdc_result = group.evaluate_many(seed_blends)
    tdc_mean, tdc_std = tdc_result[name]
    elapsed = time.time() - start

    result = {
        "endpoint": ENDPOINT,
        "sota": SOTA,
        "seeds": seeds,
        "protocol": "full-data MapLight GBM + 5-fold GINE, 0.80 GNN blend",
        "gnn_ensemble_auroc": gnn_auc,
        "gbm_ensemble_auroc": gbm_auc,
        "submission_ensemble_auroc": ensemble_auc,
        "gap_to_sota": ensemble_auc - SOTA,
        "tdc_evaluate_many": [float(tdc_mean), float(tdc_std)],
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": elapsed,
    }
    with open(OUTPUT_DIR / "pgp_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(OUTPUT_DIR / "pgp_submission.npz", prediction=blend_prediction, y_test=test_y)

    log("=" * 70)
    log(f"GNN ensemble AUROC:        {gnn_auc:.4f}")
    log(f"GBM ensemble AUROC:        {gbm_auc:.4f}")
    log(f"Submission ensemble AUROC: {ensemble_auc:.4f} (SOTA {SOTA:.3f})")
    log(f"evaluate_many:             {tdc_mean:.4f} +/- {tdc_std:.4f}")
    log(f"Saved: {OUTPUT_DIR / 'pgp_results.json'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
