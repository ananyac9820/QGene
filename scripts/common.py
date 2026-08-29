"""
Shared data preparation for the QGene camera-ready rerun.

Every script in scripts/ imports from here so that the train/val/test split,
the feature engineering and the random seed are guaranteed identical across
the classical arm, the quantum arm, the hybrid and the learning curve.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── the one seed used everywhere ──────────────────────────────────────
SEED = 42

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEDUP_CSV = os.path.join(BASE, "data", "brca_mutations_dedup.csv")

FEATURES = [
    'gene_enc', 'type_enc', 'position', 'log_length', 'review_score',
    'nuc_position', 'aa_position', 'has_protein_change', 'is_frameshift',
    'is_snv', 'is_expert', 'nuc_x_review', 'aa_x_gene', 'type_x_gene'
]

# task 8: label-quality ablation drops the two curation-confidence features
ABLATION_DROP = ['review_score', 'is_expert']
FEATURES_ABLATION = [f for f in FEATURES if f not in ABLATION_DROP]


def parse_features(df):
    """Feature engineering - byte-for-byte the logic from retrain_models.py."""
    df = df.copy()

    df['gene_enc'] = df['GeneSymbol'].map({'BRCA1': 0, 'BRCA2': 1}).fillna(0)

    type_map = {
        'single nucleotide variant': 0, 'deletion': 1, 'insertion': 2,
        'duplication': 3, 'indel': 4, 'microsatellite': 5, 'inversion': 6,
        'protein only': 0, 'copy number loss': 1, 'copy number gain': 3,
        'complex': 4, 'variation': 0
    }
    df['type_enc'] = df['Type'].str.lower().map(type_map).fillna(0)

    df['position'] = df['Start'].fillna(0)
    df['mut_length'] = (df['Stop'] - df['Start']).fillna(1).clip(1, 10000)
    df['log_length'] = np.log1p(df['mut_length'])

    review_map = {
        'reviewed by expert panel': 4,
        'criteria provided, multiple submitters, no conflicts': 3,
        'criteria provided, single submitter': 2,
        'no assertion criteria provided': 1,
        'no assertion provided': 0
    }
    df['review_score'] = df['ReviewStatus'].map(review_map).fillna(1)
    df['is_expert'] = (df['ReviewStatus'] == 'reviewed by expert panel').astype(int)

    def get_nuc(x):
        m = re.search(r'c\.[-*]?(\d+)', str(x))
        return int(m.group(1)) if m else 0

    def get_aa(x):
        m = re.search(r'p\.[A-Za-z]+(\d+)', str(x))
        return int(m.group(1)) if m else 0

    df['nuc_position'] = df['Name'].apply(get_nuc)
    df['aa_position'] = df['Name'].apply(get_aa)
    df['has_protein_change'] = df['Name'].str.contains(r'p\.', regex=True, na=False).astype(int)
    df['is_frameshift'] = df['Name'].str.contains('fs', na=False).astype(int)
    df['is_snv'] = (df['Type'].str.lower() == 'single nucleotide variant').astype(int)

    df['nuc_x_review'] = df['nuc_position'] * df['review_score']
    df['aa_x_gene'] = df['aa_position'] * df['gene_enc']
    df['type_x_gene'] = df['type_enc'] * df['gene_enc']

    return df


def load_splits(features=None, scale=True):
    """
    Load the DEDUPLICATED csv and produce the 70/15/15 stratified split.

    Returns a dict. Splitting is done on the feature *frame* so that column
    subsets (the ablation) get exactly the same rows in each split.
    """
    features = list(FEATURES if features is None else features)

    df = pd.read_csv(DEDUP_CSV, low_memory=False)
    n_raw = len(df)
    assert df.duplicated(subset=["Name", "GeneSymbol"]).sum() == 0, \
        "input csv still contains duplicate variants - run scripts/dedup.py"

    df = parse_features(df)
    X = df[features].astype(float)
    y = df['Label'].values.astype(int)

    # 70 / 15 / 15, stratified, explicit seed
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15 / 0.85, random_state=SEED, stratify=y_tr
    )

    out = {
        "n_raw": n_raw, "features": features,
        "X_train_df": X_train, "X_val_df": X_val, "X_test_df": X_test,
        "X_train": X_train.values, "X_val": X_val.values, "X_test": X_test.values,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "seed": SEED,
    }

    if scale:
        scaler = StandardScaler()
        out["X_train_s"] = scaler.fit_transform(X_train.values)   # fitted on TRAIN only
        out["X_val_s"] = scaler.transform(X_val.values)
        out["X_test_s"] = scaler.transform(X_test.values)
        out["scaler"] = scaler

    return out


def split_report(d):
    lines = []
    for name in ("train", "val", "test"):
        y = d[f"y_{name}"]
        n = len(y)
        p = int((y == 1).sum())
        b = int((y == 0).sum())
        lines.append(f"  {name:<5} n={n:<6} pathogenic={p:<6} benign={b:<6} "
                     f"pathogenic={p / n * 100:.2f}%")
    total = len(d["y_train"]) + len(d["y_val"]) + len(d["y_test"])
    lines.append(f"  total n={total}")
    return "\n".join(lines)
