"""
QGene - Model Retraining Script
Run this from your QGene project folder:
    python retrain_models.py
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("QGene Model Retraining — Enhanced Features")
print("=" * 60)

# ── 1. Load Data ────────────────────────────────────────────────
print("\n[1/5] Loading data...")
df = pd.read_csv(os.path.join(BASE, 'data/brca_mutations.csv'))
print(f"      Total variants: {len(df)}")

# ── 2. Feature Engineering ──────────────────────────────────────
print("\n[2/5] Engineering features...")

def parse_features(df):
    df = df.copy()

    # Gene
    df['gene_enc'] = df['GeneSymbol'].map({'BRCA1': 0, 'BRCA2': 1}).fillna(0)

    # Mutation type
    type_map = {
        'single nucleotide variant': 0, 'deletion': 1, 'insertion': 2,
        'duplication': 3, 'indel': 4, 'microsatellite': 5, 'inversion': 6,
        'protein only': 0, 'copy number loss': 1, 'copy number gain': 3,
        'complex': 4, 'variation': 0
    }
    df['type_enc'] = df['Type'].str.lower().map(type_map).fillna(0)

    # Position and length
    df['position']   = df['Start'].fillna(0)
    df['mut_length'] = (df['Stop'] - df['Start']).fillna(1).clip(1, 10000)
    df['log_length'] = np.log1p(df['mut_length'])

    # Review confidence
    review_map = {
        'reviewed by expert panel': 4,
        'criteria provided, multiple submitters, no conflicts': 3,
        'criteria provided, single submitter': 2,
        'no assertion criteria provided': 1,
        'no assertion provided': 0
    }
    df['review_score'] = df['ReviewStatus'].map(review_map).fillna(1)
    df['is_expert']    = (df['ReviewStatus'] == 'reviewed by expert panel').astype(int)

    # Parse nucleotide position from Name (c.XXXX)
    def get_nuc(x):
        m = re.search(r'c\.[-*]?(\d+)', str(x))
        return int(m.group(1)) if m else 0

    # Parse amino acid position from Name (p.XXXNNN)
    def get_aa(x):
        m = re.search(r'p\.[A-Za-z]+(\d+)', str(x))
        return int(m.group(1)) if m else 0

    df['nuc_position']       = df['Name'].apply(get_nuc)
    df['aa_position']        = df['Name'].apply(get_aa)
    df['has_protein_change'] = df['Name'].str.contains(r'p\.', regex=True, na=False).astype(int)
    df['is_frameshift']      = df['Name'].str.contains('fs', na=False).astype(int)
    df['is_snv']             = (df['Type'].str.lower() == 'single nucleotide variant').astype(int)

    # Interaction features
    df['nuc_x_review'] = df['nuc_position'] * df['review_score']
    df['aa_x_gene']    = df['aa_position']  * df['gene_enc']
    df['type_x_gene']  = df['type_enc']     * df['gene_enc']

    return df

df = parse_features(df)
print("      Features engineered.")

FEATURES = [
    'gene_enc', 'type_enc', 'position', 'log_length', 'review_score',
    'nuc_position', 'aa_position', 'has_protein_change', 'is_frameshift',
    'is_snv', 'is_expert', 'nuc_x_review', 'aa_x_gene', 'type_x_gene'
]

X = df[FEATURES].values
y = df['Label'].values

# ── 3. Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_val_s   = scaler.transform(X_val)

print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 4. Train Classical Models ────────────────────────────────────
print("\n[3/5] Training classical models...")

# Random Forest
print("      Training Random Forest (500 trees)...")
rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_f1  = f1_score(y_test, rf.predict(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"      RF  → Accuracy: {rf_acc*100:.2f}%  F1: {rf_f1:.4f}  AUC: {rf_auc:.4f}")

# SVM
print("      Training SVM (RBF, C=100)...")
svm = SVC(
    kernel='rbf', C=100, gamma='scale',
    probability=True, class_weight='balanced', random_state=42
)
svm.fit(X_train_s, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_s))
svm_f1  = f1_score(y_test, svm.predict(X_test_s))
svm_auc = roc_auc_score(y_test, svm.predict_proba(X_test_s)[:, 1])
print(f"      SVM → Accuracy: {svm_acc*100:.2f}%  F1: {svm_f1:.4f}  AUC: {svm_auc:.4f}")

# ── 5. Train QSVM ────────────────────────────────────────────────
print("\n[4/5] Training QSVM...")

# PCA to 4 dims for quantum
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

# Scale PCA output to [-pi, pi] for ZZFeatureMap
pca_scaler     = StandardScaler()
X_train_q      = pca_scaler.fit_transform(X_train_pca)
X_test_q       = pca_scaler.transform(X_test_pca)

# Use 500 samples for QSVM (quantum compute constraint)
np.random.seed(42)
idx = np.random.choice(len(X_train_q), size=500, replace=False)
X_qtrain = X_train_q[idx]
y_qtrain = y_train[idx]

print("      Building quantum kernel (ZZFeatureMap, 4 qubits, 2 reps)...")
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
    qkernel     = FidelityQuantumKernel(feature_map=feature_map)

    print("      Computing kernel matrix (500×500)... this may take a few minutes")
    K_train = qkernel.evaluate(X_qtrain)

    qsvm = SVC(kernel='precomputed', probability=True, class_weight='balanced', random_state=42)
    qsvm.fit(K_train, y_qtrain)

    print("      Computing test kernel matrix (500-test samples)...")
    # Use 500 test samples for evaluation
    idx_test = np.random.choice(len(X_test_q), size=min(500, len(X_test_q)), replace=False)
    X_qtest  = X_test_q[idx_test]
    y_qtest  = y_test[idx_test]

    K_test   = qkernel.evaluate(X_qtest, X_qtrain)
    qsvm_acc = accuracy_score(y_qtest, qsvm.predict(K_test))
    qsvm_f1  = f1_score(y_qtest, qsvm.predict(K_test))
    qsvm_auc = roc_auc_score(y_qtest, qsvm.predict_proba(K_test)[:, 1])
    print(f"      QSVM→ Accuracy: {qsvm_acc*100:.2f}%  F1: {qsvm_f1:.4f}  AUC: {qsvm_auc:.4f}")
    quantum_available = True

except Exception as e:
    print(f"      WARNING: Quantum training failed ({e})")
    print("      Falling back to classical SVM as QSVM substitute...")
    qsvm             = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                           class_weight='balanced', random_state=42)
    qsvm.fit(X_train_q[:500], y_train[:500])
    qsvm_acc         = accuracy_score(y_test, qsvm.predict(X_test_q))
    qsvm_f1          = f1_score(y_test, qsvm.predict(X_test_q))
    qsvm_auc         = roc_auc_score(y_test, qsvm.predict_proba(X_test_q)[:, 1])
    print(f"      QSVM→ Accuracy: {qsvm_acc*100:.2f}%  F1: {qsvm_f1:.4f}  AUC: {qsvm_auc:.4f}")
    quantum_available = False
    X_qtrain          = X_train_q[:500]

# ── 6. Hybrid Ensemble ───────────────────────────────────────────
print("\n[5/5] Finding optimal hybrid weights...")

best_acc, best_wc, best_wq = 0, 0.5, 0.5

rf_val_prob  = rf.predict_proba(X_val)[:, 1]
svm_val_prob = svm.predict_proba(X_val_s)[:, 1]
cl_val       = (rf_val_prob + svm_val_prob) / 2

if quantum_available:
    K_val       = qkernel.evaluate(pca_scaler.transform(pca.transform(X_val_s)), X_qtrain)
    q_val_prob  = qsvm.predict_proba(K_val)[:, 1]
else:
    q_val_prob  = qsvm.predict_proba(pca_scaler.transform(pca.transform(X_val_s)))[:, 1]

for wc in np.arange(0.3, 1.0, 0.1):
    wq   = 1 - wc
    pred = (wc * cl_val + wq * q_val_prob) > 0.5
    acc  = accuracy_score(y_val, pred)
    if acc > best_acc:
        best_acc, best_wc, best_wq = acc, wc, wq

print(f"      Best weights → Classical: {best_wc:.1f} / Quantum: {best_wq:.1f}")

# Final hybrid on test set
rf_test_prob  = rf.predict_proba(X_test)[:, 1]
svm_test_prob = svm.predict_proba(X_test_s)[:, 1]
cl_test       = (rf_test_prob + svm_test_prob) / 2

if quantum_available:
    K_test_full  = qkernel.evaluate(X_test_q, X_qtrain)
    q_test_prob  = qsvm.predict_proba(K_test_full)[:, 1]
else:
    q_test_prob  = qsvm.predict_proba(X_test_q)[:, 1]

hybrid_prob  = best_wc * cl_test + best_wq * q_test_prob
hybrid_pred  = (hybrid_prob > 0.5).astype(int)
hybrid_acc   = accuracy_score(y_test, hybrid_pred)
hybrid_f1    = f1_score(y_test, hybrid_pred)
hybrid_auc   = roc_auc_score(y_test, hybrid_prob)

print(f"      Hybrid → Accuracy: {hybrid_acc*100:.2f}%  F1: {hybrid_f1:.4f}  AUC: {hybrid_auc:.4f}")

# ── 7. Save Models ───────────────────────────────────────────────
print("\nSaving models...")

def save(obj, path):
    with open(os.path.join(BASE, path), 'wb') as f:
        pickle.dump(obj, f)

save(rf,    'models/random_forest_model.pkl')
save(svm,   'models/svm_model.pkl')
save(qsvm,  'models/qsvm_upgraded.pkl')
save(pca,   'models/pca_transformer.pkl')
save(scaler,'models/feature_scaler.pkl')
save(pca_scaler, 'models/pca_scaler.pkl')
save({'w_classical': best_wc, 'w_quantum': best_wq,
      'accuracy': hybrid_acc, 'f1': hybrid_f1, 'roc_auc': hybrid_auc},
     'models/hybrid_config.pkl')

np.save(os.path.join(BASE, 'data/X_qsvm_test.npy'), X_qtrain)
save(FEATURES, 'models/feature_names.pkl')

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Random Forest → {rf_acc*100:.2f}%")
print(f"SVM           → {svm_acc*100:.2f}%")
print(f"QSVM          → {qsvm_acc*100:.2f}%")
print(f"Hybrid        → {hybrid_acc*100:.2f}%")
print("=" * 60)
print("All models saved. Update qgene_app.py feature extraction next.")
print("=" * 60)
