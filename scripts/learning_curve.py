"""
Task 7 - QSVM vs matched-subset classical SVM learning curve.

This is a REAL experiment. gen_fig2.py is not reused: its curve was a
hardcoded literal, so there is nothing in it to reuse.

Design
------
For each n in [100..500]:
  * draw ONE stratified subset of size n from the deduplicated TRAINING split
  * train the QSVM on that subset            (quantum kernel, 4-D PCA angles)
  * train a classical RBF SVM on the SAME indices, two ways:
      - 4-D PCA angle features  -> isolates kernel choice (quantum vs RBF)
      - 14-D scaled features    -> matches the full-data baseline's inputs
  * evaluate all arms on the FULL 2,846-row test set

The index array is created once per n and shared by every arm; identity is
asserted, not assumed.
"""

import os, sys, time, json, hashlib, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, load_splits, split_report

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel

N_GRID = [100, 150, 200, 250, 300, 350, 400, 450, 500]

print("=" * 74)
print("QGene Task 7 - learning curve on DEDUPLICATED data")
print(f"seed = {SEED}")
print("=" * 74)

d = load_splits()
print("\nsplits:")
print(split_report(d))

X_train_s, X_test_s = d["X_train_s"], d["X_test_s"]
y_train, y_test = d["y_train"], d["y_test"]

# ── quantum representation: PCA(4) fitted on TRAIN only, angles in [0, pi] ──
pca = PCA(n_components=4, random_state=SEED)
Xtr_pca = pca.fit_transform(X_train_s)
Xte_pca = pca.transform(X_test_s)

ang = MinMaxScaler(feature_range=(0, np.pi))
Xtr_q = ang.fit_transform(Xtr_pca)
Xte_q = ang.transform(Xte_pca)
print(f"\nPCA(4) explained variance ratio : {pca.explained_variance_ratio_.round(4).tolist()}"
      f"  (sum {pca.explained_variance_ratio_.sum():.4f})")
print(f"train angle range {Xtr_q.min():.3f}..{Xtr_q.max():.3f} | "
      f"test angle range {Xte_q.min():.3f}..{Xte_q.max():.3f}")

feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='full')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)
print(f"feature map: ZZFeatureMap 4 qubits reps=2 full entanglement, "
      f"decomposed depth {feature_map.decompose().depth()}")

# one-off equivalence check against the spec-named FidelityQuantumKernel
_chk = Xtr_q[:24]
_k1 = FidelityQuantumKernel(feature_map=feature_map).evaluate(_chk)
_k2 = qkernel.evaluate(_chk)
print(f"FidelityStatevectorKernel vs FidelityQuantumKernel on 24x24: "
      f"max|diff| = {np.abs(_k1 - _k2).max():.2e}  (identical noiseless statevector fidelity)")

# ── full-data classical baselines ─────────────────────────────────────
print("\nfull-data classical baselines (all 13,279 training rows):")
base14 = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', random_state=SEED)
base14.fit(X_train_s, y_train)
b14_acc = accuracy_score(y_test, base14.predict(X_test_s))
b14_auc = roc_auc_score(y_test, base14.decision_function(X_test_s))
print(f"  14-D scaled : acc {b14_acc*100:.2f}%  AUC {b14_auc:.4f}")

base4 = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', random_state=SEED)
base4.fit(Xtr_q, y_train)
b4_acc = accuracy_score(y_test, base4.predict(Xte_q))
b4_auc = roc_auc_score(y_test, base4.decision_function(Xte_q))
print(f"  4-D PCA     : acc {b4_acc*100:.2f}%  AUC {b4_auc:.4f}")

# ── the curve ─────────────────────────────────────────────────────────
rows, kernel_seconds = [], 0.0
print("\n" + "-" * 74)
print(f"{'n':>5} {'QSVM acc':>9} {'QSVM AUC':>9} | {'SVM4 acc':>9} {'SVM4 AUC':>9} | "
      f"{'SVM14 acc':>10} {'SVM14 AUC':>10} | {'kernel s':>8}")
print("-" * 74)

for n in N_GRID:
    # ---- ONE index array, shared by every arm ----
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=SEED)
    idx, _ = next(sss.split(np.zeros(len(y_train)), y_train))
    idx = np.sort(idx)

    idx_q, idx_c4, idx_c14 = idx, idx, idx        # same object, but verify anyway
    assert np.array_equal(idx_q, idx_c4), f"n={n}: quantum/classical-4D indices differ"
    assert np.array_equal(idx_q, idx_c14), f"n={n}: quantum/classical-14D indices differ"
    assert len(np.unique(idx)) == n, f"n={n}: duplicate indices in subset"
    y_sub = y_train[idx]
    assert np.array_equal(y_train[idx_q], y_train[idx_c14]), f"n={n}: label vectors differ"
    idx_hash = hashlib.sha1(idx.tobytes()).hexdigest()[:12]

    Xq_sub = Xtr_q[idx]

    # ---- quantum arm ----
    t0 = time.perf_counter()
    K_tr = qkernel.evaluate(Xq_sub)
    K_te = qkernel.evaluate(Xte_q, Xq_sub)
    dt = time.perf_counter() - t0
    kernel_seconds += dt

    qsvm = SVC(kernel='precomputed', C=100, class_weight='balanced', random_state=SEED)
    qsvm.fit(K_tr, y_sub)
    q_acc = accuracy_score(y_test, qsvm.predict(K_te))
    q_auc = roc_auc_score(y_test, qsvm.decision_function(K_te))

    # ---- classical arm, identical indices, 4-D PCA angles ----
    c4 = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', random_state=SEED)
    c4.fit(Xtr_q[idx_c4], y_train[idx_c4])
    c4_acc = accuracy_score(y_test, c4.predict(Xte_q))
    c4_auc = roc_auc_score(y_test, c4.decision_function(Xte_q))

    # ---- classical arm, identical indices, 14-D scaled ----
    c14 = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', random_state=SEED)
    c14.fit(X_train_s[idx_c14], y_train[idx_c14])
    c14_acc = accuracy_score(y_test, c14.predict(X_test_s))
    c14_auc = roc_auc_score(y_test, c14.decision_function(X_test_s))

    rows.append(dict(n=n, idx_sha1=idx_hash, n_path=int((y_sub == 1).sum()),
                     n_ben=int((y_sub == 0).sum()),
                     qsvm_acc=q_acc, qsvm_auc=q_auc,
                     svm4_acc=c4_acc, svm4_auc=c4_auc,
                     svm14_acc=c14_acc, svm14_auc=c14_auc,
                     kernel_s=dt))
    print(f"{n:5d} {q_acc*100:8.2f}% {q_auc:9.4f} | {c4_acc*100:8.2f}% {c4_auc:9.4f} | "
          f"{c14_acc*100:9.2f}% {c14_auc:10.4f} | {dt:8.1f}")

print("-" * 74)
print(f"total quantum-kernel wall-clock: {kernel_seconds:.1f}s "
      f"= {int(kernel_seconds//3600):02d}:{int(kernel_seconds%3600//60):02d} (hh:mm)")

out = dict(seed=SEED, n_grid=N_GRID, rows=rows,
           baseline_14d=dict(acc=b14_acc, auc=b14_auc, n_train=int(len(y_train))),
           baseline_4d=dict(acc=b4_acc, auc=b4_auc, n_train=int(len(y_train))),
           kernel_seconds_total=kernel_seconds,
           test_n=int(len(y_test)))
os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
with open(os.path.join(BASE, 'results', 'learning_curve.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("wrote results/learning_curve.json")
