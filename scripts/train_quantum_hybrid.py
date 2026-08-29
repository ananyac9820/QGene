"""
Tasks 3, 4, 5 - quantum arm, Platt calibration, hybrid weight search,
and the raw test predictions that every downstream table is computed from.

The 500-sample QSVM subset is rebuilt from the DEDUPLICATED training split;
the old subset was drawn from a contaminated training set.
"""

import os, sys, time, json, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, load_splits, split_report

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import _SigmoidCalibration

from sklearn.metrics import accuracy_score, f1_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel

N_Q = 500
print("=" * 70)
print("QGene - quantum arm + Platt calibration + hybrid (deduplicated data)")
print("seed = %d" % SEED)
print("=" * 70)

d = load_splits()
print("\nsplits:")
print(split_report(d))
X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
X_train_s, X_val_s, X_test_s = d["X_train_s"], d["X_val_s"], d["X_test_s"]
y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]

# -- classical models (same config as scripts/train_classical.py) -----
print("\n[1/6] classical models...")
rf = RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_leaf=1,
                            class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True,
          class_weight='balanced', random_state=SEED)
svm.fit(X_train_s, y_train)
print("      RF  raw test acc %.2f%%" % (accuracy_score(y_test, rf.predict(X_test)) * 100))
print("      SVM raw test acc %.2f%%" % (accuracy_score(y_test, svm.predict(X_test_s)) * 100))

# -- quantum representation -------------------------------------------
print("\n[2/6] PCA(4) -> angles in [0, pi] (both fitted on TRAIN only)...")
pca = PCA(n_components=4, random_state=SEED)
Xtr_pca = pca.fit_transform(X_train_s)
ang = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr_pca)
Xtr_q = ang.transform(Xtr_pca)
Xva_q = ang.transform(pca.transform(X_val_s))
Xte_q = ang.transform(pca.transform(X_test_s))
print("      explained variance %s (sum %.4f)"
      % (pca.explained_variance_ratio_.round(4).tolist(), pca.explained_variance_ratio_.sum()))

idx_q, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=N_Q, random_state=SEED)
                .split(np.zeros(len(y_train)), y_train))
idx_q = np.sort(idx_q)
Xq, yq = Xtr_q[idx_q], y_train[idx_q]
print("      QSVM subset n=%d pathogenic=%d benign=%d"
      % (len(idx_q), int((yq == 1).sum()), int((yq == 0).sum())))

# -- quantum kernel ---------------------------------------------------
print("\n[3/6] quantum kernel (ZZFeatureMap 4q reps=2 full entanglement)...")
fmap = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='full')
qk = FidelityStatevectorKernel(feature_map=fmap)

_c = Xtr_q[:24]
_diff = np.abs(FidelityQuantumKernel(feature_map=fmap).evaluate(_c) - qk.evaluate(_c)).max()
print("      FidelityStatevectorKernel == FidelityQuantumKernel: max|diff| %.2e" % _diff)

t0 = time.perf_counter(); K_tr = qk.evaluate(Xq); t_tr = time.perf_counter() - t0
t0 = time.perf_counter(); K_va = qk.evaluate(Xva_q, Xq); t_va = time.perf_counter() - t0
t0 = time.perf_counter(); K_te = qk.evaluate(Xte_q, Xq); t_te = time.perf_counter() - t0
t_all = t_tr + t_va + t_te


def hhmm(s):
    return "%02d:%02d" % (int(s // 3600), int(s % 3600 // 60))


print("      K_train %s %.1fs | K_val %s %.1fs | K_test %s %.1fs"
      % (K_tr.shape, t_tr, K_va.shape, t_va, K_te.shape, t_te))
print("      total kernel wall-clock %.1fs = %s (hh:mm)" % (t_all, hhmm(t_all)))

# -- QSVM -------------------------------------------------------------
print("\n[4/6] QSVM (SVC precomputed)...")
qsvm = SVC(kernel='precomputed', class_weight='balanced', random_state=SEED)  # spec: default C=1
qsvm.fit(K_tr, yq)
q_acc_spec = accuracy_score(y_test, qsvm.predict(K_te))
print("      QSVM (C=1, paper config)  raw test acc %.2f%%" % (q_acc_spec * 100))
qsvm_c100 = SVC(kernel='precomputed', C=100, class_weight='balanced', random_state=SEED)
qsvm_c100.fit(K_tr, yq)
q_acc_c100 = accuracy_score(y_test, qsvm_c100.predict(K_te))
print("      QSVM (C=100, fig3 config) raw test acc %.2f%%  <- matches fig3 at n=500"
      % (q_acc_c100 * 100))

# -- Platt calibration on the VALIDATION split ------------------------
# _SigmoidCalibration is the exact sigmoid calibrator CalibratedClassifierCV
# uses internally (Platt scaling with Platt's prior correction on the targets).
# It is applied here directly to prefit scores, because CalibratedClassifierCV
# cannot cross-validate a non-square precomputed kernel matrix.
print("\n[5/6] Platt scaling all three models on the validation split...")


def platt(score_val, score_te, y_v):
    cal = _SigmoidCalibration().fit(score_val, y_v)
    return cal.predict(score_val), cal.predict(score_te), cal


rf_val_p, rf_te_p, rf_cal = platt(rf.predict_proba(X_val)[:, 1],
                                  rf.predict_proba(X_test)[:, 1], y_val)
svm_val_p, svm_te_p, svm_cal = platt(svm.decision_function(X_val_s),
                                     svm.decision_function(X_test_s), y_val)
q_val_p, q_te_p, q_cal = platt(qsvm.decision_function(K_va),
                               qsvm.decision_function(K_te), y_val)
for nm, c in (('rf', rf_cal), ('svm', svm_cal), ('qsvm', q_cal)):
    print("      %-5s Platt a=%+.4f b=%+.4f" % (nm, c.a_, c.b_))

# -- hybrid weight grid search on validation --------------------------
print("\n[6/6] hybrid weight grid search on validation...")
cl_val = (rf_val_p + svm_val_p) / 2
cl_te = (rf_te_p + svm_te_p) / 2
grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best = None
print("        w_C   w_Q   val acc   val F1")
for wc in grid:
    p = wc * cl_val + (1 - wc) * q_val_p
    pred = (p > 0.5).astype(int)
    a = accuracy_score(y_val, pred)
    f = f1_score(y_val, pred)
    print("      %5.1f %5.1f %8.2f%% %8.4f" % (wc, 1 - wc, a * 100, f))
    if best is None or a > best[2]:
        best = (wc, 1 - wc, a, f)
w_c, w_q, best_val_acc, best_val_f1 = best
print("      -> winning weights  w_C=%.1f  w_Q=%.1f  (val acc %.2f%%, val F1 %.4f)"
      % (w_c, w_q, best_val_acc * 100, best_val_f1))

hy_te_p = w_c * cl_te + w_q * q_te_p

# -- save raw predictions - everything downstream reads THIS -----------
os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
np.savez(os.path.join(BASE, 'results', 'test_preds.npz'),
         y_true=y_test,
         rf_pred=(rf_te_p > .5).astype(int), rf_prob=rf_te_p,
         svm_pred=(svm_te_p > .5).astype(int), svm_prob=svm_te_p,
         qsvm_pred=(q_te_p > .5).astype(int), qsvm_prob=q_te_p,
         hybrid_pred=(hy_te_p > .5).astype(int), hybrid_prob=hy_te_p)
print("\nwrote results/test_preds.npz")


def save(o, p):
    with open(os.path.join(BASE, p), 'wb') as f:
        pickle.dump(o, f)


save(qsvm, 'models/qsvm_dedup.pkl')
save(pca, 'models/pca_transformer_dedup.pkl')
save(ang, 'models/pca_angle_scaler_dedup.pkl')
save({'w_classical': w_c, 'w_quantum': w_q, 'val_accuracy': best_val_acc,
      'val_f1': best_val_f1, 'seed': SEED}, 'models/hybrid_config_dedup.pkl')
json.dump({'seed': SEED, 'n_qsvm': N_Q, 'w_classical': w_c, 'w_quantum': w_q,
           'val_accuracy': best_val_acc, 'val_f1': best_val_f1,
           'weight_grid': grid,
           'kernel_seconds': {'train': t_tr, 'val': t_va, 'test': t_te, 'total': t_all},
           'kernel_hhmm': hhmm(t_all),
           'qsvm_raw_acc_C1': q_acc_spec, 'qsvm_raw_acc_C100': q_acc_c100,
           'pca_explained_variance': pca.explained_variance_ratio_.tolist()},
          open(os.path.join(BASE, 'results', 'hybrid_config.json'), 'w'), indent=2)
print("wrote results/hybrid_config.json + models/*_dedup.pkl")
