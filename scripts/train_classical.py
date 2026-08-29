"""
Task 2 - rerun the classical pipeline on the DEDUPLICATED data.

Hyperparameters are identical to retrain_models.py:
  RF  : 500 trees, gini, min_samples_leaf=1, class_weight='balanced'
  SVM : RBF, C=100, gamma='scale', class_weight='balanced', probability=True
  StandardScaler fitted on train only; 70/15/15 stratified.
New .pkl files are written alongside the originals with a _dedup suffix.
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, load_splits, split_report, FEATURES

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("=" * 62)
print("QGene - classical retrain on deduplicated data")
print(f"random_state / seed = {SEED} (used everywhere)")
print("=" * 62)

d = load_splits()
print(f"\ndeduplicated rows loaded: {d['n_raw']}")
print(f"features ({len(FEATURES)}): {FEATURES}")
print("\nsplit sizes and class balance:")
print(split_report(d))

X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
X_train_s, X_val_s, X_test_s = d["X_train_s"], d["X_val_s"], d["X_test_s"]
y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]

# ── Random Forest ─────────────────────────────────────────────────────
print("\n[1/2] Random Forest (500 trees, gini, min_samples_leaf=1, balanced)...")
rf = RandomForestClassifier(
    n_estimators=500,
    criterion='gini',
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
print(f"      RF  test  -> acc {rf_acc * 100:.2f}%  F1 {f1_score(y_test, rf_pred):.4f}  "
      f"AUC {roc_auc_score(y_test, rf_prob):.4f}")
print(f"      RF  train -> acc {accuracy_score(y_train, rf.predict(X_train)) * 100:.2f}% "
      f"(train fit, for leakage context)")

# ── SVM ───────────────────────────────────────────────────────────────
print("\n[2/2] SVM (RBF, C=100, gamma=scale, balanced, probability=True)...")
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True,
          class_weight='balanced', random_state=SEED)
svm.fit(X_train_s, y_train)

svm_pred = svm.predict(X_test_s)
svm_prob = svm.predict_proba(X_test_s)[:, 1]
svm_acc = accuracy_score(y_test, svm_pred)
print(f"      SVM test  -> acc {svm_acc * 100:.2f}%  F1 {f1_score(y_test, svm_pred):.4f}  "
      f"AUC {roc_auc_score(y_test, svm_prob):.4f}")

# ── save alongside, never in place ────────────────────────────────────
def save(obj, path):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'wb') as f:
        pickle.dump(obj, f)
    print(f"      saved {path}")

print("\nsaving new artifacts (originals untouched):")
save(rf, 'models/random_forest_dedup.pkl')
save(svm, 'models/svm_dedup.pkl')
save(d["scaler"], 'models/feature_scaler_dedup.pkl')
save(FEATURES, 'models/feature_names_dedup.pkl')

print("\n" + "=" * 62)
print(f"RF  accuracy on deduplicated test set : {rf_acc * 100:.2f}%")
print(f"SVM accuracy on deduplicated test set : {svm_acc * 100:.2f}%")
print("=" * 62)
