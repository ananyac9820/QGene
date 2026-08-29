"""
Tasks 5 and 6 - Table 4, Table 5, consistency self-checks, and McNemar.

Everything here is computed FROM results/test_preds.npz. No summary metric
from any earlier script is read or trusted.
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2 as chi2_dist

Z = np.load(os.path.join(BASE, 'results', 'test_preds.npz'))
y = Z['y_true']
MODELS = ['rf', 'svm', 'qsvm', 'hybrid']
LABEL = {'rf': 'Random Forest', 'svm': 'SVM (RBF)', 'qsvm': 'QSVM', 'hybrid': 'Hybrid'}

N = len(y)
P = int((y == 1).sum())
B = int((y == 0).sum())
print("=" * 78)
print("Computed from results/test_preds.npz")
print("test set n=%d | pathogenic=%d | benign=%d" % (N, P, B))
print("=" * 78)

# ---------------- Table 4 ----------------
print("\nTABLE 4 - test-set performance")
print("-" * 78)
print("%-14s %9s %10s %8s %8s %9s" % ("Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"))
print("-" * 78)
t4 = {}
for m in MODELS:
    p, pr = Z[m + '_pred'], Z[m + '_prob']
    t4[m] = dict(acc=accuracy_score(y, p), prec=precision_score(y, p),
                 rec=recall_score(y, p), f1=f1_score(y, p), auc=roc_auc_score(y, pr))
    r = t4[m]
    print("%-14s %8.2f%% %10.3f %8.3f %8.3f %9.3f"
          % (LABEL[m], r['acc'] * 100, r['prec'], r['rec'], r['f1'], r['auc']))

# ---------------- Table 5 ----------------
print("\nTABLE 5 - confusion counts and clinical miss rate")
print("-" * 78)
print("%-14s %6s %6s %5s %6s %10s %18s"
      % ("Model", "TP", "TN", "FP", "FN", "FN % test", "missed pathogenic %"))
print("-" * 78)
t5 = {}
for m in MODELS:
    tn, fp, fn, tp = confusion_matrix(y, Z[m + '_pred'], labels=[0, 1]).ravel()
    t5[m] = dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                 fn_pct=fn / N * 100, missed=(1 - t4[m]['rec']) * 100)
    r = t5[m]
    print("%-14s %6d %6d %5d %6d %9.2f%% %17.2f%%"
          % (LABEL[m], r['tp'], r['tn'], r['fp'], r['fn'], r['fn_pct'], r['missed']))

# ---------------- consistency self-checks ----------------
print("\nCONSISTENCY SELF-CHECKS")
print("-" * 78)
checks, allok = [], True
for m in MODELS:
    c, r = t5[m], t4[m]
    tp, tn, fp, fn = c['tp'], c['tn'], c['fp'], c['fn']
    acc_c = (tp + tn) / (tp + tn + fp + fn)
    prec_c = tp / (tp + fp) if tp + fp else 0.0
    rec_c = tp / (tp + fn) if tp + fn else 0.0
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if prec_c + rec_c else 0.0
    res = [
        ("TP+TN+FP+FN == n", tp + tn + fp + fn == N),
        ("TP+FN == pathogenic", tp + fn == P),
        ("TN+FP == benign", tn + fp == B),
        ("accuracy from counts (2dp)", round(acc_c * 100, 2) == round(r['acc'] * 100, 2)),
        ("precision from counts (3dp)", round(prec_c, 3) == round(r['prec'], 3)),
        ("recall from counts (3dp)", round(rec_c, 3) == round(r['rec'], 3)),
        ("F1 from counts (3dp)", round(f1_c, 3) == round(r['f1'], 3)),
    ]
    ok = all(v for _, v in res)
    allok &= ok
    checks.append((m, res, ok))
    print("%-14s %s" % (LABEL[m], "PASS - all 7 checks" if ok else "FAIL"))
    for nm, v in res:
        if not v:
            print("      FAIL: %s" % nm)
print("-" * 78)
print("OVERALL: %s" % ("ALL CHECKS PASS" if allok else "MISMATCH PRESENT"))

# ---------------- McNemar, RF vs QSVM ----------------
print("\nMcNEMAR - RF vs QSVM (from the saved prediction arrays)")
print("-" * 78)
rf_ok = (Z['rf_pred'] == y)
q_ok = (Z['qsvm_pred'] == y)
both_right = int((rf_ok & q_ok).sum())
b = int((~rf_ok & q_ok).sum())     # RF wrong, QSVM right
c = int((rf_ok & ~q_ok).sum())     # RF right, QSVM wrong
both_wrong = int((~rf_ok & ~q_ok).sum())

print("convention: b = RF wrong & QSVM right ; c = RF right & QSVM wrong")
print("  both right : %d" % both_right)
print("  b          : %d" % b)
print("  c          : %d" % c)
print("  both wrong : %d" % both_wrong)
print("  total      : %d  (== n: %s)" % (both_right + b + c + both_wrong,
                                         both_right + b + c + both_wrong == N))

rf_err, q_err = int((~rf_ok).sum()), int((~q_ok).sum())
print("\nsanity checks:")
print("  b + both_wrong = %d + %d = %d   RF total errors   = %d   -> %s"
      % (b, both_wrong, b + both_wrong, rf_err, "PASS" if b + both_wrong == rf_err else "FAIL"))
print("  c + both_wrong = %d + %d = %d   QSVM total errors = %d   -> %s"
      % (c, both_wrong, c + both_wrong, q_err, "PASS" if c + both_wrong == q_err else "FAIL"))

table = [[both_right, c], [b, both_wrong]]
res = mcnemar(table, exact=False, correction=True)
manual = (abs(b - c) - 1) ** 2 / (b + c)
print("\nstatsmodels mcnemar(exact=False, correction=True):")
print("  chi-square = %.4f" % res.statistic)
print("  p-value    = %.6g" % res.pvalue)
print("\nhand-computed, so the arithmetic is verifiable:")
print("  chi2 = (|b-c|-1)^2 / (b+c) = (|%d-%d|-1)^2 / %d = %d / %d = %.4f"
      % (b, c, b + c, (abs(b - c) - 1) ** 2, b + c, manual))
print("  p    = P(chi2_1 > %.4f) = %.6g" % (manual, chi2_dist.sf(manual, 1)))
print("  matches statsmodels: %s" % np.isclose(manual, res.statistic))

json.dump({'n': N, 'pathogenic': P, 'benign': B,
           'table4': {m: t4[m] for m in MODELS},
           'table5': {m: t5[m] for m in MODELS},
           'all_checks_pass': bool(allok),
           'mcnemar': {'convention': 'b = RF wrong & QSVM right; c = RF right & QSVM wrong',
                       'both_right': both_right, 'b': b, 'c': c, 'both_wrong': both_wrong,
                       'chi2': float(res.statistic), 'p': float(res.pvalue),
                       'rf_errors': rf_err, 'qsvm_errors': q_err}},
          open(os.path.join(BASE, 'results', 'tables.json'), 'w'), indent=2)
print("\nwrote results/tables.json")
