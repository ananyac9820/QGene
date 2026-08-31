"""
Regenerate the three stale paper figures from results/test_preds.npz.

  figures/fig2_performance.png  grouped bars, 4 models x 5 metrics, ~3:1
  figures/fig4_roc.png          ROC curves, 4 models, AUCs in legend
  figures/fig5_shap.png         mean |SHAP| bar, all 14 features, descending

Every metric is recomputed from the stored arrays. Each figure is verified
against Table 4 BEFORE savefig is called - a mismatch raises and nothing is
written. No VQC anywhere: it is not in test_preds.npz and is not plotted.
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, load_splits

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

mpl.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

MODELS = ['rf', 'svm', 'qsvm', 'hybrid']
LABEL = {'rf': 'Random Forest', 'svm': 'SVM (RBF)', 'qsvm': 'QSVM', 'hybrid': 'Hybrid'}
COL = {'rf': '#2E7D32', 'svm': '#2E5BA8', 'qsvm': '#7B2D8B', 'hybrid': '#B85000'}

# Table 4 as it must appear in the paper. Verification target, not a data source.
EXPECTED = {
    'rf':     dict(acc=87.67, prec=0.883, rec=0.880, f1=0.881, auc=0.946),
    'svm':    dict(acc=88.79, prec=0.993, rec=0.790, f1=0.880, auc=0.954),
    'qsvm':   dict(acc=85.31, prec=0.890, rec=0.818, f1=0.853, auc=0.911),
    'hybrid': dict(acc=88.69, prec=0.963, rec=0.813, f1=0.882, auc=0.948),
}

Z = np.load(os.path.join(BASE, 'results', 'test_preds.npz'))
y = Z['y_true']
print("=" * 74)
print("source: results/test_preds.npz  |  test n=%d  pathogenic=%d  benign=%d"
      % (len(y), int((y == 1).sum()), int((y == 0).sum())))
print("=" * 74)

# ---- recompute Table 4 from the stored arrays --------------------------
T4 = {}
for m in MODELS:
    p, pr = Z[m + '_pred'], Z[m + '_prob']
    T4[m] = dict(acc=accuracy_score(y, p) * 100, prec=precision_score(y, p),
                 rec=recall_score(y, p), f1=f1_score(y, p), auc=roc_auc_score(y, pr))


def verify(tag):
    """Abort before writing anything if the arrays disagree with Table 4."""
    print("\n[verify] %s against Table 4" % tag)
    bad = []
    for m in MODELS:
        for k, dp in (('acc', 2), ('prec', 3), ('rec', 3), ('f1', 3), ('auc', 3)):
            got, exp = round(T4[m][k], dp), EXPECTED[m][k]
            if got != exp:
                bad.append("%s.%s computed %s != Table 4 %s" % (m, k, got, exp))
    for m in MODELS:
        e = EXPECTED[m]
        print("  %-14s %6.2f%% %6.3f %6.3f %6.3f %6.3f   OK"
              % (LABEL[m], e['acc'], e['prec'], e['rec'], e['f1'], e['auc']))
    if bad:
        raise SystemExit("VERIFICATION FAILED - nothing written:\n  " + "\n  ".join(bad))
    print("  all 20 cells match Table 4 -> safe to save")


assert 'vqc_pred' not in Z.files and 'vqc_prob' not in Z.files, \
    "VQC arrays present in npz - they must not be plotted"
print("\nVQC check: no vqc_* arrays in test_preds.npz -> VQC cannot be plotted")

# ======================= fig2: grouped bars ============================
verify("fig2_performance")
METRICS = [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'),
           ('f1', 'F1'), ('auc', 'ROC-AUC')]

fig, ax = plt.subplots(figsize=(15, 5))          # 3:1
fig.patch.set_facecolor('white')
w, xs = 0.2, np.arange(len(METRICS))
for i, m in enumerate(MODELS):
    # accuracy is stored as a percentage; plot every metric on a 0-1 scale
    vals = [T4[m][k] / 100 if k == 'acc' else T4[m][k] for k, _ in METRICS]
    bars = ax.bar(xs + (i - 1.5) * w, vals, w, label=LABEL[m],
                  color=COL[m], edgecolor='white', linewidth=1.0)
    for bar, v, (k, _) in zip(bars, vals, METRICS):
        txt = "%.2f%%" % T4[m]['acc'] if k == 'acc' else "%.3f" % v
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, txt,
                ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(xs)
ax.set_xticklabels([n for _, n in METRICS], fontsize=13)
ax.set_ylim(0, 1.14)
ax.set_ylabel('Score', fontsize=12)
ax.tick_params(axis='y', labelsize=10)
ax.set_title('QGene test-set performance, deduplicated BRCA1/2 variants '
             '(n = %s, seed = %d)' % (format(len(y), ','), SEED), fontsize=13, pad=12)
ax.legend(fontsize=11, ncol=4, loc='upper center', framealpha=0.96)
ax.grid(True, axis='y', ls='--', alpha=0.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
p2 = os.path.join(FIG, 'fig2_performance.png')
plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
im = plt.imread(p2)
print("saved fig2_performance.png  %dx%d px  aspect %.2f:1  dpi=300"
      % (im.shape[1], im.shape[0], im.shape[1] / im.shape[0]))

# ======================= fig4: ROC curves ==============================
verify("fig4_roc")
fig, ax = plt.subplots(figsize=(7.0, 6.2))
fig.patch.set_facecolor('white')
legend_aucs = {}
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, Z[m + '_prob'])
    a = roc_auc_score(y, Z[m + '_prob'])
    legend_aucs[m] = round(a, 3)
    ax.plot(fpr, tpr, color=COL[m], lw=2.2, ls='--' if m == 'qsvm' else '-',
            label='%s (AUC = %.3f)' % (LABEL[m], a))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1, label='Random classifier')

for m in MODELS:
    assert legend_aucs[m] == EXPECTED[m]['auc'], \
        "legend AUC %s for %s != Table 4 %s" % (legend_aucs[m], m, EXPECTED[m]['auc'])
print("  legend AUCs %s -> match Table 4"
      % ", ".join("%s %.3f" % (LABEL[m], legend_aucs[m]) for m in MODELS))

ax.set_xlabel('False positive rate', fontsize=11)
ax.set_ylabel('True positive rate', fontsize=11)
ax.set_title('ROC curves on the deduplicated test set\n(n = %s, seed = %d)'
             % (format(len(y), ','), SEED), fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.grid(True, ls='--', alpha=0.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
p4 = os.path.join(FIG, 'fig4_roc.png')
plt.savefig(p4, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("saved fig4_roc.png  dpi=300  (4 curves, no VQC)")

# ======================= fig5: SHAP ====================================
print("\n[fig5] retraining RF and running TreeExplainer...")
import shap
d = load_splits()
rf = RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_leaf=1,
                            class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(d["X_train"], d["y_train"])

# sanity: this RF must be the one behind Table 4's RF row
rf_acc = accuracy_score(d["y_test"], rf.predict(d["X_test"])) * 100
assert round(rf_acc, 2) == EXPECTED['rf']['acc'], \
    "SHAP RF accuracy %.2f != Table 4 RF %.2f" % (rf_acc, EXPECTED['rf']['acc'])
print("  RF for SHAP reproduces Table 4 RF accuracy: %.2f%%" % rf_acc)

X_df = d["X_test_df"]                       # pandas DataFrame, all 14 features
assert hasattr(X_df, 'columns') and X_df.shape[1] == 14, "SHAP input must be a 14-col DataFrame"
print("  SHAP input: %s %s, columns=%d" % (type(X_df).__name__, X_df.shape, X_df.shape[1]))

sv = np.asarray(shap.TreeExplainer(rf).shap_values(X_df))
if sv.ndim == 3:                            # (n, features, classes) -> positive class
    sv = sv[:, :, 1]
mean_abs = np.abs(sv).mean(axis=0)
assert len(mean_abs) == 14, "expected 14 SHAP values, got %d" % len(mean_abs)

order_desc = np.argsort(mean_abs)[::-1]     # largest first
names_desc = [X_df.columns[i] for i in order_desc]
vals_desc = mean_abs[order_desc]

fig, ax = plt.subplots(figsize=(8.2, 6.4))
fig.patch.set_facecolor('white')
ypos = np.arange(14)[::-1]                  # largest at the top
ax.barh(ypos, vals_desc, color='#2E5BA8', height=0.74)
ax.set_yticks(ypos)
ax.set_yticklabels(names_desc, fontsize=10)
for yy, v in zip(ypos, vals_desc):
    ax.text(v + vals_desc.max() * 0.013, yy, "%.4f" % v, va='center', fontsize=8.5)
ax.set_xlabel('mean(|SHAP value|)   -   mean impact on model output magnitude', fontsize=10.5)
ax.set_xlim(0, vals_desc.max() * 1.18)
ax.set_title('Random Forest feature importance (SHAP, TreeExplainer)\n'
             'all 14 features, %s test rows, seed = %d'
             % (format(len(X_df), ','), SEED), fontsize=12)
ax.grid(True, axis='x', ls='--', alpha=0.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
p5 = os.path.join(FIG, 'fig5_shap.png')
plt.savefig(p5, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("saved fig5_shap.png  dpi=300  (horizontal bars, 14 features, descending)")

print("\n" + "=" * 74)
print("RANKED FEATURE LIST for section 5.4  (mean |SHAP|, descending)")
print("=" * 74)
print("%-5s %-22s %-10s %-9s" % ("rank", "feature", "mean|SHAP|", "% of total"))
print("-" * 74)
tot = vals_desc.sum()
for r, (nm, v) in enumerate(zip(names_desc, vals_desc), 1):
    print("%-5d %-22s %-10.5f %8.2f%%" % (r, nm, v, v / tot * 100))
print("-" * 74)
print("%-28s %-10.5f %8.2f%%" % ("TOTAL", tot, 100.0))
top3 = vals_desc[:3].sum() / tot * 100
print("\ntop 3 features account for %.2f%% of total mean|SHAP|" % top3)
print("top 5 features account for %.2f%% of total mean|SHAP|" % (vals_desc[:5].sum() / tot * 100))

json.dump({'ranked': [{'rank': r, 'feature': nm, 'mean_abs_shap': float(v),
                       'pct_of_total': float(v / tot * 100)}
                      for r, (nm, v) in enumerate(zip(names_desc, vals_desc), 1)],
           'n_rows': int(len(X_df)), 'seed': SEED},
          open(os.path.join(BASE, 'results', 'shap_ranked.json'), 'w'), indent=2)
print("\nwrote results/shap_ranked.json")
