"""
Task 9 - regenerate figures at dpi=300.

fig1 architecture : SKIPPED - no source exists in the repo (hand-drawn).
fig2 grouped metric bars   <- results/test_preds.npz  (via results/tables.json)
fig3 learning curves       <- scripts/fig3_learning_curves.py (separate)
fig4 ROC curves            <- results/test_preds.npz
fig5 SHAP mean-|value| bar <- retrained RF, features passed as a DataFrame
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, FEATURES, load_splits

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

mpl.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

Z = np.load(os.path.join(BASE, 'results', 'test_preds.npz'))
T = json.load(open(os.path.join(BASE, 'results', 'tables.json')))
y = Z['y_true']

MODELS = ['rf', 'svm', 'qsvm', 'hybrid']
LABEL = {'rf': 'Random Forest', 'svm': 'SVM (RBF)', 'qsvm': 'QSVM', 'hybrid': 'Hybrid'}
COL = {'rf': '#2E7D32', 'svm': '#2E5BA8', 'qsvm': '#7B2D8B', 'hybrid': '#B85000'}

print("fig1 architecture : SKIPPED (no source in repo - hand-drawn)")

# ---------------- fig2: grouped metric bars ----------------
METRICS = [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'),
           ('f1', 'F1'), ('auc', 'ROC-AUC')]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
fig.patch.set_facecolor('white')
w, xs = 0.2, np.arange(len(METRICS))
for i, m in enumerate(MODELS):
    vals = [T['table4'][m][k] for k, _ in METRICS]
    bars = ax.bar(xs + (i - 1.5) * w, vals, w, label=LABEL[m],
                  color=COL[m], edgecolor='white', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, "%.3f" % v,
                ha='center', va='bottom', fontsize=6.4, rotation=90)
ax.set_xticks(xs)
ax.set_xticklabels([n for _, n in METRICS], fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=10)
ax.set_title('QGene test-set performance on deduplicated BRCA1/2 variants\n'
             '(n=%d; seed=%d)' % (T['n'], SEED), fontsize=11)
ax.legend(fontsize=8.5, ncol=4, loc='upper center', framealpha=0.95)
ax.grid(True, axis='y', ls='--', alpha=0.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig2_metric_bars.png'), dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("fig2 saved -> figures/fig2_metric_bars.png (dpi=300)")

# ---------------- fig4: ROC curves ----------------
fig, ax = plt.subplots(figsize=(6.4, 5.8))
fig.patch.set_facecolor('white')
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, Z[m + '_prob'])
    ax.plot(fpr, tpr, color=COL[m], lw=2.0,
            ls='--' if m == 'qsvm' else '-',
            label='%s (AUC = %.4f)' % (LABEL[m], roc_auc_score(y, Z[m + '_prob'])))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1, label='Random classifier')
ax.set_xlabel('False positive rate', fontsize=10)
ax.set_ylabel('True positive rate', fontsize=10)
ax.set_title('ROC curves on the deduplicated test set\n(n=%d; seed=%d)' % (T['n'], SEED),
             fontsize=11)
ax.legend(loc='lower right', fontsize=8.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.grid(True, ls='--', alpha=0.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig4_roc_curves.png'), dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("fig4 saved -> figures/fig4_roc_curves.png (dpi=300)")

# ---------------- fig5: SHAP mean absolute value ----------------
import shap
d = load_splits()
rf = RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_leaf=1,
                            class_weight='balanced', random_state=SEED, n_jobs=-1)
rf.fit(d["X_train"], d["y_train"])

# features passed as a pandas DataFrame, as required
X_df = d["X_test_df"].sample(n=min(800, len(d["X_test_df"])), random_state=SEED)
assert hasattr(X_df, 'columns'), "SHAP input must be a DataFrame"
print("SHAP input type: %s, shape %s" % (type(X_df).__name__, X_df.shape))

sv = shap.TreeExplainer(rf).shap_values(X_df)
sv = np.asarray(sv)
if sv.ndim == 3:                      # (n, features, classes) -> positive class
    sv = sv[:, :, 1]
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)
names = [X_df.columns[i] for i in order]

fig, ax = plt.subplots(figsize=(7.2, 5.4))
fig.patch.set_facecolor('white')
ax.barh(range(len(order)), mean_abs[order], color='#2E5BA8', height=0.72)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(names, fontsize=9)
for i, v in enumerate(mean_abs[order]):
    ax.text(v + mean_abs.max() * 0.012, i, "%.4f" % v, va='center', fontsize=7.6)
ax.set_xlabel('mean(|SHAP value|)  -  mean impact on model output', fontsize=9.5)
ax.set_xlim(0, mean_abs.max() * 1.16)
ax.set_title('Random Forest feature importance (SHAP)\ndeduplicated data, %d test rows, seed=%d'
             % (len(X_df), SEED), fontsize=11)
ax.grid(True, axis='x', ls='--', alpha=0.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig5_shap_importance.png'), dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("fig5 saved -> figures/fig5_shap_importance.png (dpi=300)")

json.dump({'shap_mean_abs': {X_df.columns[i]: float(mean_abs[i])
                             for i in range(len(mean_abs))}},
          open(os.path.join(BASE, 'results', 'shap.json'), 'w'), indent=2)
print("\ntop SHAP features:")
for i in order[::-1][:6]:
    print("  %-20s %.4f" % (X_df.columns[i], mean_abs[i]))
