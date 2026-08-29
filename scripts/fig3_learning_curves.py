"""Task 7 figure - plots ONLY from results/learning_curve.json (no literals)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE

mpl.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif', 'serif']

R = json.load(open(os.path.join(BASE, 'results', 'learning_curve.json')))
n = np.array([r['n'] for r in R['rows']])
get = lambda k: np.array([r[k] for r in R['rows']])

Q, C4, C14 = '#7B2D8B', '#B85000', '#2E5BA8'
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
fig.patch.set_facecolor('white')

panels = [
    (axes[0], 'qsvm_acc', 'svm4_acc', 'svm14_acc',
     R['baseline_14d']['acc'], 'Accuracy on full test set', 100.0, '%'),
    (axes[1], 'qsvm_auc', 'svm4_auc', 'svm14_auc',
     R['baseline_14d']['auc'], 'ROC-AUC on full test set', 1.0, ''),
]

for ax, kq, k4, k14, base, ylab, mult, suf in panels:
    ax.plot(n, get(kq) * mult, color=Q, lw=2.2, marker='o', ms=6, zorder=4,
            label='QSVM (ZZFeatureMap, 4 qubits, reps=2)')
    ax.plot(n, get(k4) * mult, color=C4, lw=2.2, marker='s', ms=6, zorder=3,
            label='Classical RBF SVM - identical indices, 4-D PCA')
    ax.plot(n, get(k14) * mult, color=C14, lw=1.8, marker='^', ms=5, ls='-.', zorder=2,
            label='Classical RBF SVM - identical indices, 14-D')
    ax.axhline(base * mult, color='#444444', lw=1.6, ls='--', zorder=1,
               label=f'Full-data classical SVM (n={R["baseline_14d"]["n_train"]:,})')
    ax.fill_between(n, get(kq) * mult, get(k4) * mult, alpha=0.10, color=C4, zorder=0)
    ax.set_xlabel('QSVM / matched classical training-subset size $n$', fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xticks(n)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, ls='--', alpha=0.32, color='gray')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

# annotate the two gaps the paper quotes, computed from the arrays
for ax, kq, k4, mult, fmt in [(axes[0], 'qsvm_acc', 'svm4_acc', 100.0, '{:.2f} pts'),
                              (axes[1], 'qsvm_auc', 'svm4_auc', 1.0, '{:.4f}')]:
    for xi, lab_off in ((100, 12), (500, -150)):
        i = int(np.where(n == xi)[0][0])
        lo, hi = get(kq)[i] * mult, get(k4)[i] * mult
        ax.annotate('', xy=(xi, hi), xytext=(xi, lo),
                    arrowprops=dict(arrowstyle='<->', color='#B00020', lw=1.5))
        ax.text(xi + lab_off, (lo + hi) / 2, fmt.format(hi - lo), fontsize=8,
                color='#B00020', va='center', fontweight='bold')

axes[0].legend(fontsize=7.6, loc='lower right', framealpha=0.95)
fig.suptitle('QSVM vs index-matched classical SVM on deduplicated BRCA1/2 variants\n'
             f'(test set n={R["test_n"]:,}; subsets drawn from the {R["baseline_14d"]["n_train"]:,}-row training split; seed={R["seed"]})',
             fontsize=11, y=1.04)
plt.tight_layout()
os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)
out = os.path.join(BASE, 'figures', 'fig3_learning_curves.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"saved {out} at dpi=300")

# gaps, printed from the same arrays that were plotted
print("\ngap = matched classical (4-D PCA, identical indices) minus QSVM:")
for xi in (100, 500):
    i = int(np.where(n == xi)[0][0])
    print(f"  n={xi}: accuracy {(get('svm4_acc')[i]-get('qsvm_acc')[i])*100:+.2f} pts "
          f"| ROC-AUC {get('svm4_auc')[i]-get('qsvm_auc')[i]:+.4f}")
print("gap vs 14-D matched classical:")
for xi in (100, 500):
    i = int(np.where(n == xi)[0][0])
    print(f"  n={xi}: accuracy {(get('svm14_acc')[i]-get('qsvm_acc')[i])*100:+.2f} pts "
          f"| ROC-AUC {get('svm14_auc')[i]-get('qsvm_auc')[i]:+.4f}")
print(f"\nQSVM never exceeds the matched 4-D classical arm: "
      f"{bool((get('qsvm_acc') < get('svm4_acc')).all())}")
