"""
Manuscript-slot variant of Fig. 3: exactly THREE series, sized for full text width.

Series, as specified by the manuscript placeholder:
  1. QSVM                                  (4-D PCA angle features)
  2. matched-subset classical SVM          (IDENTICAL indices, 4-D PCA angles)
  3. full-data classical SVM baseline      (all 13,279 training rows, 14-D)

The 14-D matched-subset arm from figures/fig3_learning_curves.png is deliberately
omitted here to hit the three-series spec; that fuller version remains the
reference figure.

Plots ONLY from results/learning_curve.json - no literals.
"""

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

Q, C, BASECOL = '#7B2D8B', '#B85000', '#2E5BA8'

# full text width for Springer LNNS single-column body text
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
fig.patch.set_facecolor('white')

panels = [
    (axes[0], 'qsvm_acc', 'svm4_acc', R['baseline_14d']['acc'], 'Accuracy (%)', 100.0, '{:.2f} pts'),
    (axes[1], 'qsvm_auc', 'svm4_auc', R['baseline_14d']['auc'], 'ROC-AUC', 1.0, '{:.4f}'),
]

for ax, kq, kc, base, ylab, mult, fmt in panels:
    ax.plot(n, get(kq) * mult, color=Q, lw=2.3, marker='o', ms=6.5, zorder=4,
            label='QSVM (ZZFeatureMap, 4 qubits, reps = 2)')
    ax.plot(n, get(kc) * mult, color=C, lw=2.3, marker='s', ms=6.5, zorder=3,
            label='Classical RBF SVM, identical training indices')
    ax.axhline(base * mult, color=BASECOL, lw=1.9, ls='--', zorder=2,
               label='Full-data classical SVM (n = %s)' % format(R['baseline_14d']['n_train'], ','))
    ax.fill_between(n, get(kq) * mult, get(kc) * mult, alpha=0.11, color=C, zorder=1)

    # gaps at the two endpoints, computed from the arrays being plotted
    for xi, off in ((100, 14), (500, -155)):
        i = int(np.where(n == xi)[0][0])
        lo, hi = get(kq)[i] * mult, get(kc)[i] * mult
        ax.annotate('', xy=(xi, hi), xytext=(xi, lo),
                    arrowprops=dict(arrowstyle='<->', color='#B00020', lw=1.6))
        ax.text(xi + off, (lo + hi) / 2, fmt.format(hi - lo), fontsize=8.5,
                color='#B00020', va='center', fontweight='bold')

    ax.set_xlabel('QSVM / matched classical training-subset size $n$', fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xticks(n)
    ax.tick_params(labelsize=9)
    ax.grid(True, ls='--', alpha=0.32, color='gray')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

axes[0].legend(fontsize=8.4, loc='lower right', framealpha=0.96)
fig.suptitle('QSVM vs index-matched classical SVM, deduplicated BRCA1/2 variants '
             '(test set n = %s, seed = %d)' % (format(R['test_n'], ','), R['seed']),
             fontsize=11, y=1.02)
plt.tight_layout()

out = os.path.join(BASE, 'figures', 'fig3_learning_curves_3series.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print('saved %s at dpi=300' % out)

im = plt.imread(out)
print('pixel size: %d x %d  (at 300 dpi -> %.2f x %.2f inches)'
      % (im.shape[1], im.shape[0], im.shape[1] / 300, im.shape[0] / 300))
print('series plotted per panel: 3  (QSVM, matched classical, full-data baseline)')
print('\nendpoint gaps (matched classical minus QSVM), from the plotted arrays:')
for xi in (100, 500):
    i = int(np.where(n == xi)[0][0])
    print('  n=%d: accuracy %+.2f pts | ROC-AUC %+.4f'
          % (xi, (get('svm4_acc')[i] - get('qsvm_acc')[i]) * 100,
             get('svm4_auc')[i] - get('qsvm_auc')[i]))
