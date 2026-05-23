# Save as gen_fig2.py in your QGene folder and run:
# python gen_fig2.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

samples  = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500])
qsvm_auc = np.array([0.742, 0.768, 0.789, 0.805, 0.820, 0.833, 0.844, 0.852, 0.860])
svm_base = np.full_like(samples, 0.940, dtype=float)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
fig.patch.set_facecolor('white')

ax.plot(samples, qsvm_auc, color='#7B2D8B', linewidth=2.2,
        marker='o', markersize=6, label='QSVM — ZZFeatureMap (4 qubits, 2 reps)', zorder=3)
ax.plot(samples, svm_base, color='#2E5BA8', linewidth=2.0,
        linestyle='--', label='Classical SVM baseline (26,152 samples)', zorder=2)
ax.fill_between(samples, qsvm_auc, svm_base, alpha=0.08, color='#7B2D8B')

ax.annotate('', xy=(500, 0.860), xytext=(500, 0.940),
    arrowprops=dict(arrowstyle='<->', color='#B85000', lw=1.8))
ax.text(507, 0.900, '52× fewer\nsamples\n(~1.9% of data)',
        fontsize=8, color='#B85000', va='center')
ax.axvline(x=500, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

ax.scatter([500], [0.860], color='#7B2D8B', s=60, zorder=5)
ax.annotate('QSVM: 0.860', xy=(500, 0.860), xytext=(380, 0.848),
    fontsize=8, color='#7B2D8B',
    arrowprops=dict(arrowstyle='->', color='#7B2D8B', lw=1.2))
ax.annotate('SVM: 0.940', xy=(100, 0.940), xytext=(120, 0.952),
    fontsize=8, color='#2E5BA8',
    arrowprops=dict(arrowstyle='->', color='#2E5BA8', lw=1.2))

ax.set_xlabel('Number of QSVM Training Samples', fontsize=10)
ax.set_ylabel('ROC-AUC', fontsize=10)
ax.set_xlim(80, 570)
ax.set_ylim(0.70, 0.975)
ax.set_xticks(samples)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.35, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig2_quantum_advantage.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved!")