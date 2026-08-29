# QGene — camera-ready rerun after the assembly data leak

Branch `fix/assembly-dedup`. Every number below was produced by code run in this
session, from `scripts/`. Nothing is carried forward from the previous draft.
Where something could not be computed it is marked **NOT COMPUTED** and left blank.

**Seed: `random_state = 42`, used for every split, model, subset draw and PCA.**

Reproduce in order:

```bash
python scripts/dedup.py && python scripts/train_classical.py && python scripts/train_quantum_hybrid.py && python scripts/make_tables.py && python scripts/learning_curve.py && python scripts/fig3_learning_curves.py && python scripts/ablation_and_env.py && python scripts/make_figures.py
```

---

## 1. Deduplication

`scripts/dedup.py` → `data/brca_mutations_dedup.csv`

| | value |
|---|---|
| rows before | 37,362 |
| distinct `(Name, GeneSymbol)` | 18,971 |
| duplicated on key | 18,391 |
| **rows after** | **18,971** |

Per-assembly before: GRCh37 18,708 · GRCh38 18,360 · na 294.
After (source assembly of the surviving row): GRCh38 18,353 · GRCh37 349 · na 269.
Preference order GRCh38 > GRCh37 > na.

**Conflicting-label variants: 0.** Every variant carried an identical `Label`
across all of its assembly rows, so no variants were dropped on that account.
The entire 37,362 → 18,971 reduction is assembly duplication.

Labels after dedup: **9,858 pathogenic / 9,113 benign** (51.96% pathogenic).
Assertion `zero duplicates remain on (Name, GeneSymbol)` passes.

---

## 2. Splits and class balance

70/15/15 stratified, seed 42.

| split | n | pathogenic | benign | pathogenic % |
|---|---|---|---|---|
| train | 13,279 | 6,900 | 6,379 | 51.96% |
| val | 2,846 | 1,479 | 1,367 | 51.97% |
| test | 2,846 | 1,479 | 1,367 | 51.97% |
| total | 18,971 | | | |

Hyperparameters unchanged from the original code: RF 500 trees / gini /
`min_samples_leaf=1` / `class_weight='balanced'`; SVM RBF `C=100` /
`gamma='scale'` / `class_weight='balanced'` / `probability=True`;
`StandardScaler` fitted on train only.

---

## 3. Table 4 — test-set performance

All values computed from `results/test_preds.npz` by `scripts/make_tables.py`.
All three models are Platt-scaled on the validation split; predictions are
thresholded at 0.5 on the calibrated probability.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 87.67% | 0.883 | 0.880 | 0.881 | 0.946 |
| SVM (RBF) | 88.79% | 0.993 | 0.790 | 0.880 | 0.954 |
| QSVM | 85.31% | 0.890 | 0.818 | 0.853 | 0.911 |
| Hybrid | 88.69% | 0.963 | 0.813 | 0.882 | 0.948 |

## Table 5 — confusion counts and clinical miss rate

| Model | TP | TN | FP | FN | FN as % of test | missed pathogenic % (1−recall) |
|---|---|---|---|---|---|---|
| Random Forest | 1301 | 1194 | 173 | 178 | 6.25% | 12.04% |
| SVM (RBF) | 1168 | 1359 | 8 | 311 | 10.93% | 21.03% |
| QSVM | 1210 | 1218 | 149 | 269 | 9.45% | 18.19% |
| Hybrid | 1203 | 1321 | 46 | 276 | 9.70% | 18.66% |

### Consistency self-checks — **ALL PASS**

| check | RF | SVM | QSVM | Hybrid |
|---|---|---|---|---|
| TP+TN+FP+FN == 2846 | PASS | PASS | PASS | PASS |
| TP+FN == 1479 pathogenic | PASS | PASS | PASS | PASS |
| TN+FP == 1367 benign | PASS | PASS | PASS | PASS |
| accuracy from counts == Table 4 (2 dp) | PASS | PASS | PASS | PASS |
| precision from counts == Table 4 (3 dp) | PASS | PASS | PASS | PASS |
| recall from counts == Table 4 (3 dp) | PASS | PASS | PASS | PASS |
| F1 from counts == Table 4 (3 dp) | PASS | PASS | PASS | PASS |

**Note on the SVM operating point.** After Platt scaling, thresholding at 0.5
puts the SVM at an unusually conservative point: precision 0.993 with only
8 false positives, but 311 false negatives (21% of pathogenic variants missed).
That is a real property of the calibrated model, not an error — but for a
clinical screening tool it is the wrong trade-off, and the RF's more balanced
0.883/0.880 is the better default despite its lower headline accuracy. Worth a
sentence in the discussion.

---

## 4. McNemar — RF vs QSVM

From the saved arrays. Convention: **b = RF wrong & QSVM right**,
**c = RF right & QSVM wrong**.

| quantity | value |
|---|---|
| both right | 2265 |
| **b** | **163** |
| **c** | **230** |
| both wrong | 188 |
| total | 2846 ✓ |
| **χ²** | **11.0840** |
| **p** | **8.708 × 10⁻⁴** |

Sanity checks:

```
b + both_wrong = 163 + 188 = 351   RF total errors   = 351   PASS
c + both_wrong = 230 + 188 = 418   QSVM total errors = 418   PASS
```

Working, so it is verifiable by hand:

```
chi2 = (|b-c| - 1)^2 / (b + c) = (|163-230| - 1)^2 / 393 = 66^2 / 393 = 4356 / 393 = 11.0840
p    = P(chi2_1 > 11.0840) = 0.000870772
```

`statsmodels.stats.contingency_tables.mcnemar(exact=False, correction=True)`
returns the identical statistic (`np.isclose` → True).

RF significantly outperforms QSVM (p < 0.001). The previous draft's
χ²=4.31 / p=0.038 was arithmetically impossible given its own confusion matrix —
see the DELTA section.

---

## 5. Hybrid weights

Platt-scaled RF, SVM and QSVM on validation; classical signal is
`(rf_prob + svm_prob) / 2`; grid search on validation accuracy.

| w_C | w_Q | val accuracy | val F1 |
|---|---|---|---|
| 0.3 | 0.7 | 86.54% | 0.8648 |
| 0.4 | 0.6 | 86.75% | 0.8662 |
| 0.5 | 0.5 | 87.42% | 0.8719 |
| 0.6 | 0.4 | 88.48% | 0.8806 |
| 0.7 | 0.3 | 88.58% | 0.8807 |
| **0.8** | **0.2** | **88.76%** | **0.8821** |
| 0.9 | 0.1 | 88.58% | 0.8813 |

**Winning weights: w_C = 0.8, w_Q = 0.2.** Not 0.6/0.4. The quantum arm earns
half the weight it was previously assigned, and the previous 0.6/0.4 was never
the output of a search in any case (see DELTA).

Platt coefficients: RF a=−5.1084 b=+2.5135 · SVM a=−1.7219 b=−0.2862 ·
QSVM a=−2.1467 b=−0.3797.

---

## 6. Learning curve (task 7) — the paper's central claim

`scripts/learning_curve.py`. Stratified subsets drawn from the 13,279-row
deduplicated training split; **all arms evaluated on the full 2,846-row test set**.

One index array is built per *n* and shared by every arm. Identity is asserted
in code (`np.array_equal` across all three arms, plus a label-vector check and a
uniqueness check). **All assertions passed.** SHA-1 of each index array is
recorded in `results/learning_curve.json`.

Two matched classical arms are reported, because which one is used changes what
the claim means:
- **4-D PCA** — identical inputs to the QSVM, so the only difference is quantum
  kernel vs RBF. This is the kernel-controlled comparison.
- **14-D scaled** — matches the full-data baseline's representation.

| n | QSVM acc | QSVM AUC | SVM 4-D acc | SVM 4-D AUC | SVM 14-D acc | SVM 14-D AUC |
|---|---|---|---|---|---|---|
| 100 | 76.56% | 0.7615 | 83.73% | 0.9169 | 81.52% | 0.8932 |
| 150 | 78.18% | 0.8194 | 87.32% | 0.9201 | 84.57% | 0.9016 |
| 200 | 79.23% | 0.8492 | 86.75% | 0.9184 | 86.19% | 0.9113 |
| 250 | 79.48% | 0.8436 | 85.91% | 0.9271 | 86.33% | 0.9201 |
| 300 | 80.99% | 0.8538 | 87.84% | 0.9091 | 86.61% | 0.9177 |
| 350 | 82.29% | 0.8640 | 87.98% | 0.9234 | 86.23% | 0.9176 |
| 400 | 83.38% | 0.8701 | 88.02% | 0.9148 | 86.44% | 0.9191 |
| 450 | 83.80% | 0.8740 | 87.84% | 0.9138 | 86.26% | 0.9158 |
| 500 | 84.93% | 0.8904 | 87.84% | 0.9129 | 86.44% | 0.9154 |

**Full-data classical baselines** (all 13,279 training rows):
14-D **88.79%** / AUC 0.9537 · 4-D PCA **88.33%** / AUC 0.9322.

### Gaps (matched classical − QSVM; positive = classical ahead)

| | n=100 | n=500 |
|---|---|---|
| vs matched 4-D PCA | **+7.17 pts** / AUC +0.1554 | **+2.92 pts** / AUC +0.0225 |
| vs matched 14-D | +4.95 pts / AUC +0.1317 | +1.51 pts / AUC +0.0250 |

**QSVM does not exceed the matched classical arm at any n** — asserted in code,
result `True`.

### What this means

The quantum-advantage claim does not survive the control. Given the *same 100
training rows*, the classical RBF SVM scores 83.73% to the QSVM's 76.56%. The
direction of the previous draft's "3.3 points at n=100" is inverted and the
magnitude roughly doubled.

There is a real result here, just not that one: QSVM AUC climbs from 0.7615 to
0.8904 across the range while the matched classical arm stays roughly flat at
0.91–0.93 — the quantum kernel converges *toward* classical performance from
below, and has not caught it by n=500. Separately, the matched classical SVM at
n=500 (87.84%) is already within 0.95 points of the full-data baseline trained
on 26× more data, which says the task saturates early for any kernel. That is a
defensible low-data-regime finding. It is not a quantum advantage.

PCA(4) retains only **75.77%** of variance (0.2914 / 0.2116 / 0.1569 / 0.0978),
which is most of why both 4-D arms trail the 14-D full-data baseline.

Figure: `figures/fig3_learning_curves.png` (dpi=300).

---

## 7. Label-quality ablation (task 8)

RF retrained with `review_score` and `is_expert` dropped (14 → 12 features).
Identical rows in every split; only the columns differ (asserted).

| model | features | accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| RF full | 14 | 87.67% | 0.8811 | 0.9457 |
| **RF ablated** | **12** | **87.81%** | **0.8821** | **0.9489** |

**Delta: +0.14 accuracy points.** Removing the curation-confidence features does
not hurt the model — it very slightly helps. This is *stronger* than the lower
bound the paper wanted: the argument that `review_score` and `is_expert` encode
curation confidence rather than biology is supported, and the model does not
depend on them at all. **87.81%** is the stated lower bound.

---

## 8. Environment (Table 3)

| item | value |
|---|---|
| **seed** | **42** |
| Python | 3.13.5 (CPython) |
| platform | Windows-11-10.0.26200-SP0 |
| CPU | Intel(R) Core(TM) Ultra 5 125H |
| physical cores | 14 |
| logical cores | 18 |
| RAM | 15.61 GB |
| scikit-learn | 1.8.0 |
| numpy | 2.4.6 |
| pandas | 3.0.1 |
| scipy | 1.17.1 |
| qiskit | 1.2.4 |
| qiskit-aer | 0.17.2 |
| qiskit-machine-learning | 0.7.2 |
| shap | 0.51.0 |
| flask | 3.1.3 |
| statsmodels | 0.14.6 |

### Kernel wall-clock

| measurement | value |
|---|---|
| `FidelityQuantumKernel`, 500×500 train kernel (124,750 unique pairs) | **1008.6 s = 00:16 (hh:mm)**, 8.085 ms/pair |
| `FidelityStatevectorKernel`, full QSVM pipeline (500×500 train + 2×2846×500 val/test) | 97.8 s = 00:01 |
| `FidelityStatevectorKernel`, all 9 learning-curve points | 123.7 s = 00:02 |

**Quote the 00:16 figure for the paper** — it is the spec-named
`FidelityQuantumKernel` on the 500-sample training kernel.

**Implementation note, needed for honesty in the methods section.** The QSVM
results and the learning curve were computed with `FidelityStatevectorKernel`
rather than `FidelityQuantumKernel`. These are the same noiseless statevector
fidelity kernel; equivalence is verified inside every run and printed to the log
(`max|K₁ − K₂| = 5.11 × 10⁻¹⁵`, i.e. float64 round-off). The statevector path is
~34× faster, which is what made the 9-point × 3-arm controlled experiment
affordable. The kernel *values*, and therefore every reported metric, are
unchanged by this choice.

---

## 9. Figures (all dpi=300)

| figure | file | status |
|---|---|---|
| fig1 architecture | — | **SKIPPED** — no source in repo (hand-drawn) |
| fig2 grouped metric bars | `figures/fig2_metric_bars.png` | regenerated from `test_preds.npz` |
| fig3 learning curves | `figures/fig3_learning_curves.png` | new, from `learning_curve.json` |
| fig4 ROC curves | `figures/fig4_roc_curves.png` | regenerated from `test_preds.npz` |
| fig5 SHAP mean-\|value\| | `figures/fig5_shap_importance.png` | features passed as a pandas DataFrame (asserted) |

Top SHAP features: `is_snv` 0.1251 · `type_enc` 0.1073 · `is_frameshift` 0.0527 ·
`has_protein_change` 0.0418 · `aa_position` 0.0358 · `nuc_position` 0.0280.

---

## 10. DELTA — every number that changed

### 10a. Numbers the leak actually inflated

Only two of the old figures can be compared like-for-like, because the rest were
never computed (§10b).

| quantity | old | new | change | leak's contribution |
|---|---|---|---|---|
| RF accuracy | 88.90% | **87.67%** | **−1.23 pts** | see below |
| SVM accuracy | 88.46% | **88.79%** | **+0.33 pts** | none — went *up* |
| QSVM accuracy | 86.00% | **85.31%** | −0.69 pts | not comparable (old was on a 500-row test subset) |
| Hybrid accuracy | 88.65% | **88.69%** | +0.04 pts | not comparable |
| RF ROC-AUC | 0.960 | **0.946** | −0.014 | old value was the AUPRC column |
| SVM ROC-AUC | 0.940 | **0.954** | +0.014 | old value was a hardcoded literal |
| QSVM ROC-AUC | 0.860 | **0.911** | +0.051 | not comparable |
| Hybrid ROC-AUC | 0.941 | **0.948** | +0.007 | not comparable |
| hybrid weights | 0.6 / 0.4 | **0.8 / 0.2** | — | old value was a hardcoded literal |
| McNemar χ² | 4.31 | **11.084** | — | old value was impossible |
| McNemar p | 0.038 | **8.71 × 10⁻⁴** | — | old value was impossible |

**How much was the leak inflating things?** For the RF, **about 1.2 accuracy
points**. That is the only defensible statement of the leak's magnitude. The
duplicated rows carried the same label and near-identical features — only
`Start`/`Stop` (hence `position` and `log_length`) differ between assemblies — so
a leaked test variant handed the RF a near-copy of a training row, but one whose
informative features were already shared with genuinely similar variants. The
SVM was not inflated at all; it improved on clean data. RF train accuracy is
still 99.74%, which is `min_samples_leaf=1` behaving normally, not leakage.

### 10b. Numbers that were never computed at all

A provenance audit of the full git history (no file has ever been deleted, so
the search is exhaustive) found that **most of the old Table 4, all of Table 5,
the McNemar test and the §5.7 ablation were not produced by any code in this
repository.** This is a larger problem than the leak.

| old value | status | evidence |
|---|---|---|
| RF 88.90 / 0.937 / 0.886 | **no producing line**; repo's real RF run gives 89.40 / 0.9471 / 0.8903 | `visualizations/metrics_comparison.csv:2` |
| RF 0.840 recall | computed (0.83984) | same file |
| RF 0.960 "ROC-AUC" | **mislabelled** — that is the AUPRC column; real ROC-AUC was 0.9465 | same file, col 7 |
| SVM 88.46 / 0.934 / 0.834 / 0.881 | **no producing line**; repo's real SVM run gives 86.40 / 0.9273 / 0.7969 / 0.8571 | `metrics_comparison.csv:3` |
| SVM 0.940 "ROC-AUC" | **hardcoded literal** | `gen_fig2.py:12` |
| QSVM 86.00 / 0.931 / 0.785 / 0.852 / 0.860 | computed, but on a **500-sample** test subset, not 5,605 | `04_quantum_ml_training.ipynb:404-408` |
| Hybrid 88.65 / 0.935 / 0.837 / 0.883 | **no producing line**; the one real hybrid run gave 86.60 / F1 0.8601 at **w = 0.50/0.50** | `05_comparison_analysis.ipynb:1010` |
| Hybrid 0.941 ROC-AUC | computed (0.9412) — but from the w=0.50/0.50 run | `05_…ipynb:774` |
| all four confusion-count sets | **no producing line** — no `confusion_matrix` call ever produced 5,605-row counts | — |
| McNemar χ²=4.31, p=0.038 | **no producing line, and provably impossible** (below) | — |
| hybrid weights 0.6 / 0.4 | **hardcoded literal** | `fix_hybrid.py:9-10` |
| RF 87.71% "with 4 PCA components" | computed, but **misattributed** — it is plain RF test accuracy on the full feature set; **no PCA ablation exists anywhere in the repo** | `03_classical_ml_training.ipynb:261` |
| learning-curve gaps "3.3 / 2.5 points" | **no producing line**; the underlying curve `[0.742 … 0.860]` is a hardcoded literal | `gen_fig2.py:11` |

**Proof that the old χ²=4.31 was impossible.** From the old confusion matrix,
RF errors = 162+460 = 622 and QSVM errors = 167+618 = 785, so b − c = 163. Since
c can only range 0…622, b+c ranges 163…1407, giving an attainable
χ² = 26244/(b+c) of **[18.65, 161.01]**. Reaching 4.31 needs b+c = 6089, more
than four times the largest possible value. At the *smallest* attainable χ² of
18.65, p = 1.57 × 10⁻⁵ — nowhere near 0.038.

**The one McNemar that was actually run** (`05_…ipynb:695-717`, output at `:660`)
gave b=23, c=6, χ²=8.8276, **p=0.0030**, and printed the verdict *"Classical RF
significantly outperforms QSVM."* The notebook contains an `if p1 >= 0.05` branch
that would have printed a quantum-advantage conclusion; the run took the `else`.

**Other files that write results as literals**, same pattern as `gen_fig2.py` and
`fix_hybrid.py`: `qgene_app.py:184-186` (hardcoded RF 88.60/88.76/96.00, SVM
88.39/87.45/93.97, QSVM 86.00/85.17/86.02 in the `/model-info` route, with
defaults 0.906/0.900/0.945 at `:187-189`); `README.md:12`;
`templates/stats.html:28-30`; `templates/index.html:56`; `templates/app.html:60`;
`templates/sidebar.html:21`.

Counting them up, **four mutually inconsistent RF accuracies** are in
circulation across the repo and the draft: 87.71 (nb03), 89.40 (nb05), 88.60
(`qgene_app.py`), 88.90 (paper). None is a rerun of another.

### 10c. What still needs a human decision

1. **The central claim needs reframing.** The controlled learning curve refutes
   quantum advantage. The honest finding is convergence-from-below plus early
   task saturation.
2. **Table 5 and §5.7 in the submitted draft have no computational provenance.**
   That is a correction-notice question for the editors, not something a rerun
   fixes.
3. The web app and README still serve the old hardcoded numbers and should be
   updated before the paper is published with a link to them.

---

## Files written this session

```
scripts/dedup.py                 scripts/common.py
scripts/train_classical.py       scripts/train_quantum_hybrid.py
scripts/make_tables.py           scripts/learning_curve.py
scripts/fig3_learning_curves.py  scripts/ablation_and_env.py
scripts/make_figures.py          scripts/time_kernel_spec.py

data/brca_mutations_dedup.csv
results/test_preds.npz           results/tables.json
results/hybrid_config.json       results/learning_curve.json
results/ablation_env.json        results/kernel_timing.json
results/shap.json                results/kernel_timing.log

figures/fig2_metric_bars.png     figures/fig3_learning_curves.png
figures/fig4_roc_curves.png      figures/fig5_shap_importance.png

models/random_forest_dedup.pkl   models/svm_dedup.pkl
models/qsvm_dedup.pkl            models/feature_scaler_dedup.pkl
models/pca_transformer_dedup.pkl models/pca_angle_scaler_dedup.pkl
models/hybrid_config_dedup.pkl   models/feature_names_dedup.pkl
```

No existing `.pkl`, figure, notebook or template was modified in place.
