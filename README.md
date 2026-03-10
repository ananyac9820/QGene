QGene – Hybrid Classical-Quantum Machine Learning for BRCA Variant Classification
📌 Project Overview
QGene is a hybrid machine learning system designed to classify BRCA1 and BRCA2 gene mutations as:

Benign (Harmless)
Pathogenic (Disease-causing)

The system combines:

Classical Machine Learning (Random Forest, SVM)
Quantum Machine Learning (VQC, QSVM using Qiskit)
Hybrid Ensemble Modeling
Explainability (SHAP)

The final goal is to build a research-grade prototype with a web interface for mutation prediction.

🎯 Current Status: Week 3 Complete ✅
Timeline: 8-week project
Progress: 3/8 weeks completed (37.5%)
Status: ON SCHEDULE

📊 Week 1 Progress (Completed ✅)
Data Acquisition & Exploration
✅ Environment setup (Python 3.12.7 + Anaconda + VS Code)
✅ Installed required ML & Quantum libraries (scikit-learn, qiskit, shap, pandas, numpy, flask)
✅ Downloaded ClinVar dataset from NCBI
✅ Filtered BRCA1 & BRCA2 variants
✅ Cleaned dataset: 37,362 variants
✅ Created binary classification labels (0=Benign, 1=Pathogenic)
✅ Performed comprehensive exploratory data analysis
✅ Generated 9 detailed visualizations
Dataset Characteristics:

Total Variants: 37,362
BRCA1: 15,476 variants (41.4%)
BRCA2: 21,886 variants (58.6%)
Class Balance: 48.7% benign, 51.3% pathogenic (excellent balance!)
Data Quality: Zero missing values
Most Common Type: Single nucleotide variant (21,247 occurrences)


🔧 Week 2 Progress (Completed ✅)
Feature Engineering & Data Preprocessing
✅ Converted categorical features to numerical encodings
✅ Engineered biologically meaningful features:

Gene encoding (BRCA1=0, BRCA2=1)
Mutation type encoding (LabelEncoder)
Mutation length calculation (Stop - Start)
Genomic position (Start coordinate)

✅ Created ML-ready feature matrix with 4 features:

Gene_encoded: Gene identity
Type_encoded: Mutation mechanism
Start: Genomic position hotspot information
Mutation_length: Structural severity

✅ Applied stratified train/validation/test split:

Training: 26,153 samples (70%)
Validation: 5,604 samples (15%)
Test: 5,605 samples (15%)

✅ Applied StandardScaler normalization (mean≈0, std=1)
✅ Verified class balance maintained across all splits
✅ Saved preprocessed arrays (.npy format) for reproducibility

🤖 Week 3 Progress (Completed ✅)
Classical Machine Learning Model Training
✅ Trained Random Forest Classifier (100 trees, max_depth=20)
✅ Trained Support Vector Machine (RBF kernel, C=1.0)
✅ Comprehensive model evaluation with multiple metrics
✅ Created performance visualization suite
✅ Saved trained models (.pkl format)
📊 Final Test Set Performance
🌲 Random Forest (BEST MODEL 🏆)

Accuracy: 87.71%
Precision: 94.03%
Recall: 81.18%
F1-Score: 87.13%
ROC-AUC: 92.73%

🎯 Support Vector Machine (SVM)

Accuracy: 86.23%
Precision: 93.65%
Recall: 78.46%
F1-Score: 85.38%
ROC-AUC: 86.60%

Key Findings:

Both models exceed 85% accuracy threshold ✅
Random Forest outperforms SVM across all metrics
High precision (>93%) indicates low false positive rate
Models ready for comparison with quantum approaches

🛠️ Technologies Used
Machine Learning & Data Science

Python 3.12.7 (Anaconda distribution)
scikit-learn – Classical ML algorithms
pandas – Data manipulation
numpy – Numerical computing
matplotlib & seaborn – Data visualization

Quantum Computing (Upcoming Week 4-5)

Qiskit 2.3.0 – IBM Quantum framework
qiskit-machine-learning – Quantum ML algorithms

Explainability & Web (Upcoming)

SHAP – Model interpretability
Flask – Web application framework

Development Tools

VS Code – IDE with Jupyter extension
Git/GitHub – Version control
Jupyter Notebooks – Interactive development

📊 Performance Metrics Explained

Accuracy: Overall correctness (correctly classified / total samples)
Precision: Of predicted pathogenic, how many are truly pathogenic (minimizes false alarms)
Recall (Sensitivity): Of actual pathogenic, how many were detected (minimizes missed cases)
F1-Score: Harmonic mean of precision and recall (balanced metric)
ROC-AUC: Area under receiver operating characteristic curve (discrimination ability)

Medical Context: High precision is critical to avoid unnecessary patient anxiety from false positives, while high recall ensures dangerous mutations aren't missed
