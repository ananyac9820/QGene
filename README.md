QGene — Quantum-Classical Hybrid BRCA Variant Classifier

A hybrid classical-quantum machine learning system for BRCA1/BRCA2 genetic mutation pathogenicity prediction, with explainability, comparative analysis, and a full web interface.

🔬 Live Demo: qgene.onrender.com

Overview
QGene is a research-grade prototype that classifies BRCA1 and BRCA2 gene variants as Pathogenic or Benign using a hybrid ensemble of classical and quantum machine learning models. It is built on 37,362 ClinVar variants and combines Random Forest, SVM, Quantum SVM (via ZZFeatureMap kernel), and a learned-weight hybrid ensemble.
The system includes SHAP-based explainability, a comparative analysis of classical vs quantum approaches, and a full web application with PDF report generation.

Key Results
ModelAccuracyF1-ScoreROC-AUCTraining SamplesRandom Forest87.71%87.13%92.73%26,153SVM86.23%85.38%86.60%26,153QSVM (ZZFeatureMap)86.00%85.17%86.02%500VQC78.20%75.51%84.74%200Hybrid Ensemble ★86.60%86.01%94.12%Ensemble
Key finding: QSVM achieves statistically equivalent performance to classical SVM while using 52× fewer training samples, demonstrating quantum kernel efficiency in the low-data regime. The hybrid ensemble outperforms all individual models on ROC-AUC (94.12%).

Features

4 ML models — Random Forest, SVM, QSVM, VQC
Quantum kernel — ZZFeatureMap with entanglement (2 reps, Qiskit 2.3)
Hybrid ensemble — weights optimised on validation set
SHAP explainability — feature-level contributions for every prediction
Statistical validation — McNemar's test, learning curves, ROC overlay
PDF report generation — downloadable per-prediction report (jsPDF)
Batch prediction — CSV upload for multiple variants
7-page web application — landing, predict, disease info, dataset, stats, news, contact


Project Structure
QGene/
├── qgene_app.py                  # Flask backend — all routes and prediction logic
├── requirements.txt              # Python dependencies
├── render.yaml                   # Render.com deployment config
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml_training.ipynb
│   ├── 04_quantum_ml_training.ipynb
│   └── 05_comparison_analysis.ipynb
│
├── models/                       # Trained model files (Git LFS)
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── qsvm_upgraded.pkl
│   ├── quantum_kernel_zz.pkl
│   ├── pca_transformer.pkl
│   └── hybrid_config.pkl
│
├── data/                         # Preprocessed arrays (Git LFS)
│   ├── X_train.npy / y_train.npy
│   ├── X_val.npy   / y_val.npy
│   ├── X_test.npy  / y_test.npy
│   └── X_qsvm_test.npy + prediction arrays
│
├── visualizations/               # Publication figures
│   ├── fig1_performance_comparison.png
│   ├── fig2_roc_curves.png
│   ├── fig3_confusion_matrices.png
│   ├── fig4_quantum_advantage.png   ← key finding
│   ├── fig5_kernel_heatmap.png
│   └── fig6_summary.png
│
├── templates/                    # Flask HTML templates
│   ├── landing.html
│   ├── app.html
│   ├── sidebar.html
│   ├── about.html
│   ├── dataset.html
│   ├── stats.html
│   ├── news.html
│   └── contact.html
│
└── static/
    ├── css/style.css
    └── js/script.js

Setup & Installation
Prerequisites

Python 3.12.7
Anaconda or virtualenv

1. Clone the repository
bashgit clone https://github.com/ananyac9820/QGene.git
cd QGene
2. Create and activate virtual environment
bashpython -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Run the application
bashpython qgene_app.py
Visit http://localhost:5000

Quantum Implementation
The quantum component uses Qiskit 2.3 with qiskit-machine-learning 0.9.0.
QSVM architecture:

Feature reduction: PCA (4D → 2D, 70.5% variance retained)
Feature map: ZZFeatureMap (2 qubits, 2 reps) — captures qubit entanglement
Kernel: FidelityQuantumKernel via StatevectorSampler
Classifier: SVC with precomputed kernel matrix
Training set: 500 stratified samples

VQC architecture:

Feature map: ZZFeatureMap (2 reps)
Ansatz: RealAmplitudes (2 reps)
Optimizer: COBYLA (150 iterations)
Training set: 200 stratified samples


Important: Quantum models must be run in the .venv kernel (not Anaconda base), as qiskit-machine-learning is installed there.


Dataset

Source: NCBI ClinVar — ncbi.nlm.nih.gov/clinvar
Genes: BRCA1 (15,476 variants) and BRCA2 (21,886 variants)
Total variants: 37,362 (after quality filtering)
Class balance: 48.7% Benign / 51.3% Pathogenic
Features: gene, mutation type, genomic position, mutation length


8-Week Development Timeline
WeekMilestoneStatus1Data acquisition & exploration✅2Feature engineering & preprocessing✅3Classical ML (RF, SVM, SHAP)✅4Quantum ML setup (QSVM, VQC)✅5Comparison analysis & hybrid ensemble✅6Web application development✅7Deployment (Render.com)✅8Documentation & presentation🔄

Technologies
CategoryToolsClassical MLscikit-learn 1.8, NumPy, pandasQuantum MLQiskit 2.3, qiskit-machine-learning 0.9.0, qiskit-aer 0.17.2ExplainabilitySHAP 0.45.1WebFlask, jsPDFVisualisationmatplotlib, seabornDeploymentRender.com, Git LFSVersion controlGitHub

Limitations & Future Work

Quantum models trained on small subsets (500/200 samples) due to simulation cost — real quantum hardware would allow larger training sets
VQC optimizer (COBYLA) converged early; future work could explore SPSA or gradient-based optimizers
Feature set limited to 4 engineered features; incorporating conservation scores and allele frequency may improve performance
Extension to other BRCA-related genes (PALB2, CHEK2, ATM) is a natural next step


Disclaimer
⚠️ QGene is an academic research prototype. Predictions are generated by machine learning models and have not been clinically validated. This tool is intended for research and educational purposes only and should never be used as the basis for any medical or clinical decision. Always consult a qualified healthcare professional or genetic counselor.

Authors
Ananya Choudhari

Built with Qiskit · scikit-learn · Flask · ClinVar data
