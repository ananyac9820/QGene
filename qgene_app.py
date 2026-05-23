"""
QGene - Genetic Mutation Prediction System
Flask Backend
"""
from flask import Flask, render_template, request, jsonify, send_file
import pickle, numpy as np, pandas as pd
from datetime import datetime
import os
import re

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    with open(os.path.join(BASE, path), 'rb') as f:
        return pickle.load(f)

print("Loading models...")
rf         = load('models/random_forest_model.pkl')
svm        = load('models/svm_model.pkl')
qsvm       = load('models/qsvm_upgraded.pkl')
pca        = load('models/pca_transformer.pkl')
pca_scaler = load('models/pca_scaler.pkl')
scaler     = load('models/feature_scaler.pkl')
hybrid_cfg = load('models/hybrid_config.pkl')
X_qtrain   = np.load(os.path.join(BASE, 'data/X_qsvm_test.npy'))

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
qkernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=4, reps=2))

W_C = hybrid_cfg['w_classical']
W_Q = hybrid_cfg['w_quantum']
print(f"✅ Models loaded — Hybrid weights: Classical {W_C:.2f} / Quantum {W_Q:.2f}")

# ── Maps ───────────────────────────────────────────────────────────
REVIEW_MAP = {
    'reviewed by expert panel': 4,
    'criteria provided, multiple submitters, no conflicts': 3,
    'criteria provided, single submitter': 2,
    'no assertion criteria provided': 1,
    'no assertion provided': 0
}

TYPE_MAP = {
    'single nucleotide variant': 0, 'deletion': 1, 'insertion': 2,
    'duplication': 3, 'indel': 4, 'microsatellite': 5, 'inversion': 6,
    'missense': 0, 'nonsense': 1, 'frameshift': 4, 'silent': 0, 'splice site': 2,
}

# ── Feature Extraction ─────────────────────────────────────────────
def extract_features(data):
    gene         = 0 if str(data.get('gene', 'BRCA1')).upper() == 'BRCA1' else 1
    mtype_str    = str(data.get('mutation_type', 'missense')).lower()
    mtype        = TYPE_MAP.get(mtype_str, 0)
    position     = float(data.get('position', 41196312))
    length       = float(data.get('mutation_length', 1))
    log_length   = np.log1p(length)
    review_score = REVIEW_MAP.get(str(data.get('review_status', 'criteria provided, single submitter')), 2)
    is_expert    = 1 if review_score == 4 else 0

    name = str(data.get('variant_name', ''))
    nuc_m = re.search(r'c\.[-*]?(\d+)', name)
    aa_m  = re.search(r'p\.[A-Za-z]+(\d+)', name)
    nuc_position = int(nuc_m.group(1)) if nuc_m else 0
    aa_position  = int(aa_m.group(1))  if aa_m  else 0

    has_protein_change = 1 if 'p.' in name else 0
    is_frameshift = 1 if ('fs' in name or mtype_str in ('frameshift', 'indel')) else 0
    is_snv             = 1 if mtype_str == 'single nucleotide variant' else 0
    nuc_x_review       = nuc_position * review_score
    aa_x_gene          = aa_position  * gene
    type_x_gene        = mtype        * gene

    return np.array([[
        gene, mtype, position, log_length, review_score,
        nuc_position, aa_position, has_protein_change, is_frameshift,
        is_snv, is_expert, nuc_x_review, aa_x_gene, type_x_gene
    ]], dtype=float)

# ── Quantum Probability ────────────────────────────────────────────
def get_quantum_prob(features_14d):
    features_s   = scaler.transform(features_14d)
    features_pca = pca.transform(features_s)
    features_q   = pca_scaler.transform(features_pca)
    K_single     = qkernel.evaluate(features_q, X_qtrain)
    return float(qsvm.predict_proba(K_single)[0][1])

# ── SHAP Explanation ───────────────────────────────────────────────
def get_shap_explanation(features_14d):
    try:
        import shap
        import pandas as pd
        FEATURE_NAMES = ['gene_enc','type_enc','position','log_length','review_score',
                         'nuc_position','aa_position','has_protein_change','is_frameshift',
                         'is_snv','is_expert','nuc_x_review','aa_x_gene','type_x_gene']
        df_feat    = pd.DataFrame(features_14d, columns=FEATURE_NAMES)
        explainer  = shap.TreeExplainer(rf)
        shap_out   = explainer.shap_values(df_feat)
        # Handle both old and new shap output formats
        if isinstance(shap_out, list):
            sv = np.array(shap_out[1]).flatten()
        else:
            sv = np.array(shap_out).flatten()
        vals = features_14d[0]
        expl = [{'feature': n, 'value': round(float(v), 4),
                 'contribution': round(float(c), 4),
                 'impact': 'increases risk' if c > 0 else 'decreases risk'}
                for n, v, c in zip(FEATURE_NAMES, vals, sv)]
        return sorted(expl, key=lambda x: abs(x['contribution']), reverse=True)[:6]
    except Exception as e:
        return [{'feature': f'SHAP error: {str(e)[:60]}', 'value': 0,
                 'contribution': 0.0, 'impact': 'unknown'}]
# ── Routes ─────────────────────────────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/about')
def about():
    return render_template('about.html', active='about')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html', active='dataset')

@app.route('/stats')
def stats():
    return render_template('stats.html', active='stats')

@app.route('/news')
def news():
    return render_template('news.html', active='news')

@app.route('/contact')
def contact():
    return render_template('contact.html', active='contact')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.json
        features = extract_features(data)

        rf_prob  = float(rf.predict_proba(features)[0][1])
        svm_prob = float(svm.predict_proba(scaler.transform(features))[0][1])
        cl_avg   = (rf_prob + svm_prob) / 2
        qs_prob  = get_quantum_prob(features)
        hy_prob  = W_C * cl_avg + W_Q * qs_prob

        pred     = 'PATHOGENIC' if hy_prob > 0.5 else 'BENIGN'
        conf     = hy_prob if hy_prob > 0.5 else (1 - hy_prob)
        expl     = get_shap_explanation(features)

        return jsonify({
            'success': True,
            'prediction': pred,
            'confidence': round(conf * 100, 2),
            'probabilities': {
                'classical': {
                    'random_forest': round(rf_prob * 100, 2),
                    'svm':           round(svm_prob * 100, 2),
                    'average':       round(cl_avg  * 100, 2),
                },
                'quantum': {'qsvm': round(qs_prob * 100, 2)},
                'hybrid':  round(hy_prob * 100, 2),
                'weights': {'classical': round(W_C, 2), 'quantum': round(W_Q, 2)},
            },
            'explanation': expl,
            'input':       data,
            'timestamp':   datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    return jsonify({
        'models': {
            'random_forest': {'accuracy': 88.60, 'f1': 88.76, 'roc_auc': 96.00, 'train_samples': 26153},
            'svm':           {'accuracy': 88.39, 'f1': 87.45, 'roc_auc': 93.97, 'train_samples': 26153},
            'qsvm':          {'accuracy': 86.00, 'f1': 85.17, 'roc_auc': 86.02, 'train_samples': 500},
            'hybrid':        {'accuracy': round(hybrid_cfg.get('accuracy', 0.906) * 100, 2),
                              'f1':       round(hybrid_cfg.get('f1', 0.900) * 100, 2),
                              'roc_auc':  round(hybrid_cfg.get('roc_auc', 0.945) * 100, 2)},
        },
        'hybrid_weights': {'classical': W_C, 'quantum': W_Q},
        'dataset': {'total': 37362, 'brca1': 15476, 'brca2': 21886},
    })

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        df = pd.read_csv(request.files['file'])
        results = []
        for idx, row in df.iterrows():
            inp = {
                'gene':            row.get('gene', 'BRCA1'),
                'position':        row.get('position', 41196312),
                'mutation_type':   row.get('mutation_type', 'missense'),
                'mutation_length': row.get('mutation_length', 1),
                'review_status':   row.get('review_status', 'criteria provided, single submitter'),
                'variant_name':    row.get('variant_name', ''),
            }
            feat   = extract_features(inp)
            rf_p   = float(rf.predict_proba(feat)[0][1])
            svm_p  = float(svm.predict_proba(scaler.transform(feat))[0][1])
            cl_avg = (rf_p + svm_p) / 2
            qs_p   = get_quantum_prob(feat)
            hy     = W_C * cl_avg + W_Q * qs_p
            results.append({
                'index':        int(idx),
                'gene':         inp['gene'],
                'position':     inp['position'],
                'prediction':   'PATHOGENIC' if hy > 0.5 else 'BENIGN',
                'confidence':   round((hy if hy > 0.5 else 1 - hy) * 100, 2),
                'rf_prob':      round(rf_p  * 100, 2),
                'svm_prob':     round(svm_p * 100, 2),
                'qsvm_prob':    round(qs_p  * 100, 2),
                'hybrid_prob':  round(hy    * 100, 2),
            })
        return jsonify({'success': True, 'total_processed': len(results), 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)