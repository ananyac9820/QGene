"""
QGene - Genetic Mutation Prediction System
Flask Backend
"""
from flask import Flask, render_template, request, jsonify, send_file
import pickle, numpy as np, pandas as pd
from datetime import datetime
import os

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
qkernel    = load('models/quantum_kernel_zz.pkl')
hybrid_cfg = load('models/hybrid_config.pkl')
X_qtrain   = np.load(os.path.join(BASE, 'data/X_qsvm_test.npy'))

W_C = hybrid_cfg['w_classical']
W_Q = hybrid_cfg['w_quantum']
print(f"✅ Models loaded — Hybrid weights: Classical {W_C:.2f} / Quantum {W_Q:.2f}")

GENE_MAP = {'BRCA1': 0, 'BRCA2': 1}
TYPE_MAP = {
    'single nucleotide variant': 0, 'deletion': 1, 'duplication': 2,
    'insertion': 3, 'indel': 4,
    'missense': 0, 'nonsense': 1, 'frameshift': 3, 'silent': 0, 'splice site': 2,
}

def extract_features(data):
    gene   = GENE_MAP.get(str(data.get('gene','BRCA1')).upper(), 0)
    mtype  = TYPE_MAP.get(str(data.get('mutation_type','missense')).lower(), 0)
    pos    = float(data.get('position', 41196312))
    length = float(data.get('mutation_length', 1))
    return np.array([[gene, mtype, pos, length]], dtype=float)

def get_quantum_prob(features_4d):
    features_pca = pca.transform(features_4d)
    K_single     = qkernel.evaluate(features_pca, X_qtrain)
    return float(qsvm.predict_proba(K_single)[0][1])

def get_shap_explanation(features_4d):
    try:
        import shap
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(features_4d)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        names = ['Gene', 'Mutation Type', 'Position', 'Mutation Length']
        expl = [{'feature': n, 'value': float(v), 'contribution': float(c),
                 'impact': 'increases risk' if c > 0 else 'decreases risk'}
                for n, v, c in zip(names, features_4d[0], sv)]
        return sorted(expl, key=lambda x: abs(x['contribution']), reverse=True)
    except Exception as e:
        names = ['Gene', 'Mutation Type', 'Position', 'Mutation Length']
        return [{'feature': n, 'value': float(v), 'contribution': 0.0, 'impact': 'unknown'}
                for n, v in zip(names, features_4d[0])]

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
        svm_prob = float(svm.predict_proba(features)[0][1])
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
            'random_forest': {'accuracy': 87.71, 'f1': 87.13, 'roc_auc': 92.73, 'train_samples': 26153},
            'svm':           {'accuracy': 86.23, 'f1': 85.38, 'roc_auc': 86.60, 'train_samples': 26153},
            'qsvm':          {'accuracy': 86.00, 'f1': 85.17, 'roc_auc': 86.02, 'train_samples': 500},
            'hybrid':        {'accuracy': round(hybrid_cfg['accuracy']*100,2),
                              'f1':       round(hybrid_cfg['f1']*100,2),
                              'roc_auc':  round(hybrid_cfg['roc_auc']*100,2)},
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
            inp = {'gene': row.get('gene','BRCA1'), 'position': row.get('position',41196312),
                   'mutation_type': row.get('mutation_type','missense'), 'mutation_length': row.get('mutation_length',1)}
            feat   = extract_features(inp)
            rf_p   = float(rf.predict_proba(feat)[0][1])
            qs_p   = get_quantum_prob(feat)
            hy     = W_C * rf_p + W_Q * qs_p
            results.append({'index': int(idx), 'gene': inp['gene'], 'position': inp['position'],
                             'prediction': 'PATHOGENIC' if hy > 0.5 else 'BENIGN',
                             'confidence': round((hy if hy > 0.5 else 1-hy)*100, 2),
                             'rf_prob': round(rf_p*100,2), 'qsvm_prob': round(qs_p*100,2),
                             'hybrid_prob': round(hy*100,2)})
        return jsonify({'success': True, 'total_processed': len(results), 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
