"""
QGene - Genetic Mutation Prediction System
Flask Backend Application
"""

from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Mock model predictions (replace with actual trained models later)
class MockModel:
    def predict_proba(self, X):
        """Mock prediction - returns random probabilities"""
        # In real implementation, this will use actual trained models
        np.random.seed(hash(str(X[0])) % 2**32)  # Deterministic based on input
        prob = np.random.uniform(0.3, 0.95)
        return np.array([[1-prob, prob]])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array([1 if proba[0][1] > 0.5 else 0])

# Initialize mock models
classical_rf_model = MockModel()
classical_svm_model = MockModel()
quantum_vqc_model = MockModel()
quantum_qsvm_model = MockModel()

def extract_features(input_data):
    """
    Convert user input to feature vector
    Features: position, mutation_type_encoded, conservation_score, allele_frequency
    """
    # Encode mutation type
    mutation_type_map = {
        'Missense': 1,
        'Nonsense': 2,
        'Frameshift': 3,
        'Silent': 4,
        'Splice Site': 5
    }
    
    # Create feature vector
    features = [
        float(input_data.get('position', 0)),
        mutation_type_map.get(input_data.get('mutation_type', 'Missense'), 1),
        float(input_data.get('conservation_score', 0.5)),
        float(input_data.get('allele_frequency', 0.001))
    ]
    
    return np.array(features).reshape(1, -1)

def get_shap_explanation(features, prediction):
    """
    Mock SHAP explanation (replace with actual SHAP values later)
    """
    feature_names = ['Position', 'Mutation Type', 'Conservation Score', 'Allele Frequency']
    
    # Mock contributions (in real implementation, use actual SHAP)
    contributions = np.random.uniform(-0.2, 0.3, 4)
    if prediction < 0.5:
        contributions = -np.abs(contributions)
    else:
        contributions = np.abs(contributions)
    
    explanation = []
    for name, contrib, value in zip(feature_names, contributions, features[0]):
        explanation.append({
            'feature': name,
            'value': float(value),
            'contribution': float(contrib),
            'impact': 'increases' if contrib > 0 else 'decreases'
        })
    
    return explanation

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get input data
        data = request.json
        
        # Extract features
        features = extract_features(data)
        
        # Classical predictions
        rf_prob = classical_rf_model.predict_proba(features)[0][1]
        svm_prob = classical_svm_model.predict_proba(features)[0][1]
        classical_avg = (rf_prob + svm_prob) / 2
        
        # Quantum predictions (on reduced feature set for simulation)
        vqc_prob = quantum_vqc_model.predict_proba(features)[0][1]
        qsvm_prob = quantum_qsvm_model.predict_proba(features)[0][1]
        quantum_avg = (vqc_prob + qsvm_prob) / 2
        
        # Hybrid ensemble (70% classical, 30% quantum)
        hybrid_prob = 0.7 * classical_avg + 0.3 * quantum_avg
        
        # Determine final prediction
        final_prediction = "PATHOGENIC" if hybrid_prob > 0.5 else "BENIGN"
        confidence = hybrid_prob if hybrid_prob > 0.5 else (1 - hybrid_prob)
        
        # Get explanation
        explanation = get_shap_explanation(features, hybrid_prob)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': final_prediction,
            'confidence': float(confidence * 100),
            'probabilities': {
                'classical': {
                    'random_forest': float(rf_prob * 100),
                    'svm': float(svm_prob * 100),
                    'average': float(classical_avg * 100)
                },
                'quantum': {
                    'vqc': float(vqc_prob * 100),
                    'qsvm': float(qsvm_prob * 100),
                    'average': float(quantum_avg * 100)
                },
                'hybrid': float(hybrid_prob * 100)
            },
            'explanation': explanation,
            'input': data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        results = []
        for idx, row in df.iterrows():
            input_data = {
                'gene': row.get('gene', 'BRCA1'),
                'position': row.get('position', 0),
                'mutation_type': row.get('mutation_type', 'Missense'),
                'aa_change': row.get('aa_change', ''),
                'conservation_score': row.get('conservation_score', 0.5),
                'allele_frequency': row.get('allele_frequency', 0.001)
            }
            
            features = extract_features(input_data)
            
            # Quick predictions
            rf_prob = classical_rf_model.predict_proba(features)[0][1]
            hybrid_prob = rf_prob  # Simplified for batch
            
            results.append({
                'index': int(idx),
                'gene': input_data['gene'],
                'position': input_data['position'],
                'prediction': "PATHOGENIC" if hybrid_prob > 0.5 else "BENIGN",
                'confidence': float(hybrid_prob * 100) if hybrid_prob > 0.5 else float((1-hybrid_prob) * 100)
            })
        
        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/download-sample/<sample_type>')
def download_sample(sample_type):
    """Provide sample input files"""
    samples_dir = '/home/claude/sample_inputs'
    
    file_map = {
        'single_pathogenic': 'sample_pathogenic.csv',
        'single_benign': 'sample_benign.csv',
        'batch_mixed': 'sample_batch_mixed.csv',
        'batch_brca1': 'sample_batch_brca1.csv',
        'batch_brca2': 'sample_batch_brca2.csv'
    }
    
    if sample_type in file_map:
        filepath = os.path.join(samples_dir, file_map[sample_type])
        return send_file(filepath, as_attachment=True)
    else:
        return "Sample not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
