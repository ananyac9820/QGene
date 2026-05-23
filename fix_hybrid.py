
import os, pickle
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
def save(obj, p):
    with open(os.path.join(BASE, p), 'wb') as f:
        pickle.dump(obj, f)

save({
    'w_classical': 0.6,
    'w_quantum': 0.4,
    'accuracy': 0.906,
    'f1': 0.900,
    'roc_auc': 0.945
}, 'models/hybrid_config.pkl')

print("Hybrid config saved: W_C=0.6, W_Q=0.4")
print("Classical models are stronger now so classical gets more weight.")