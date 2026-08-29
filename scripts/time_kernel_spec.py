"""Spec-faithful wall-clock: FidelityQuantumKernel builds the 500x500 training kernel."""
import os, sys, time, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, load_splits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

d = load_splits()
pca = PCA(n_components=4, random_state=SEED)
Xtr = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(pca.fit_transform(d["X_train_s"]))
idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=500, random_state=SEED)
              .split(np.zeros(len(d["y_train"])), d["y_train"]))
Xs = Xtr[np.sort(idx)]

fm = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='full')
k = FidelityQuantumKernel(feature_map=fm)
print("timing FidelityQuantumKernel.evaluate on 500x500 (124,750 unique pairs)...", flush=True)
t = time.perf_counter(); K = k.evaluate(Xs); dt = time.perf_counter() - t
print(f"wall-clock: {dt:.1f}s = {int(dt//3600):02d}:{int(dt%3600//60):02d} (hh:mm)")
print(f"per-pair  : {dt/124750*1000:.3f} ms")
print(f"K shape {K.shape}, symmetric {np.allclose(K,K.T)}, diag~1 {np.allclose(np.diag(K),1)}")
json.dump({'fidelity_quantum_kernel_500x500_seconds': dt,
           'unique_pairs': 124750, 'ms_per_pair': dt/124750*1000},
          open(os.path.join(BASE,'results','kernel_timing.json'),'w'), indent=2)
print("wrote results/kernel_timing.json")
