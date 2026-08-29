"""
Task 8  - label-quality ablation: RF without review_score and is_expert.
Task 10 - environment facts for the paper's Table 3.
"""

import os, sys, json, platform, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, SEED, FEATURES, FEATURES_ABLATION, ABLATION_DROP, load_splits

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("=" * 72)
print("TASK 8 - label-quality ablation")
print("=" * 72)
print("dropped: %s" % ABLATION_DROP)
print("features: %d -> %d" % (len(FEATURES), len(FEATURES_ABLATION)))

full = load_splits()
abl = load_splits(features=FEATURES_ABLATION)

# same rows in every split (the split is driven by the same seed and stratify)
assert np.array_equal(full["y_train"], abl["y_train"]), "ablation split differs from full split"
assert np.array_equal(full["y_test"], abl["y_test"]), "ablation test split differs"
assert list(full["X_train_df"].index) == list(abl["X_train_df"].index), "row identity differs"
print("ASSERT PASS: ablation uses the identical rows, only the columns differ.\n")


def run_rf(d, tag):
    rf = RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_leaf=1,
                                class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(d["X_train"], d["y_train"])
    pred = rf.predict(d["X_test"])
    prob = rf.predict_proba(d["X_test"])[:, 1]
    r = dict(acc=accuracy_score(d["y_test"], pred),
             f1=f1_score(d["y_test"], pred),
             auc=roc_auc_score(d["y_test"], prob),
             n_features=len(d["features"]))
    print("%-28s (%2d feat) acc %6.2f%%  F1 %.4f  AUC %.4f"
          % (tag, r['n_features'], r['acc'] * 100, r['f1'], r['auc']))
    return r


r_full = run_rf(full, "RF full feature set")
r_abl = run_rf(abl, "RF ablated (no curation)")
delta = (r_abl['acc'] - r_full['acc']) * 100
print("\ndelta: %+.2f accuracy points when curation-confidence features are removed" % delta)
print("stated lower bound (ablated RF accuracy): %.2f%%" % (r_abl['acc'] * 100))

print("\n" + "=" * 72)
print("TASK 10 - environment facts (Table 3)")
print("=" * 72)

WANT = ['scikit-learn', 'numpy', 'pandas', 'scipy', 'qiskit', 'qiskit-aer',
        'qiskit-machine-learning', 'shap', 'flask']
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                        capture_output=True, text=True).stdout.splitlines()
libs = {}
for line in freeze:
    if '==' in line:
        name, ver = line.split('==', 1)
        if name.strip().lower() in WANT:
            libs[name.strip()] = ver.strip()
print("\nlibraries:")
for k in WANT:
    hit = next((n for n in libs if n.lower() == k), None)
    print("  %-26s %s" % (k, libs[hit] if hit else "NOT INSTALLED"))

pyver = platform.python_version()
print("\npython: %s (%s)" % (pyver, platform.python_implementation()))
print("platform: %s" % platform.platform())

hw = {}
try:
    import psutil
    hw['physical_cores'] = psutil.cpu_count(logical=False)
    hw['logical_cores'] = psutil.cpu_count(logical=True)
    hw['ram_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
except Exception as e:
    print("psutil unavailable: %s" % e)

cpu_name = None
try:
    out = subprocess.run(
        ['powershell', '-NoProfile', '-Command',
         "(Get-CimInstance Win32_Processor).Name"],
        capture_output=True, text=True, timeout=60).stdout.strip()
    cpu_name = out.splitlines()[0].strip() if out else None
except Exception as e:
    print("Win32_Processor query failed: %s" % e)
hw['cpu'] = cpu_name or platform.processor() or "UNKNOWN"

print("\nhardware:")
print("  CPU             : %s" % hw['cpu'])
print("  physical cores  : %s" % hw.get('physical_cores', 'UNKNOWN'))
print("  logical cores   : %s" % hw.get('logical_cores', 'UNKNOWN'))
print("  RAM (GB)        : %s" % hw.get('ram_gb', 'UNKNOWN'))
print("\nseed used everywhere: %d" % SEED)

json.dump({'ablation': {'dropped': ABLATION_DROP, 'full': r_full, 'ablated': r_abl,
                        'delta_acc_points': delta},
           'env': {'libraries': libs, 'python': pyver,
                   'platform': platform.platform(), 'hardware': hw, 'seed': SEED}},
          open(os.path.join(BASE, 'results', 'ablation_env.json'), 'w'), indent=2)
print("\nwrote results/ablation_env.json")
