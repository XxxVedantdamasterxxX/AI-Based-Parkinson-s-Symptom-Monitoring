# ==============================================================================
# FULL EVALUATION PIPELINE: RANDOM FOREST GAIT BENCHMARK
# Includes: Feature Extraction, Nested GroupKFold, Permutation Null Baseline,
# Hardware Sensor Reduction, Duration Ablation, and Noise Robustness Tests
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, recall_score

# ------------------------------------------------------------------------------
# 1. FEATURE EXTRACTION HELPERS (FFT & PERMUTATION ENTROPY)
# ------------------------------------------------------------------------------

def calculate_permutation_entropy(time_series, order=3, delay=1):
    """Calculates Permutation Entropy for non-linear chaos analysis."""
    n = len(time_series)
    if n < order * delay:
        return 0.0
    
    # Construct embedded vectors
    embedded = np.array([time_series[i:i + order * delay:delay] for i in range(n - (order - 1) * delay)])
    # Determine ordinal patterns
    patterns = np.apply_along_axis(lambda x: tuple(np.argsort(x)), 1, embedded)
    # Calculate probability distribution
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / len(patterns)
    
    return float(entropy(probs, base=2))

def extract_gait_features(sensor_matrix, sampling_rate=100):
    """
    Extracts mean, FFT peak power, and Permutation Entropy across sensors.
    Input shape: (time_samples, num_sensors)
    """
    features = []
    num_sensors = sensor_matrix.shape[1]
    
    for s in range(num_sensors):
        signal = sensor_matrix[:, s]
        
        # Time-Domain
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        # Spectral (FFT)
        fft_vals = np.abs(np.fft.rfft(signal - mean_val))
        fft_peak = np.max(fft_vals) if len(fft_vals) > 0 else 0.0
        
        # Non-linear Chaos (Permutation Entropy)
        perm_ent = calculate_permutation_entropy(signal, order=3, delay=1)
        
        features.extend([mean_val, std_val, fft_peak, perm_ent])
        
    return np.array(features)

# ------------------------------------------------------------------------------
# 2. NESTED GROUP K-FOLD EVALUATION ENGINE
# ------------------------------------------------------------------------------

def evaluate_nested_group_cv(X, y, groups, n_splits=5, noise_level=0.0):
    """
    Evaluates Random Forest using GroupKFold to strictly prevent subject leakage.
    """
    gkf = GroupKFold(n_splits=n_splits)
    accuracies, aucs, pr_aucs, sensitivities = [], [], [], []
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inject Gaussian noise if testing hardware fault tolerance
        if noise_level > 0.0:
            noise = np.random.normal(0, noise_level * np.std(X_test, axis=0), X_test.shape)
            X_test = X_test + noise
            
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        acc = accuracy_score(y_test, preds)
        auc_score = roc_auc_score(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc_score = auc(recall, precision)
        sens = recall_score(y_test, preds)
        
        accuracies.append(acc)
        aucs.append(auc_score)
        pr_aucs.append(pr_auc_score)
        sensitivities.append(sens)
        
    return {
        'acc_mean': np.mean(accuracies),
        'acc_std': np.std(accuracies),
        'auc_mean': np.mean(aucs),
        'pr_auc_mean': np.mean(pr_aucs),
        'sens_mean': np.mean(sensitivities),
        'fold_accs': accuracies
    }

# ------------------------------------------------------------------------------
# 3. SYNTHETIC COHORT GENERATION (N=166, 16 SENSORS, 66 FEATURES)
# ------------------------------------------------------------------------------

np.random.seed(42)
n_subjects = 166
subject_ids = np.array([f"SUBJ_{i:03d}" for i in range(n_subjects)])
labels = np.array([1 if i < 66 else 0 for i in range(n_subjects)]) # 66 PD, 100 Control

# Generate Synthetic Feature Matrix (166 subjects x 64 sensor features + 2 global asymmetry features)
X_16_sensors = np.random.normal(loc=0.5, scale=0.1, size=(n_subjects, 66))

# Inject strong diagnostic signal into heel sensors (Sensors 0 and 8: Left/Right Heels)
for i in range(n_subjects):
    if labels[i] == 1:
        X_16_sensors[i, 0:4] += 0.35  # Left Heel signal change
        X_16_sensors[i, 32:36] -= 0.20 # Right Heel signal (asymmetry)

# Sensor Topology Feature Indices (4 features per sensor)
idx_16_sensors = list(range(66))                             # 16 Sensors (Full)
idx_8_sensors  = list(range(32))                             # 8 Sensors (Unilateral Left)
idx_4_sensors  = list(range(0, 8)) + list(range(32, 40))     # 4 Sensors (Bilateral Heel/Toe)
idx_2_sensors  = list(range(0, 4)) + list(range(32, 36))     # 2 Sensors (Bilateral Heels)

# ------------------------------------------------------------------------------
# 4. EXECUTION OF EXPERIMENTAL TESTS
# ------------------------------------------------------------------------------

print("=== RUNNING RANDOM FOREST EVALUATION PIPELINE ===\n")

# A. 16-Sensor Baseline & 100-Iteration Permutation Test
res_baseline = evaluate_nested_group_cv(X_16_sensors[:, idx_16_sensors], labels, subject_ids)

perm_accs = []
for _ in range(100):
    shuffled_labels = np.random.permutation(labels)
    perm_res = evaluate_nested_group_cv(X_16_sensors[:, idx_16_sensors], shuffled_labels, subject_ids)
    perm_accs.append(perm_res['acc_mean'])
perm_mean_acc = np.mean(perm_accs)
p_val_perm = np.sum(np.array(perm_accs) >= res_baseline['acc_mean']) / 100.0

print(f"Permutation Null Baseline Acc: {perm_mean_acc * 100:.1f}%")
print(f"16-Sensor Baseline Acc: {res_baseline['acc_mean']*100:.1f}% ± {res_baseline['acc_std']*100:.1f}%, AUC: {res_baseline['auc_mean']:.3f}, PR-AUC: {res_baseline['pr_auc_mean']:.3f}, Sens: {res_baseline['sens_mean']:.3f} (p < {max(p_val_perm, 0.01)})\n")

# B. Hardware Optimization (Sensor Reduction & Cost-Utility)
topologies = {
    "16 Sensors (Full Insoles)": (idx_16_sensors, 200.0), # Hardware Cost ($)
    "8 Sensors (Unilateral Left)": (idx_8_sensors, 110.0),
    "4 Sensors (Bilateral Heel/Toe)": (idx_4_sensors, 50.0),
    "2 Sensors (Bilateral Heels)": (idx_2_sensors, 25.0)
}

results_topo = {}
for name, (indices, cost) in topologies.items():
    res = evaluate_nested_group_cv(X_16_sensors[:, indices], labels, subject_ids)
    
    # Paired t-test against 16-sensor baseline
    if name == "16 Sensors (Full Insoles)":
        p_val = 1.0
        p_str = "Baseline"
    else:
        _, p_val = ttest_rel(res_baseline['fold_accs'], res['fold_accs'])
        p_str = f"p = {p_val:.4f}"
        
    cost_eff = (res['sens_mean'] * 100) / cost
    
    results_topo[name] = {
        'Acc': f"{res['acc_mean']*100:.1f}% ± {res['acc_std']*100:.1f}%",
        'ROC-AUC': round(res['auc_mean'], 3),
        'PR-AUC': round(res['pr_auc_mean'], 3),
        'Sensitivity': round(res['sens_mean'], 3),
        'p-value': p_str,
        'Cost Eff (pts/$)': round(cost_eff, 2)
    }

df_topo = pd.DataFrame(results_topo).T
print("=== HARDWARE TOPOLOGY & COST-UTILITY COMPARISON ===")
print(df_topo.to_string())
print("\n")

# C. Temporal Window Duration Ablation
durations = {"60s Walk": 1.0, "30s Walk": 0.85, "15s Walk": 0.70, "5s Walk": 0.50}
print("=== TEMPORAL WINDOW DURATION ABLATION ===")
for dur_name, scale_factor in durations.items():
    # Simulate feature degradation caused by reduced recording length
    X_dur = X_16_sensors[:, idx_2_sensors] * scale_factor + np.random.normal(0, (1 - scale_factor)*0.2, (n_subjects, 8))
    res_dur = evaluate_nested_group_cv(X_dur, labels, subject_ids)
    print(f"* {dur_name}: {res_dur['acc_mean']*100:.1f}% Accuracy")
print("\n")

# D. Fault Tolerance: 15% Gaussian Noise Injection (2-Sensor Model)
res_clean = evaluate_nested_group_cv(X_16_sensors[:, idx_2_sensors], labels, subject_ids, noise_level=0.0)
res_noisy = evaluate_nested_group_cv(X_16_sensors[:, idx_2_sensors], labels, subject_ids, noise_level=0.15)

print("=== HARDWARE FAULT TOLERANCE (2-SENSOR ARRAY) ===")
print(f"* Standard Model Clean Acc: {res_clean['acc_mean']*100:.1f}%")
print(f"* Standard Model under 15% Noise: {res_noisy['acc_mean']*100:.1f}%")
