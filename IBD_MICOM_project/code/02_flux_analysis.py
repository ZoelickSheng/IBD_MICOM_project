"""
Nested Cross-Validation and Disease Classification Benchmark Pipeline
======================================================================
This script implements a rigorous machine learning framework to benchmark
MICOM-derived metabolic reaction fluxes against taxonomic relative abundances
for classifying Inflammatory Bowel Disease (IBD: CD/UC) versus healthy controls.

To prevent data leakage during supervised feature selection, the evaluation
employs a nested cross-validation scheme:
  - Outer Loop: Stratified 5-fold cross-validation for unbiased generalisation error estimation.
  - Inner Loop: Stratified 4-fold cross-validation for tuning the number of top-k
    features selected via Mutual Information (MI).
  - Classifier: Random Forest (n_estimators=300, max_depth=5, class_weight='balanced').

Outputs:
  - Outer fold performance metrics (Accuracy, ROC-AUC) for both representations.
  - Feature selection stability across folds (consensus feature registry).
  - Cohort-wide feature importances for downstream mechanistic prioritisation.
"""

from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# ==============================================================================
# 1. Directory and Path Configuration (Relative Paths)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["notebooks", "scripts", "notebooks two"] else SCRIPT_DIR

DATA_DIR = BASE_DIR / "data"
FLUX_DIR = BASE_DIR / "flux_output"
FLUX_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX_PATH = FLUX_DIR / "A2_machine_learning_input_X_community.csv"
RAW_ABUNDANCE_PATH = DATA_DIR / "ibd_taxa.csv"
METADATA_PATH = DATA_DIR / "ibd_metadata+(1).csv"
COVERAGE_PATH = DATA_DIR / "patient_coverage.xlsx"

CONSENSUS_PATH = FLUX_DIR / "A3_consensus_features_nested_cv.csv"
GLOBAL_IMPORTANCES_PATH = FLUX_DIR / "A3_rf_feature_importances_MICOM.csv"

print("[INFO] Initialising nested cross-validation machine learning pipeline...\n")

# ==============================================================================
# 2. Quality Control & Taxonomic Coverage Thresholding (> 85%)
# ==============================================================================
if not COVERAGE_PATH.exists():
    raise FileNotFoundError(f"[ERROR] Coverage metadata file not found at: {COVERAGE_PATH}")

df_cov = pd.read_excel(COVERAGE_PATH)
if df_cov.iloc[:, 1].max() <= 1.0:
    df_cov.iloc[:, 1] = df_cov.iloc[:, 1] * 100

high_quality_samples = df_cov[df_cov.iloc[:, 1] > 85.0].iloc[:, 0].astype(str).str.strip().values
print(f"[INFO] Applied taxonomic coverage filter: {len(high_quality_samples)} subjects passed >= 85.0% threshold.")

# ==============================================================================
# 3. Clinical Metadata Ingestion and Class Label Assignment
# ==============================================================================
meta_df = pd.read_csv(METADATA_PATH, index_col=0)
meta_df.index = meta_df.index.astype(str).str.strip()
meta_df.columns = [col.strip().lower() for col in meta_df.columns]

target_col = next((c for c in meta_df.columns if "diag" in c), None)
if target_col is None:
    raise KeyError("[ERROR] Diagnostic column containing 'diag' not found in clinical metadata.")

# ==============================================================================
# 4. Feature Matrix Alignment (Fluxes & Taxonomic Abundances)
# ==============================================================================
X_micom_raw = pd.read_csv(FEATURE_MATRIX_PATH, index_col=0)
X_micom_raw.index = X_micom_raw.index.astype(str).str.strip()

df_taxa = pd.read_csv(RAW_ABUNDANCE_PATH, index_col=0)
X_taxa_raw = df_taxa.T.fillna(0.0)
X_taxa_raw.index = X_taxa_raw.index.astype(str).str.strip()

# Restrict to intersection of samples across all four input sources
final_sample_pool = [
    s for s in high_quality_samples
    if (s in meta_df.index) and (s in X_micom_raw.index) and (s in X_taxa_raw.index)
]

y_list, confirmed_samples = [], []
for sample_id in final_sample_pool:
    diag_val = str(meta_df.loc[sample_id, target_col]).strip().upper()
    if diag_val in ["CD", "UC", "IBD"]:
        y_list.append(1)
        confirmed_samples.append(sample_id)
    elif diag_val in ["CONTROL", "HEALTHY", "HC"]:
        y_list.append(0)
        confirmed_samples.append(sample_id)

X_micom = X_micom_raw.loc[confirmed_samples]
X_taxa = X_taxa_raw.loc[confirmed_samples]
y = np.array(y_list)

n_ibd = sum(y == 1)
n_control = sum(y == 0)

print(f"[INFO] Cohort alignment completed. Final sample size: N = {len(y)} (IBD: {n_ibd}, Control: {n_control})\n")

# ==============================================================================
# 5. Nested Cross-Validation (Outer 5-Fold, Inner 4-Fold CV)
# ==============================================================================
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Candidate top-k features to evaluate during inner loop hyperparameter tuning
param_grid = {"selector__k": [30, 40, 50, 100, 150, 200, 250, 300, 400]}

# ----------------- Track 1: MICOM Metabolic Flux Representation -----------------
pipeline_micom = Pipeline([
    ("selector", SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=22))),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=22, class_weight="balanced"))
])

micom_acc_scores = []
micom_auc_scores = []
fold_selected_features = []
best_k_list = []

print("[INFO] Executing nested cross-validation for MICOM metabolic flux models...")
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_micom, y), start=1):
    X_train_out, X_test_out = X_micom.iloc[train_idx], X_micom.iloc[test_idx]
    y_train_out, y_test_out = y[train_idx], y[test_idx]

    # Inner CV loop: Optimize parameter k on the training fold strictly
    grid = GridSearchCV(pipeline_micom, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train_out, y_train_out)

    best_k = grid.best_params_["selector__k"]
    best_k_list.append(best_k)

    # Outer test fold evaluation using best estimator
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_out)
    y_prob = best_model.predict_proba(X_test_out)[:, 1]

    acc = accuracy_score(y_test_out, y_pred)
    auc = roc_auc_score(y_test_out, y_prob)
    micom_acc_scores.append(acc)
    micom_auc_scores.append(auc)

    # Extract selected feature indices for consensus analysis
    selected_mask = best_model.named_steps["selector"].get_support()
    fold_feats = X_micom.columns[selected_mask].tolist()
    fold_selected_features.append(set(fold_feats))

    print(f"       Outer Fold {fold}: Optimal k = {best_k:3d} | Test Accuracy = {acc:.4f} | Test ROC-AUC = {auc:.4f}")

# ----------------- Track 2: Taxonomic Relative Abundance Baseline -----------------
print("\n[INFO] Executing cross-validation for taxonomic abundance baseline models...")
rf_taxa = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=22, class_weight="balanced")
taxa_acc_scores = []
taxa_auc_scores = []

for train_idx, test_idx in outer_cv.split(X_taxa, y):
    X_train_t, X_test_t = X_taxa.iloc[train_idx], X_taxa.iloc[test_idx]
    y_train_t, y_test_t = y[train_idx], y[test_idx]

    rf_taxa.fit(X_train_t, y_train_t)
    y_pred_t = rf_taxa.predict(X_test_t)
    y_prob_t = rf_taxa.predict_proba(X_test_t)[:, 1]

    taxa_acc_scores.append(accuracy_score(y_test_t, y_pred_t))
    taxa_auc_scores.append(roc_auc_score(y_test_t, y_prob_t))

# ==============================================================================
# 6. Benchmark Evaluation Summary
# ==============================================================================
print("\n" + "=" * 85)
print(f"Classification Benchmark: Nested Cross-Validation Results (N = {len(y)})")
print("=" * 85)
print(f" Metric                  | MICOM Metabolic Flux (Nested) | Taxonomic Relative Abundance")
print("-" * 85)
print(f" Accuracy (Mean ± SD)    | {np.mean(micom_acc_scores):.4f} ± {np.std(micom_acc_scores):.4f}                | {np.mean(taxa_acc_scores):.4f} ± {np.std(taxa_acc_scores):.4f}")
print(f" ROC-AUC  (Mean ± SD)    | {np.mean(micom_auc_scores):.4f} ± {np.std(micom_auc_scores):.4f}                | {np.mean(taxa_auc_scores):.4f} ± {np.std(taxa_auc_scores):.4f}")
print("=" * 85)

# ==============================================================================
# 7. Feature Selection Stability and Consensus Analysis
# ==============================================================================
all_selected_feats = [f for s in fold_selected_features for f in s]
feat_counts = Counter(all_selected_feats)

consensus_df = pd.DataFrame([
    {
        "Reaction_ID": feat,
        "Selected_Folds_Count": count,
        "Selection_Frequency_%": (count / 5) * 100.0
    }
    for feat, count in feat_counts.items()
]).sort_values(by="Selected_Folds_Count", ascending=False)

consensus_df.to_csv(CONSENSUS_PATH, index=False)

high_consensus = consensus_df[consensus_df["Selected_Folds_Count"] == 5]
print(f"\n[INFO] Feature Selection Stability:")
print(f"       - Highly consensus features (selected in 5/5 outer folds): {len(high_consensus)}")
print(f"       - Consensus registry exported: {CONSENSUS_PATH.name}")

# ==============================================================================
# 8. Full-Cohort Model Fitting for Downstream Feature Prioritisation
# ==============================================================================
print("\n[INFO] Fitting full-cohort model for downstream mechanistic interpretation...")
final_k = Counter(best_k_list).most_common(1)[0][0]

final_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=22), k=final_k)),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=22, class_weight="balanced"))
])
final_pipeline.fit(X_micom, y)

selected_cols = X_micom.columns[final_pipeline.named_steps["selector"].get_support()]
importances = final_pipeline.named_steps["rf"].feature_importances_
global_feat_df = pd.Series(importances, index=selected_cols).sort_values(ascending=False)
global_feat_df.to_csv(GLOBAL_IMPORTANCES_PATH)

print(f"[INFO] Cohort-wide feature importances exported (k = {final_k}): {GLOBAL_IMPORTANCES_PATH.name}")
print("[INFO] Machine learning benchmarking completed successfully.")