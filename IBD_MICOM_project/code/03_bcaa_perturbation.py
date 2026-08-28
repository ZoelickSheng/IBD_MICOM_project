"""
In Silico BCAA Perturbation and Personalized Metabolic Response Pipeline
========================================================================
This script executes a cohort-wide constraint-based metabolic perturbation
experiment using AGORA2 reconstructions and the MICOM framework. It simulates
the metabolic consequences of dietary Branched-Chain Amino Acid (BCAA; leucine,
isoleucine, valine) availability gradients (1x Baseline, 2x Moderate, 10x Overdrive)
on a quality-controlled IBD and Control cohort (taxonomic model coverage > 85%).

The cohort is stratified by the colonization status of Dorea formicigenerans
(Track A: Present vs. Track B: Absent) to evaluate substrate-dependent
and species-specific metabolic capability, specifically tracking the flux of
(S)-3-methyl-2-oxopentanoate:lipoamide oxidoreductase (3MOPLPAMO) and
community-level biomass growth stability.

Features:
  - Dynamic relative path resolution compatible with standard repository layouts.
  - Hybrid model ingestion supporting native SBML (.xml) and fallback MATLAB (.mat).
  - Robust checkpointing and incremental caching to avoid redundant FBA solves.
  - Abundance-weighted medium scaling with strict anaerobic constraints.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from cobra.io import read_sbml_model
from micom import Community

# ==============================================================================
# 1. Directory and Path Configuration (Relative Paths)
# ==============================================================================
# Resolve base directory dynamically relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["notebooks", "scripts", "notebooks two"] else SCRIPT_DIR

# Input model and data directories
MODEL_DIR = BASE_DIR / "models" / "AGORA2_SBML"
MAT_MODEL_DIR = BASE_DIR / "models" / "matlab model"
DATA_DIR = BASE_DIR / "data"

MAPPING_FILE = DATA_DIR / "mapping_summary.xlsx"
ABUNDANCE_FILE = DATA_DIR / "ibd_taxa.csv"
MEDIUM_FILE = DATA_DIR / "fluxes.tsv"
METADATA_FILE = DATA_DIR / "ibd_metadata+(1).csv"
COVERAGE_FILE = DATA_DIR / "patient_coverage.xlsx"

# Results and output paths
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Historical cache sources for incremental checkpointing
PREV_20_CSV = RESULTS_DIR / "A7_1_Sim01_BCAA_flux_results.csv"
PREV_40_CSV = RESULTS_DIR / "A7_1_Sim01_BCAA_flux_results_40.csv"
OUTPUT_FULL_CSV = RESULTS_DIR / "A7_1_Sim01_BCAA_flux_results_FULL_COHORT_85.csv"
SAMPLES_INFO_CSV = RESULTS_DIR / "Sim01_BCAA_stratified_samples_info_FULL_85.csv"

print("[INFO] Initialising cohort-wide in silico BCAA dietary perturbation FBA pipeline (Coverage > 85%)...\n")

# ==============================================================================
# 2. Quality Control & Cohort Stratification (Coverage > 85%)
# ==============================================================================
# 2a. Taxonomic coverage thresholding (> 85%)
df_cov = pd.read_excel(COVERAGE_FILE)
if df_cov.iloc[:, 1].max() <= 1.0:
    df_cov.iloc[:, 1] = df_cov.iloc[:, 1] * 100
valid_cov_samples = set(df_cov[df_cov.iloc[:, 1] > 85.0].iloc[:, 0].astype(str).str.strip())

# 2b. Clinical metadata ingestion
meta_df = pd.read_csv(METADATA_FILE, index_col=0)
meta_df.index = meta_df.index.astype(str).str.strip()
diag_col = next(c for c in meta_df.columns if "diag" in c.lower())

# 2c. Species abundance ingestion and Dorea formicigenerans profiling
abundance_df = pd.read_csv(ABUNDANCE_FILE)
taxa_col = abundance_df.columns[0]

dorea_row = abundance_df[
    abundance_df[taxa_col].str.contains(r"Dorea_formicigenerans|Dorea formicigenerans", case=False, na=False)
]
if dorea_row.empty:
    raise ValueError("[ERROR] Target species 'Dorea formicigenerans' not identified in abundance profile.")

# Filter cohort to include only QC-passed samples with valid metadata
full_cohort_samples = [
    s for s in abundance_df.columns[1:]
    if (s in valid_cov_samples) and (s in meta_df.index)
]

sample_list = []
for s in full_cohort_samples:
    diag_raw = str(meta_df.loc[s, diag_col]).strip().upper()
    if diag_raw in ["CONTROL", "HEALTHY", "HC"]:
        clinical_group = "Control"
    elif diag_raw in ["CD", "UC", "IBD"]:
        clinical_group = "IBD"
    else:
        continue

    d_ab = float(dorea_row[s].values[0]) if s in abundance_df.columns else 0.0
    strat_group = "Track A (Dorea Present)" if d_ab > 0.0 else "Track B (Dorea Absent)"

    sample_list.append({
        "Sample_ID": s,
        "Clinical_Diagnosis": clinical_group,
        "Stratification_Group": strat_group,
        "Dorea_Abundance": d_ab
    })

df_stratified = pd.DataFrame(sample_list)
df_stratified.to_csv(SAMPLES_INFO_CSV, index=False)

n_track_a = sum(df_stratified["Stratification_Group"] == "Track A (Dorea Present)")
n_track_b = sum(df_stratified["Stratification_Group"] == "Track B (Dorea Absent)")

print(f"[INFO] Cohort QC completed. Eligible subjects: N = {len(df_stratified)}")
print(f"       - Track A (Dorea-present): n = {n_track_a}")
print(f"       - Track B (Dorea-absent):  n = {n_track_b}\n")

# ==============================================================================
# 3. Incremental Checkpointing & Cache Recovery
# ==============================================================================
cached_results = {}
candidate_sources = [OUTPUT_FULL_CSV, PREV_40_CSV, PREV_20_CSV]

for src in candidate_sources:
    if src.exists():
        try:
            df_old = pd.read_csv(src)
            for sid, group in df_old.groupby("Sample_ID"):
                # Require all three dietary gradient conditions for a valid checkpoint
                if len(group) >= 3 and sid not in cached_results:
                    cached_results[sid] = group.to_dict("records")
        except Exception:
            pass

print(f"[INFO] Recovered {len(cached_results)} fully simulated profiles from previous checkpoints.")
remaining_samples = [s for s in df_stratified["Sample_ID"] if s not in cached_results]
print(f"[INFO] Samples scheduled for FBA simulation: N = {len(remaining_samples)}\n")

# Initialise final results container with verified cached subjects
final_simulation_results = []
for sid in cached_results:
    if sid in set(df_stratified["Sample_ID"]):
        final_simulation_results.extend(cached_results[sid])

# ==============================================================================
# 4. Hybrid Model Routing (SBML XML & MATLAB Matrix Compatibility)
# ==============================================================================
mapping_df = pd.read_excel(MAPPING_FILE)
mapped_taxa = mapping_df[mapping_df["Status"] == "Matched"]["Excel_Name"]
filtered_abundance_df = abundance_df[abundance_df[taxa_col].isin(mapped_taxa)].copy()

valid_models_paths = {}
for taxon in mapped_taxa:
    try:
        file_match = mapping_df[mapping_df["Excel_Name"] == taxon]["AGORA2_File"].values[0]
        sbml_path = MODEL_DIR / file_match

        # Route A: Validate SBML format integrity
        if sbml_path.exists():
            try:
                _ = read_sbml_model(str(sbml_path))
                valid_models_paths[taxon] = str(sbml_path)
                continue
            except Exception:
                pass

        # Route B: Fall back to corresponding MATLAB matrix reconstruction
        mat_file_name = file_match.replace(".xml", ".mat")
        mat_path = MAT_MODEL_DIR / mat_file_name
        if mat_path.exists():
            valid_models_paths[taxon] = str(mat_path)
    except Exception:
        pass

# Ingest VMH Western dietary medium constraints
medium_df = pd.read_csv(MEDIUM_FILE, sep="\t")
diet_dict = pd.Series(medium_df["Flux Value"].values, index=medium_df["Reaction"]).to_dict()

# ==============================================================================
# 5. In Silico BCAA Perturbation Simulation Loop
# ==============================================================================
DIET_SCALES = {
    "Baseline_1x": 1.0,
    "Moderate_2x": 2.0,
    "High_Overdrive_10x": 10.0
}
TARGET_REACTION_ID = "3MOPLPAMO"
ABUNDANCE_THRESHOLD = 0.001
DISTAL_GUT_DIET_SCALE = 0.1
COOPERATIVE_FRACTION = 0.8
OXYGEN_UPPER_BOUND = 0.001

sample_group_map = dict(zip(df_stratified["Sample_ID"], df_stratified["Stratification_Group"]))

for idx, sample in enumerate(df_stratified["Sample_ID"].tolist(), start=1):
    group = sample_group_map[sample]

    if sample in cached_results:
        print(f"[{idx}/{len(df_stratified)}] Checkpoint hit: Sample '{sample}' already processed. Skipping.")
        continue

    print(f"\n[{idx}/{len(df_stratified)}] Simulating sample: {sample} ({group})...")

    sample_abundance = filtered_abundance_df[[taxa_col, sample]].set_index(taxa_col).to_dict()[sample]
    sample_abundance = {
        tax: ab for tax, ab in sample_abundance.items()
        if tax in valid_models_paths and ab >= ABUNDANCE_THRESHOLD
    }

    if len(sample_abundance) == 0:
        print(f"       [WARNING] No active models above abundance threshold ({ABUNDANCE_THRESHOLD*100}%) for sample '{sample}'. Skipping.")
        continue

    taxonomy_df = pd.DataFrame({
        "id": list(sample_abundance.keys()),
        "abundance": list(sample_abundance.values()),
        "file": [valid_models_paths[tax] for tax in sample_abundance.keys()]
    })

    try:
        community = Community(taxonomy_df, id=sample)
        base_medium_series = community.medium.copy()

        for diet_name, bcaa_mult in DIET_SCALES.items():
            current_medium = {}

            for rxn_id, orig_bound in base_medium_series.items():
                parts = rxn_id.split("__")
                base_id = parts[0] if parts else rxn_id
                species_id = parts[-1] if len(parts) > 1 else None
                abundance_weight = sample_abundance.get(species_id, 1.0) if species_id else 1.0

                base_id_clean = base_id[:-2] + "[e]" if base_id.endswith("_m") else base_id
                variant_bracket = base_id_clean.replace("[e]", "(e)")
                variant_pure = base_id_clean.replace("[e]", "")

                target_flux = None
                if base_id_clean in diet_dict:
                    target_flux = diet_dict[base_id_clean]
                elif base_id in diet_dict:
                    target_flux = diet_dict[base_id]
                elif variant_bracket in diet_dict:
                    target_flux = diet_dict[variant_bracket]
                elif variant_pure in diet_dict:
                    target_flux = diet_dict[variant_pure]

                # Identify Branched-Chain Amino Acid (BCAA) exchange reactions
                rxn_lower = rxn_id.lower()
                is_bcaa = (
                    ("ile" in rxn_lower or "leu" in rxn_lower or "val" in rxn_lower)
                    and ("_l" in rxn_lower or "l_m" in rxn_lower)
                )
                mult = bcaa_mult if is_bcaa else 1.0

                if target_flux is not None:
                    val = abs(target_flux) * DISTAL_GUT_DIET_SCALE * abundance_weight * mult
                    current_medium[rxn_id] = max(val, 0.001)
                else:
                    current_medium[rxn_id] = (
                        orig_bound * abundance_weight * mult if is_bcaa else orig_bound * abundance_weight
                    )

                # Enforce strict anaerobic boundary condition
                if "EX_o2" in rxn_id:
                    current_medium[rxn_id] = OXYGEN_UPPER_BOUND

            community.medium = pd.Series(current_medium)

            # Solve community cooperative tradeoff FBA
            try:
                solution = community.cooperative_tradeoff(fraction=COOPERATIVE_FRACTION, fluxes=True)
                growth_rate = solution.objective_value
                flux_df = solution.fluxes
            except Exception:
                solution = community.optimize(fluxes=True)
                growth_rate = solution.objective_value
                flux_df = solution.fluxes

            # Extract target reaction flux (3MOPLPAMO)
            rxn_flux = 0.0
            if flux_df is not None:
                matching_cols = [c for c in flux_df.columns if TARGET_REACTION_ID in c]
                if matching_cols:
                    rxn_flux = flux_df[matching_cols].sum().sum()
                elif TARGET_REACTION_ID in flux_df.index:
                    rxn_flux = flux_df.loc[TARGET_REACTION_ID].sum()

            new_record = {
                "Sample_ID": sample,
                "Stratification_Group": group,
                "Diet_Condition": diet_name,
                "BCAA_Multiplier": bcaa_mult,
                "Growth_Rate": growth_rate,
                "3MOPLPAMO_Flux": rxn_flux
            }
            final_simulation_results.append(new_record)
            print(f"       [{diet_name:<18}] Growth: {growth_rate:.4f} h^-1 | 3MOPLPAMO Flux: {rxn_flux:.6f} mmol/gDW/h")

        # Save state after each completed sample to ensure fault tolerance
        pd.DataFrame(final_simulation_results).to_csv(OUTPUT_FULL_CSV, index=False)

    except Exception as e:
        print(f"[ERROR] Simulation failed for sample '{sample}': {e}")

print("\n" + "=" * 85)
print("[INFO] BCAA perturbation simulation pipeline completed successfully.")
print(f"[INFO] Cohort flux results saved to: {OUTPUT_FULL_CSV}")
print("=" * 85)