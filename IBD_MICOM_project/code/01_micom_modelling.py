"""
Personalised Gut Microbial Community Metabolic Modelling Pipeline
==================================================================
This script constructs subject-specific microbial community models using
AGORA2 genome-scale metabolic reconstructions and the MICOM framework.
It integrates a dual-format model routing mechanism (SBML XML and MATLAB MAT),
calibrates dietary nutrient uptake bounds based on individual taxon abundances
and Western-diet constraints (VMH database), enforces strict anaerobic conditions,
and executes cooperative tradeoff flux balance analysis (FBA).

Outputs:
  - Subject-specific community reaction flux distributions (CSV)
  - Cohort-wide microbial community growth rate summary (CSV)
"""

import os
from pathlib import Path
import pandas as pd
from cobra.io import read_sbml_model
from micom import Community

# ==============================================================================
# 1. Directory and Path Configuration (Relative Paths)
# ==============================================================================
# Determine base directory dynamically relative to this script or current working directory
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["notebooks", "scripts", "notebooks two"] else SCRIPT_DIR

# Input data directories and file paths
MODEL_DIR = BASE_DIR / "models" / "AGORA2_SBML"
MAT_MODEL_DIR = BASE_DIR / "models" / "matlab model"
DATA_DIR = BASE_DIR / "data"

MAPPING_FILE = DATA_DIR / "mapping_summary.xlsx"
ABUNDANCE_FILE = DATA_DIR / "ibd_taxa.csv"
MEDIUM_FILE = DATA_DIR / "fluxes.tsv"

# Output directories and summary paths
OUTPUT_DIR = BASE_DIR / "flux_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROWTH_SUMMARY_CSV = OUTPUT_DIR / "A1_growth_rates_summary.csv"

# ==============================================================================
# 2. Data Ingestion and Taxonomic Filtering
# ==============================================================================
print("[INFO] Loading taxonomic mapping and abundance profiles...")
mapping_df = pd.read_excel(MAPPING_FILE)
abundance_df = pd.read_csv(ABUNDANCE_FILE)

# Retain only taxonomically matched species in AGORA2
mapped_taxa = mapping_df[mapping_df["Status"] == "Matched"]["Excel_Name"]
taxa_col = abundance_df.columns[0]

# Filter abundance matrix to include only mapped taxa
filtered_abundance_df = abundance_df[abundance_df[taxa_col].isin(mapped_taxa)].copy()

# ==============================================================================
# 3. Dietary Medium Ingestion (VMH Constraints)
# ==============================================================================
if MEDIUM_FILE.exists():
    medium_df = pd.read_csv(MEDIUM_FILE, sep="\t")
    diet_dict = pd.Series(
        medium_df["Flux Value"].values, index=medium_df["Reaction"]
    ).to_dict()
    print(f"[INFO] Loaded VMH dietary medium with {len(diet_dict)} nutrient exchange constraints.")
else:
    raise FileNotFoundError(f"[ERROR] Medium definition file not found at: {MEDIUM_FILE}")

# ==============================================================================
# 4. Hybrid Model Routing (SBML XML & MATLAB Matrix Compatibility)
# ==============================================================================
print("[INFO] Verifying model integrity and establishing dual-format file routing...")

valid_models_paths = {}
sbml_count = 0
mat_count = 0

for taxon in mapped_taxa:
    try:
        file_match = mapping_df[mapping_df["Excel_Name"] == taxon]["AGORA2_File"].values[0]
        sbml_path = MODEL_DIR / file_match

        # Route A: Validate native SBML (.xml) model file integrity
        if sbml_path.exists():
            try:
                _ = read_sbml_model(str(sbml_path))
                valid_models_paths[taxon] = str(sbml_path)
                sbml_count += 1
                continue
            except Exception:
                pass  # Fall back to MATLAB format if SBML parsing fails

        # Route B: Fall back to corresponding MATLAB (.mat) model file
        mat_file_name = file_match.replace(".xml", ".mat")
        mat_path = MAT_MODEL_DIR / mat_file_name

        if mat_path.exists():
            valid_models_paths[taxon] = str(mat_path)
            mat_count += 1
        else:
            print(f"[WARNING] Missing fallback MAT model for taxon '{taxon}' at: {mat_path}")

    except Exception as e:
        print(f"[ERROR] Failed routing model for taxon '{taxon}': {e}")

print(f"[INFO] Model routing complete. Bound {len(valid_models_paths)} total metabolic models.")
print(f"       - Standard SBML (XML): {sbml_count} models")
print(f"       - Fallback MATLAB (MAT): {mat_count} models")

# ==============================================================================
# 5. Community Model Construction and FBA Simulation Loop
# ==============================================================================
print("\n[INFO] Starting individual-level microbial community simulations...")

# Simulation parameter configuration
DIET_SCALE = 0.1               # Scaling factor for distal gut nutrient availability
ABUNDANCE_THRESHOLD = 0.001    # 0.1% relative abundance cutoff
COOPERATIVE_FRACTION = 0.8     # Tradeoff fraction for community growth vs. taxon growth
OXYGEN_UPPER_BOUND = 0.001     # Strict anaerobic constraint (mmol/gDW/h)

sample_columns = abundance_df.columns[1:]

for idx, sample in enumerate(sample_columns, start=1):
    flux_csv_path = OUTPUT_DIR / f"flux_{sample}.csv"

    # Checkpoint: Skip previously calculated samples
    if flux_csv_path.exists():
        print(f"[{idx}/{len(sample_columns)}] Sample '{sample}' already processed. Skipping.")
        continue

    print(f"\n[{idx}/{len(sample_columns)}] Simulating sample: {sample}")
    sample_abundance = (
        filtered_abundance_df[[taxa_col, sample]]
        .set_index(taxa_col)
        .to_dict()[sample]
    )

    # Filter species by abundance threshold and model availability
    sample_abundance = {
        tax: ab for tax, ab in sample_abundance.items()
        if tax in valid_models_paths and ab >= ABUNDANCE_THRESHOLD
    }

    if len(sample_abundance) == 0:
        print(f"[WARNING] No qualifying active taxa (>= {ABUNDANCE_THRESHOLD*100}%) for sample '{sample}'. Skipping.")
        continue

    try:
        # 5a. Construct MICOM taxonomy DataFrame
        taxonomy_df = pd.DataFrame({
            "id": list(sample_abundance.keys()),
            "abundance": list(sample_abundance.values()),
            "file": [valid_models_paths[tax] for tax in sample_abundance.keys()]
        })

        # 5b. Initialise community metabolic model
        community = Community(taxonomy_df, id=sample)

        # 5c. Apply dietary and environmental boundary constraints
        my_medium_series = community.medium
        my_medium = (
            my_medium_series.to_dict()
            if hasattr(my_medium_series, "to_dict")
            else dict(my_medium_series)
        )

        matched_nutrient_count = 0
        for rxn_id in my_medium.keys():
            parts = rxn_id.split("__")
            base_id = parts[0] if parts else rxn_id
            species_id = parts[-1] if len(parts) > 1 else None

            # Capture taxon-specific relative abundance weighting
            abundance_weight = sample_abundance.get(species_id, 1.0) if species_id else 1.0

            # Normalise reaction identifier formats across database conventions
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

            if target_flux is not None:
                # Scaled bound = Base VMH flux * Distal gut scaling * Taxon relative abundance
                val = abs(target_flux) * DIET_SCALE * abundance_weight
                my_medium[rxn_id] = max(val, 0.001)
                matched_nutrient_count += 1
            else:
                # Scale unmapped basal exchange fluxes by taxon relative abundance
                my_medium[rxn_id] = my_medium[rxn_id] * abundance_weight

        # Enforce anaerobic environment constraint
        for rxn_id in my_medium.keys():
            if "EX_o2" in rxn_id:
                my_medium[rxn_id] = OXYGEN_UPPER_BOUND

        community.medium = my_medium
        print(f"       Applied {matched_nutrient_count} dietary uptake constraints.")

        # 5d. Solve community cooperative tradeoff FBA
        try:
            solution = community.cooperative_tradeoff(fraction=COOPERATIVE_FRACTION, fluxes=True)
            growth_rate = solution.objective_value
            flux_df = solution.fluxes
        except Exception as solver_err:
            print(f"[WARNING] Cooperative tradeoff solver encountered numerical instability: {solver_err}. Retrying with standard optimize()...")
            solution = community.optimize(fluxes=True)
            growth_rate = solution.objective_value
            flux_df = solution.fluxes

        print(f"       Optimization successful. Community growth rate: {growth_rate:.4f} h^-1")

        # 5e. Append community growth rate to summary registry
        summary_df = pd.DataFrame([{"Sample_ID": sample, "Growth_Rate": growth_rate}])
        if not GROWTH_SUMMARY_CSV.exists():
            summary_df.to_csv(GROWTH_SUMMARY_CSV, index=False)
        else:
            summary_df.to_csv(GROWTH_SUMMARY_CSV, mode="a", header=False, index=False)

        # 5f. Export community reaction flux profile
        if flux_df is not None:
            flux_df.to_csv(flux_csv_path)
            print(f"       Flux distribution exported: {flux_csv_path.name}")
        else:
            print(f"[WARNING] Flux DataFrame is empty for sample '{sample}'.")

    except Exception as e:
        print(f"[ERROR] Simulation failed for sample '{sample}': {e}")

print("\n[INFO] Pipeline execution completed.")