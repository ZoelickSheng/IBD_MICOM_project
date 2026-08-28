Mechanistic Modelling of Gut Microbial Metabolism in IBD
This repository contains the main computational workflows used for my MSc project on personalised gut microbial metabolic modelling in inflammatory bowel disease (IBD) using AGORA2 and MICOM.

Contents
MICOM_modelling.py
Construction of subject-specific microbial community models and extraction of community metabolic fluxes.

classification.py
Nested cross-validation comparing MICOM-derived metabolic flux features with taxonomic abundance features.

BCAA_perturbation.py
Virtual BCAA perturbation analysis of the 3MOPLPAMO reaction.

Data
The original microbiome abundance data were obtained from the publicly available IBD cohort reported by Franzosa et al.

AGORA2 metabolic reconstructions and other third-party resources are not redistributed in this repository and should be obtained from their original sources.

Software
The analysis was performed in Python using MICOM, COBRApy, scikit-learn, pandas, NumPy, and SciPy.

Generative AI use
Generative AI was used to assist with code organisation, refactoring, and documentation, including the refinement of code comments. All analysis logic, parameters, outputs, and scientific interpretations were reviewed and verified by the author.
