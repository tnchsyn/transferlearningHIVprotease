# Data (in `data/raw/`):
ADJ.xlsx, F_Xs.mat , F_Ys.mat, Xs.mat, Ys.mat: Find detailed explanations within the training code (src/training/WGCN_Train.py).
Stanford_Data.xlsx: Stanford data for 8 PIs (downloaded 27/12/2022).
External_Data.xlsx: ChEMBL-curated dataset. Refer to the manuscript for full details.
Muts.mat: Contains unique mutations found in the Stanford dataset.
ChemProp_8PI.mat: ChemProp transfer learning representation (8 PIs, in alphabetical order).
ChemProp_External.mat: ChemProp transfer learning representation (external PIs).
F_CP.mat: Provides index start/end points for the 8 PIs within F_Xs.

# Codes:
src/training/WGCN_Train.py: Primary training code for the WGCN-Chemprop-Physco7 model. Generates 5-fold cross-validation predictions (external data). Key output saved to outputs/EXTER_TEST_RESULTS_WGCN_CPROP.mat (used by Post_Analysis.mat).
src/postprocess/Post_Analysis.mat: Analyzes external dataset predictions. Provides performance metrics found in the manuscript's Table 3 (Scenario 3) and Table S3.
src/postprocess/class_perform.mat: Calculates classification metrics (used by Post_Analysis.mat).
src/postprocess/str_char_improved.m: Function extracts unique mutations from isolates (used by Post_Analysis.mat).