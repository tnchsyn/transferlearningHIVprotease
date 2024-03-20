# This repository supports the findings in " Improving Predictive Efficacy for Drug Resistance in Novel HIV-1 Protease Inhibitors through Transfer Learning Mechanisms ". Reproduce the WGCN-Chemprop-Physco7 model's external dataset performance using the provided codes. Python-PyTorch is used for training, MATLAB for pre/post-processing.

# Data
ADJ.xlsx, F_Xs.mat, F_Ys.mat, Xs.mat, Ys.mat: Find detailed explanations within the training code (WGCN_Train.py).
Stanford_Data.xlsx: Stanford data for 8 PIs (downloaded 27/12/2022).
External_Data.xlsx: ChEMBL-curated dataset. Refer to the manuscript for full details.
Muts.mat: Contains unique mutations found in the Stanford dataset.
ChemProp_8PI.mat: ChemProp transfer learning representation (8 PIs, in alphabetical order).
ChemProp_External.mat: ChemProp transfer learning representation (external PIs).
F_CP.mat: Provides index start/end points for the 8 PIs within F_Xs.

# Codes
WGCN_Train.py: Primary training code for the WGCN-Chemprop-Physco7 model. Generates 5-fold cross-validation predictions (external data). Key output: EXTER_TEST_RESULTS_WGCN_CPROP.mat (used by Post_Analysis.mat).
Post_Analysis.mat: Analyzes external dataset predictions. Provides performance metrics found in the manuscript's Table 3 (Scenario 3) and Table S3.
class_perform.mat: Calculates classification metrics (used by Post_Analysis.mat).
str_char_improved: Function extracts unique mutations from isolates (used by Post_Analysis.mat).
