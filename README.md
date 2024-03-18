# This repository is a product of the article "Transfer learning improves prediction accuracy in testing novel HIV-1 protease inhibitors against drug resistance". The provided codes can be used to regenerate the external dataset performance of the WGCN-Chemprop-Physco7 model. 

# Data:
1) You can find explanations for ADJ.xlsx, F_Xs , F_Ys, Xs, Ys in the training code WGCN_Train.py.
2) Stanford_Data.xlsx is the Stanford data for 8 PI and downloaded in 27/12/2022.
3) External_Data.xlsx was curated from the ChEMBL database, see the manuscript for the details.
4) Muts.mat contains the unique mutations occured in the Stanford dataset.
5) ChemProp_8PI.mat contains the Chemprop tranfer learning representation of 8 PIs with alphabetic order in rows.
5) ChemProp_External.mat contains the Chemprop tranfer learning representation of external PIs.
6) F_CP.mat gives the index start and end points for 8 PIs in F_Xs.
# Codes:
1) WGCN_Train.py: The main training code for WGCN-Chemprop-Physco7 model. This code produces 5-fold cross validation predictions for the external data. The main output is EXTER_TEST_RESULTS_WGCN_CPROP.mat that is used in Post_Analysis.mat.
2) Post_Analysis.mat: This code analyses the predictions for the external data set and provides performance metrics provided in Table 3 - Scenario 3 of the manuscript (as well as Table S3).
3) class_perform.mat: This function evaluates calssification metrics and used in Post_Analysis.mat.
4) str_char_improved: This function is used in extracting unique mutations from the isolates, and this is used in Post_Analysis.mat.
