# Overview

This repository contains the code and data that support the paper **“Improving Predictive Efficacy for Drug Resistance in Novel HIV-1 Protease Inhibitors through Transfer Learning Mechanisms”** (Journal of Chemical Information and Modeling, 2024, 64 (20), 7844-7863. DOI: 10.1021/acs.jcim.4c01037) by Huseyin Tunc, Sumeyye Yilmaz, Busra Nur Darendeli Kiraz, Murat Sari, Seyfullah Enes Kotil, Ozge Sensoy, and Serdar Durdagi. The workflow combines Python/PyTorch for model training with MATLAB for pre- and post-processing.

The repository enables reproduction of the WGCN-Chemprop-Physco7 model performance on the external dataset referenced in the manuscript.

## Repository structure

```
data/raw/           # All provided datasets and feature matrices
outputs/            # Generated artifacts (e.g., EXTER_TEST_RESULTS_WGCN_CPROP.mat)
src/preprocess/     # Reserved for preprocessing utilities
src/training/       # Python training pipeline
src/postprocess/    # MATLAB post-processing and analysis scripts
```

## Data (in `data/raw/`)

| File | Description |
| --- | --- |
| `ADJ.xlsx`, `F_Xs.mat`, `F_Ys.mat`, `Xs.mat`, `Ys.mat` | See the training script (`WGCN_Train.py`) for detailed feature and label descriptions. |
| `Stanford_Data.xlsx` | Stanford dataset for eight protease inhibitors (downloaded 27/12/2022). |
| `External_Data.xlsx` | ChEMBL-curated dataset (full details in the manuscript). |
| `Muts.mat` | Unique mutations identified in the Stanford dataset. |
| `ChemProp_8PI.mat` | ChemProp transfer learning representation for the eight protease inhibitors (alphabetical order). |
| `ChemProp_External.mat` | ChemProp transfer learning representation for the external protease inhibitors. |
| `F_CP.mat` | Start and end indices for each of the eight protease inhibitors within `F_Xs`. |

# Code

| Script | Purpose |
| --- | --- |
| `src/training/WGCN_Train.py` | Main training pipeline for the WGCN-Chemprop-Physco7 model. Produces 5-fold cross-validation predictions on the external dataset and writes `outputs/EXTER_TEST_RESULTS_WGCN_CPROP.mat`, which is consumed by `Post_Analysis.mat`. |
| `src/postprocess/Post_Analysis.mat` | Analyzes external dataset predictions and reports performance metrics corresponding to Table 3 (Scenario 3) and Table S3 of the manuscript. |
| `src/postprocess/class_perform.mat` | Computes classification metrics used by `Post_Analysis.mat`. |
| `src/postprocess/str_char_improved.m` | Extracts unique mutations from isolates; used by `Post_Analysis.mat`. |

# How to Reproduce Results

1. Train the model using `python src/training/WGCN_Train.py` (Python/PyTorch). The script loads datasets from `data/raw/` and writes `outputs/EXTER_TEST_RESULTS_WGCN_CPROP.mat`.
2. Run `src/postprocess/Post_Analysis.mat` (MATLAB) to compute the performance metrics reported in the manuscript tables. The script reads inputs from `data/raw/` and the generated output file in `outputs/`.

# Citation

If you use this repository, please cite the manuscript mentioned above.
