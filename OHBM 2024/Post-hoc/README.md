The following code was used for post-hoc analyses, which consisted of various comparisons between this project and that of Goyal et al. 2022.

'nik_ts_to_fc.py' - converts the time series information from original study to fc matrices in Schaefer 400 or 200 parcellated space
'nik_mats.py' - regresses out site confounds from related matrices
'nik_group_and_spearman.py' - create gradients and individual-to-group spearman correlations
'nik_CCA.py' - reproduce the original CCA, then conduct on reduced 4617 sample and outlier 396 sample
'nik_mia_fc.py' - all analyses that use this study's fc matrices
'nik_schaefer_CCA.py' - all analyses with Goyal data parcellated in Schaefer space
'nik_gradients_CCA.ipynb' - analysis with this study's gradients flattened as input into CCA
