This code was used for the gradient portion of the project. The files should be run in the following order:

'individual_files.py' - creates the functional connectivity matrices using functions in 'IndividualMatrix.py'.
'matrix_site_regression.py' - deconstructs the matrices to be flat, regresses site confounds, and reconstructs them.
'site_regressed_group_and_spearman.py' - creates group level matrix and gradients, then creates individual gradients aligned to the group gradients. Also correlates group gradients to HCP gradients and creates individual to group correlations.
