from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc
import numpy as np
import os
import pandas as pd

#reads in all matrices
mat_dir = "/data/NIMH_scratch/zwallymi/gradients/individual_files/matrices"
grad_dir = "/data/NIMH_scratch/zwallymi/gradients/individual_files/gradients"
mat_suffix = "_connectivity_matrix.npy"
grad_suffix = "_gradients.npy"
os.chdir('/data/NIMH_scratch/zwallymi/gradients/individual_files/')
with open('/data/NIMH_scratch/zwallymi/behavioral/good_subjects.txt', 'r') as file:
        subjects_list = [line.strip() for line in file]

matrices = []
gradients = []
for sub in subjects_list :
    temp = 'sub-' + sub[:4] + sub[5:]
    matrices.append(np.load(f"{mat_dir}/{temp}{mat_suffix}", allow_pickle=True))
    gradients.append(np.load(f"{grad_dir}/{temp}{grad_suffix}", allow_pickle=True))
        
#combine into a 3D array
all_matrices = np.stack(matrices)
#calculate group matrix
group_matrix = np.average(all_matrices, axis=0)
#calculate group gradients and convert to DataFrame
group_gradients = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
group_gradients = group_gradients.fit(group_matrix)
group_df = pd.DataFrame(group_gradients.gradients_)

#load HCP adult gradients and convert to DataFrame
hcp_matrix = load_group_fc('schaefer', scale=400)
hcp_gradient = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
hcp_gradient.fit(hcp_matrix)
hcp_df = pd.DataFrame(hcp_gradient.gradients_)

#filter dfs for first 2 gradients, get Hypothesis 1 correlations, and make readable
hcp_2_df = hcp_df[hcp_df.columns[0:2]]
group_2_df = group_df[group_df.columns[0:2]]
merged_df = pd.concat([group_2_df, hcp_2_df], axis=1)
hcp_correlation = merged_df.corr(method='spearman')
hcp_corr = {"Group primary to HCP primary: ": hcp_correlation.iloc[0, 2], "Group primary to HCP secondary: ": hcp_correlation.iloc[0, 3],
            "Group secondary to HCP primary: ": hcp_correlation.iloc[1, 2], "Group secondary to HCP secondary: ": hcp_correlation.iloc[1, 3]}

#calculate individual to group Spearman correlations, put into DataFrame
correlations = []
for grad, sub in zip(gradients, subjects_list) :
    individual_df = pd.DataFrame(grad)
    correlation = group_df.corrwith(individual_df, axis=0, method='spearman')
    new_series = pd.Series([sub])
    result_series = pd.concat([new_series, correlation], ignore_index=True)
    correlations.append(result_series)
correlations_df = pd.DataFrame(correlations)
