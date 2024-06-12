import numpy as np
import os
import pandas as pd
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc

def main() :
    #load in matrices and subject list
    dir = '/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/CCA_site_regressed_matrices'
    files = os.listdir('/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/CCA_site_regressed_matrices')
    matrices = [np.load(f'{dir}/{f}') for f in files]
    subjects = [x[4:8]+'_'+x[8:19] for x in files]

    #make group level gradient
    all_matrices = np.stack(matrices)
    group_matrix = np.average(all_matrices, axis=0)
    group_gradients = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
    group_gradients = group_gradients.fit(group_matrix)
    group_df = pd.DataFrame(group_gradients.gradients_)

    #make individal gradients aligned to group gradient
    new_gradients = []
    gm = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0, alignment='procrustes')
    for i, mat in enumerate(matrices) :
        gradient = gm.fit([mat], reference=group_gradients.gradients_)
        new_gradients.append(gradient.aligned_)
        sub = f'sub-{subjects[i][:4]}{subjects[i][5:]}'
        np.save(f'/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/CCA_site_regressed_gradients/{sub}_gradients.npy', gradient.aligned_)

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

    print(hcp_corr)
    #calculate individual to group spearman correlations and save
    group_df = pd.DataFrame(group_gradients.gradients_)
    correlations = []
    for grad, sub in zip(new_gradients, subjects) :
        individual_df = pd.DataFrame(grad[0])
        correlation = group_df.corrwith(individual_df, axis=0, method='spearman')
        new_series = pd.Series([sub])
        result_series = pd.concat([new_series, correlation], ignore_index=True)
        correlations.append(result_series)
    np.save('/data/NIMH_scratch/zwallymi/gradients/CCA_regressed_spearman_correlations.npy', correlations)
    
if __name__ == "__main__":
    main()