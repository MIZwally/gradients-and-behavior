from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This code is a subset of the original 'group_and_spearman.py' that computes the individual and group 
# level gradients as well as the individual to group spearman correlations for the Schaefer fc matrices 
# made from Goyal et al. 2022's time series data.

def main() :
    parcels = '400'
    #parcels = '200
    
    #load in matrices and subject list
    dir = f'/data/NIMH_scratch/zwallymi/nik_files/matrices_{parcels}'
    os.chdir(dir)
    files = os.listdir()
    matrices = [np.load(f'{dir}/{f}') for f in files]
    subjects = [x[4:8]+'_'+x[8:19] for x in files]
    subjects_list = [x[:19] for x in files]

    #combine into a 3D array
    all_matrices = np.stack(matrices)
    #calculate group matrix
    group_matrix = np.average(all_matrices, axis=0)

    group_gradients = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
    group_gradients = group_gradients.fit(group_matrix)

    group_df = pd.DataFrame(group_gradients.gradients_)
    
    np.save(f'/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/schaefer_{parcels}_group_gradients.npy', group_gradients.gradients_)
    
    gradients = []
    gm = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0, alignment='procrustes')
    for sub, mat in zip(subjects_list, matrices) :
        gradient = gm.fit(mat, reference=group_gradients.gradients_)
        gradients.append(gradient.aligned_)
        np.save(f'/data/NIMH_scratch/zwallymi/nik_files/gradients_{parcels}/{sub}_gradients.npy', gradient.aligned_)

    correlations = []
    for grad, sub in zip(gradients, subjects_list) :
        individual_df = pd.DataFrame(grad)
        correlation = group_df.corrwith(individual_df, axis=0, method='spearman')
        new_series = pd.Series([sub])
        result_series = pd.concat([new_series, correlation], ignore_index=True)
        correlations.append(result_series)
    correlations_df = pd.DataFrame(correlations)
    
    np.save(f'/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/schaefer_{parcels}_spearman_correlations', 
            np.array(correlations_df[correlations_df.columns[1:]]))

if __name__ == "__main__":
    main() 