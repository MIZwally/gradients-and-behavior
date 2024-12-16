import argparse
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc
import numpy as np
import os
import pandas as pd
from sklearn.utils import resample
from utils import *
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

def main() :
    #load in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("job_num", type=str, help="Description for arg1")
    args = parser.parse_args()
    #get subject list in proper format
    with open('/data/NIMH_scratch/zwallymi/gradients_and_behavior/good_subjects.txt', 'r') as file:
            old_list = [line.strip() for line in file] 
    subjects_list = []
    for sub in old_list :
        subjects_list.append('sub-' + sub[:4] + sub[5:])
    #create directory shortcuts
    mat_dir = "/data/NIMH_scratch/zwallymi/gradients/individual_files/matrices"
    mat_suffix = "_connectivity_matrix.npy"
    grad_suffix = "_gradients.npy"
    #create GradientMap objects with and without Procrustes alignment
    gm = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
    gmp = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0, alignment='procrustes')        
    
    #make directory for this sample and resample matrices
    os.chdir('/data/NIMH_scratch/zwallymi/gradients_and_behavior/bootstrap')
    os.mkdir(f'resample-{args.job_num}')
    os.chdir(f'resample-{args.job_num}')
    resamp = resample(subjects_list, replace=True, n_samples=7179)
    sub_resamp = [x for x in resamp]
    for i, stri in enumerate(sub_resamp) :
        sub_resamp[i] = stri[:8] + "_" + stri[8:]
    sub_resamp = [x[4:] for x in sub_resamp]
    matrices = []
    for sub in resamp : 
        matrices.append(np.load(f"{mat_dir}/{sub}{mat_suffix}", allow_pickle=True))
    
    #make group gradients
    group_matrix = np.average(matrices, axis=0)
    group_gradient = gm.fit(group_matrix)
    np.save(f'resample-{args.job_num}_reference_group{grad_suffix}', group_gradient.gradients_)
    
    #make individual gradients Procrustes aligned to group
    gradients = []
    for j, sub in enumerate(resamp) : 
        grad = gmp.fit([matrices[j]], reference=group_gradient.gradients_)
        gradients.append(grad)
    
    #recreate group with aligned individuals
    group_matrix = np.average(matrices, axis=0)
    group_gradient = gm.fit(group_matrix)
    np.save(f'resample-{args.job_num}_group{grad_suffix}', group_gradient.gradients_)
    
    #load HCP adult gradients and convert to DataFrame
    hcp_matrix = load_group_fc('schaefer', scale=400)
    hcp_gradient = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
    hcp_gradient.fit(hcp_matrix)
    hcp_df = pd.DataFrame(hcp_gradient.gradients_)

    #filter dfs for first 2 gradients, get Hypothesis 1 correlations
    group_df = pd.DataFrame(group_gradient.gradients_)
    hcp_2_df = hcp_df[hcp_df.columns[0:2]]
    group_2_df = group_df[group_df.columns[0:2]]
    merged_df = pd.concat([group_2_df, hcp_2_df], axis=1)
    hcp_correlation = merged_df.corr(method='spearman')
    hcp_corr = np.array([[hcp_correlation.iloc[0, 2], hcp_correlation.iloc[0, 3]],
                    [hcp_correlation.iloc[1, 2],hcp_correlation.iloc[1, 3]]])
    np.save(f'./resample-{args.job_num}_abcd_and_hcp_correlations.npy', hcp_corr)
    
    #load in individual-to-group correlations and subset
    spearman = pd.read_csv('/data/NIMH_scratch/zwallymi/gradients_and_behavior/spearman_correlations.csv')
    spe_ind = [x for x in spearman.index if spearman.iloc[x, 1] in sub_resamp]
    short_spearman = spearman.iloc[spe_ind]
    corrs = np.array(short_spearman[short_spearman.columns[2:]])
    np.save(f'./resample-{args.job_num}_spearman_correlations.npy', corrs)
    
    #load in behavioral data, subset, and do PCA
    behavior = pd.read_csv('/data/NIMH_scratch/zwallymi/gradients_and_behavior/behavior_with_subjects.csv')
    beh_ind = [y for y in behavior.index if behavior.iloc[y, 1] in sub_resamp]
    short_behavior = behavior.iloc[beh_ind]
    beh_df = short_behavior[short_behavior.columns[2:]]
    arrays = []
    for row in range(len(beh_ind)) :
        numeric = pd.to_numeric(beh_df.iloc[row], errors='coerce')
        arrays.append(np.array(numeric))
    beh_np = np.array(arrays)
    PCs, vari_explain, s = pca_wrap(beh_np, method='sPCA', reg=1, ridge_reg=0.1)
    
    #keeping only components that make up 95% of variance
    cumsum = vari_explain.cumsum()
    goodsum = [x for x in cumsum if x <= 0.95]
    beh_pca = PCs[:, :len(goodsum)]    
    np.save(f'./resample-{args.job_num}_behavior_pca.npy', beh_pca)
    
    #cca
    if len(goodsum) > 10 :
        k = 10
    else :
        k = len(goodsum)
    cca = CCA(n_components=k)
    cca = cca.fit(corrs[:,:], beh_pca[:,:k])
    Xc, Yc = cca.transform(corrs[:,:], beh_pca[:,:k])
    
    #grabbing cca correlations and saving them
    corrcoef = np.corrcoef(np.hstack((Xc, Yc)).T)
    corrcoef = corrcoef[0:k,:][:,k:].T
    diag = np.array([corrcoef[i][i] for i in range(k)])
    np.save(f'./resample-{args.job_num}_CCA_correlations.npy', diag)

if __name__ == "__main__":
    main() 