import numpy as np
import os
import pandas as pd
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc
import nibabel as nib
from neuromaps import images, nulls, stats

def main() :
    
    ########################################
    ### ABCD AND HCP GRADIENT COMPARISON ###
    ########################################
    
    #load in matrices and subject list
    dir = '/data/NIMH_scratch/zwallymi/gradients/individual_files/regressed/site_regressed_matrices'
    files = os.listdir(dir)
    matrices = [np.load(f'{dir}/{f}') for f in files]
    subjects = [x[4:8]+'_'+x[8:19] for x in files]

    #make ABCD group level gradient
    all_matrices = np.stack(matrices)
    group_matrix = np.average(all_matrices, axis=0)
    group_gradients = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
    group_gradients = group_gradients.fit(group_matrix)

    #load HCP data and make gradients
    hcp_matrix = load_group_fc('schaefer', scale=400)
    hcp_gradient = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0)
    hcp_gradient.fit(hcp_matrix)

    #generate parcellated GIFTIs for spin test
    scalar_file = nib.load('/data/MLDSST/parcellations/schaefer/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dscalar.nii')
    scalar = scalar_file.get_fdata().squeeze()
    label_dict = [f'{x}' for x in range(1, 401)]
    label_dict.insert(0, 'unknown')
    parcel_giftiL = images.construct_shape_gii(scalar[:32492], labels=label_dict, intent='NIFTI_INTENT_LABEL')
    parcel_giftiR = images.construct_shape_gii(scalar[32492:], labels=label_dict, intent='NIFTI_INTENT_LABEL')
    nib.save(parcel_giftiL, '/data/NIMH_scratch/zwallymi/gradients/scalars/parcel_giftiL.gii')
    nib.save(parcel_giftiR, '/data/NIMH_scratch/zwallymi/gradients/scalars/parcel_giftiR.gii')

    #generate null distribution via spin test
    rotated = nulls.alexander_bloch(grad, n_perm=10000, atlas='fsLR', density='32k', 
                                    parcellation=['/data/NIMH_scratch/zwallymi/gradients/scalars/parcel_giftiL.gii', 
                                                '/data/NIMH_scratch/zwallymi/gradients/scalars/parcel_giftiR.gii'])
    
    #compute correlations and p-values
    corr12, pval12 = stats.compare_images(group_gradients.gradients_[:,0], hcp_gradient.gradients_[:,1], nulls=rotated[:,:,0], metric='spearmanr')
    corr21, pval21 = stats.compare_images(group_gradients.gradients_[:,1], hcp_gradient.gradients_[:,0], nulls=rotated[:,:,0], metric='spearmanr')
    corr11, pval11 = stats.compare_images(group_gradients.gradients_[:,0], hcp_gradient.gradients_[:,0], nulls=rotated[:,:,0], metric='spearmanr')
    corr22, pval22 = stats.compare_images(group_gradients.gradients_[:,1], hcp_gradient.gradients_[:,1], nulls=rotated[:,:,0], metric='spearmanr')
    print("CORRELATIONS: ABCD primary to HCP primary: ", corr11, "ABCD primary to HCP secondary: ", corr12,
          "ABCD secondary to HCP primary: ", corr21, "ABCD secondary to HCP secondary: ", corr22)
    print(f"P-VALUES: ABCD primary to HCP primary: {pval11:.3f}", f"ABCD primary to HCP secondary: {pval12:.3f}", 
          f"ABCD secondary to HCP primary: {pval21:.4f}", f"ABCD secondary to HCP secondary: {pval22:.3f}")
    
    #################################################
    ### INDIVIDUAL TO GROUP SPEARMAN CORRELATIONS ###
    #################################################
    
    #make individal gradients aligned to group gradient
    individual_gradients = []
    gm = GradientMaps(n_components=10, kernel='normalized_angle', random_state=0, alignment='procrustes')
    for i, mat in enumerate(matrices) :
        gradient = gm.fit([mat], reference=group_gradients.gradients_)
        individual_gradients.append(gradient.aligned_)
        sub = f'sub-{subjects[i][:4]}{subjects[i][5:]}'
    
    #calculate individual to group spearman correlations
    group_df = pd.DataFrame(group_gradients.gradients_)
    correlations = []
    for grad, sub in zip(individual_gradients, subjects) :
        individual_df = pd.DataFrame(grad[0])
        correlation = group_df.corrwith(individual_df, axis=0, method='spearman')
        new_series = pd.Series([sub])
        result_series = pd.concat([new_series, correlation], ignore_index=True)
        correlations.append(result_series)
    print("Individual to group correlations are complete.")

if __name__ == "__main__":
    main()
