import numpy as np
import pandas as pd
import os
from neuroCombat import neuroCombat

#this code regresses site confounds from the matrices for specific subject sets

def main() :
    #reads in all matrices
    num_subs = '4617'
    #num_subs = '3766'
    
    mat_dir = "/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/site_regressed_matrices"
    mat_suffix = "_regressed_matrix.npy"
    os.chdir('/data/NIMH_scratch/zwallymi/gradients/individual_files/')
    with open(f'/data/NIMH_scratch/zwallymi/gradients_and_behavior/nik_{num_subs}_subs.txt', 'r') as file:
            subjects_list = [line.strip() for line in file]

    matrices = []
    for sub in subjects_list :
        matrices.append(np.load(f"{mat_dir}/{sub}{mat_suffix}", allow_pickle=True))
        
    #flattens matrices   
    flattened = [mat[np.triu_indices(400, k = 1)] for mat in matrices]
    flat_matrices = np.array(flattened)

    subjects = [x[4:8]+'_'+x[8:19] for x in subjects_list]

    #obtain site and family values for each subject
    admin = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_y_lt.csv')
    age = admin['interview_age']
    admin = admin[admin.columns[:3]]
    admin = pd.concat([admin, age], axis=1)
    admin_baseline = admin[admin['eventname']=='baseline_year_1_arm_1']
    admin_filtered = admin_baseline[admin_baseline['src_subject_id'].isin(subjects)]
    admin_filtered['src_subject_id'] = pd.Categorical(admin_filtered['src_subject_id'], categories=subjects, ordered=True)
    sorted_admin = admin_filtered.sort_values('src_subject_id')
    #obtain sex for each subject
    demo = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_p_demo.csv')
    demo_baseline = demo[demo['eventname']=='baseline_year_1_arm_1']
    sex = pd.concat([demo_baseline['src_subject_id'], demo_baseline['demo_sex_v2']], axis=1)
    sex_filtered = sex[sex['src_subject_id'].isin(sorted_admin['src_subject_id'])]
    sex_filtered['src_subject_id'] = pd.Categorical(sex_filtered['src_subject_id'], categories=subjects, ordered=True)
    sorted_sex = sex_filtered.sort_values('src_subject_id')
    site_ids = list(sorted_admin['site_id_l'])
    age_ids = list(sorted_admin['interview_age'])
    sex_ids = list(sorted_sex['demo_sex_v2'])

    #regress out confounds
    covars = pd.DataFrame({'site': site_ids, 'age': age_ids, 'sex': sex_ids})
    combat = neuroCombat(dat=flat_matrices.T, covars=covars, batch_col='site', 
                            categorical_cols=['sex'], continuous_cols=['age'])

    #save resulting matrix
    np.save(f'/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/{num_subs}_fc_matrix.npy', combat['data'].T)

if __name__ == "__main__":
    main()
