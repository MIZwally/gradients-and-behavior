import numpy as np
import pandas as pd
import os
from neuroCombat import neuroCombat

# This code regresses site data from the fc matrices created with Goyal et al.'s 
# time series data in Schaefer 200 and 400 parcellations. 

# For the 5013 subject sample, some of the participants had their site, sex and age information 
# in a separate location and had to be added in manually, hence the separate demographic 
# processing section. 

def main() :
    #reads in all matrices
    
    parcels = '400'
    #parcels = '200
    
    num_subs = '5013'
    #num_subs = '4617'
    #num_subs = '3766'

    mat_dir = f"/data/NIMH_scratch/zwallymi/nik_files/matrices_{parcels}"
    mat_suffix = f"_nik_{parcels}_matrix.npy"
    
    if num_subs == '5013' :
        with open('/data/ABCD_MBDU/goyaln2/abcd_cca_replication/data/5013/5013_subjects.txt', 'r') as file:
                subjects_list = [line.strip() for line in file]
    else :
        with open(f'/data/NIMH_scratch/zwallymi/gradients_and_behavior/nik_{num_subs}_subs.txt', 'r') as file:
                subjects_list = [line.strip() for line in file]
            
    matrices = []
    for sub in subjects_list :
        matrices.append(np.load(f"{mat_dir}/{sub}{mat_suffix}", allow_pickle=True))
        
    flattened = [mat[np.triu_indices(parcels, k = 1)] for mat in matrices]
    flat_matrices = np.array(flattened)
    
    subjects = [x[4:8]+'_'+x[8:19] for x in subjects_list]

    #obtain site and age values for each subject
    admin = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_y_lt.csv')
    age = admin['interview_age']
    admin = admin[admin.columns[:3]]
    admin = pd.concat([admin, age], axis=1)
    admin_baseline = admin[admin['eventname']=='baseline_year_1_arm_1']
    admin_filtered = admin_baseline[admin_baseline['src_subject_id'].isin(subjects)]
    #obtaining sex
    demo = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_p_demo.csv')
    demo_baseline = demo[demo['eventname']=='baseline_year_1_arm_1']
    sex = pd.concat([demo_baseline['src_subject_id'], demo_baseline['demo_sex_v2']], axis=1)
    sex_filtered = sex[sex['src_subject_id'].isin(subjects)]
    sex_filtered['src_subject_id'] = pd.Categorical(sex_filtered['src_subject_id'], categories=subjects, ordered=True)
    
    if num_subs == '5013' :
        #gathering info for subjects with info in release 4 (only if using 5013)
        admin_list = list(admin_baseline['src_subject_id'])
        r4_subs = [x for x in subjects if x not in admin_list]
        r4_subs = r4_subs[:5]

        release4 = pd.read_table('/data/ABCD_DSST/ABCD/tabulated_data/release4/abcd_lt01.txt', dtype=str)
        release4 = release4[release4.columns[4:]]
        release4_baseline = release4[release4['eventname']=='baseline_year_1_arm_1']
        release4_small = release4_baseline[release4_baseline['src_subject_id'].isin(r4_subs)]
        release4_filtered = pd.concat((release4_small[release4_small.columns[0:1]], release4_small[release4_small.columns[4:6]]), axis=1)
        release4_filtered = pd.concat((release4_filtered, release4_small[release4_small.columns[2:3]]), axis=1)
        release4_filtered['interview_age'] = [119.0, 127.0, 126.0, 117.0, 127.0]
        release4_sex = pd.DataFrame({'demo_sex_v2': [2.0, 2.0, 1.0, 1.0, 2.0]}, index=release4_filtered.index)
        release4_full = pd.concat((release4_filtered, release4_sex), axis=1)
        #insert data from missing subject
        new_row = pd.DataFrame([{'src_subject_id': 'REDACTED', 'eventname': 'baseline_year_1_arm_1', 'site_id_l': 'site17', 
                'interview_age': 116.0, 'demo_sex_v2': 2.0}])
        release4_full = pd.concat((release4_full, new_row), ignore_index=True)
        
        #adding in subjects with info in release 4 and running site regression (only for 5013)
        ad_full = pd.merge(left=admin_filtered, right=sex_filtered, on='src_subject_id')
        full = pd.concat((ad_full, release4_full))
        full['src_subject_id'] = pd.Categorical(full['src_subject_id'], categories=subjects, ordered=True)
        sorted_full = full.sort_values('src_subject_id')

        site_ids = list(sorted_full['site_id_l'])
        age_ids = list(sorted_full['interview_age'])
        sex_ids = list(sorted_full['demo_sex_v2'])

    else :
        #sorting dfs and running site regression (not 5013)
        admin_filtered['src_subject_id'] = pd.Categorical(admin_filtered['src_subject_id'], categories=subjects, ordered=True)
        sorted_admin = admin_filtered.sort_values('src_subject_id')
        sex_filtered['src_subject_id'] = pd.Categorical(sex_filtered['src_subject_id'], categories=subjects, ordered=True)
        sorted_sex = sex_filtered.sort_values('src_subject_id')

        site_ids = list(sorted_admin['site_id_l'])
        age_ids = list(sorted_admin['interview_age'])
        sex_ids = list(sorted_sex['demo_sex_v2'])
        
    covars = pd.DataFrame({'site': site_ids, 'age': age_ids, 'sex': sex_ids})
    combat = neuroCombat(dat=flat_matrices.T, covars=covars, batch_col='site', 
                                categorical_cols=['sex'], continuous_cols=['age'])
    
    np.save(f'/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/schaefer_{parcels}_{num_subs}_fc_matrix.npy', combat['data'].T)

if __name__ == "__main__":
    main() 