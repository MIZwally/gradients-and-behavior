import numpy as np
from neuroCombat import neuroCombat
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#This code was used to regress site confounds from the fc matrix data

def main() : 

    #create 3d matrix of fc matrices, flatten to 2d, and get list of subjects
    dir = '/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/matrices'
    os.chdir(dir)
    files = os.listdir()
    matrices = [np.load(f'{dir}/{f}') for f in files]
    flattened = [mat[np.triu_indices(400, k = 1)] for mat in matrices]
    flat_matrices = np.array(flattened)
    subjects = [x[4:8]+'_'+x[8:19] for x in files]

    #obtain site and family values for each subject
    admin = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_y_lt.csv')
    age = admin['interview_age']
    admin = admin[admin.columns[:3]]
    admin = pd.concat([admin, age], axis=1)
    admin_baseline = admin[admin['eventname']=='baseline_year_1_arm_1']
    admin_filtered = admin_baseline[admin_baseline['src_subject_id'].isin(subjects)]
    admin_filtered['src_subject_id'] = pd.Categorical(admin_filtered['src_subject_id'], categories=subjects, ordered=True)
    sorted_admin = admin_filtered.sort_values('src_subject_id')

    #obtaining sex
    demo = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_p_demo.csv')
    demo_baseline = demo[demo['eventname']=='baseline_year_1_arm_1']
    sex = pd.concat([demo_baseline['src_subject_id'], demo_baseline['demo_sex_v2']], axis=1)
    sex_filtered = sex[sex['src_subject_id'].isin(sorted_admin['src_subject_id'])]
    sex_filtered['src_subject_id'] = pd.Categorical(sex_filtered['src_subject_id'], categories=subjects, ordered=True)
    sorted_sex = sex_filtered.sort_values('src_subject_id')

    site_ids = list(sorted_admin['site_id_l'])
    age_ids = list(sorted_admin['interview_age'])
    sex_ids = list(sorted_sex['demo_sex_v2'])

    #remove matrices that do not have site variable
    indices = []
    for sub in subjects :
        if sub in sorted_admin['src_subject_id'].values :
            indices.append(subjects.index(sub))
    flat_matrices = flat_matrices[indices, :]

    #regress out site
    covars = pd.DataFrame({'site': site_ids, 'age': age_ids, 'sex': sex_ids})
    combat = neuroCombat(dat=flat_matrices.T, covars=covars, batch_col='site', 
                            categorical_cols=['sex'], continuous_cols=['age'])

    #save new matrices
    diag = np.diag(np.diag(matrices[0]))
    for i, data in enumerate(combat['data'].T) :
        new = np.zeros((400, 400))
        new[np.triu_indices(new.shape[0], k = 1)] = data
        new = new + new.T + diag
        newdir = '/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/site_regressed_matrices'
        sub = f'sub-{subjects[i][:4]}{subjects[i][5:]}'
        np.save(f'{newdir}/{sub}_regressed_matrix.npy', new)
        
    #save confound information
    with open('/data/NIMH_scratch/zwallymi/gradients/individual_files/parcellated/combat_estimates.pkl', 'wb') as file:
        pickle.dump(combat['estimates'], file)

if __name__ == "__main__":
    main()