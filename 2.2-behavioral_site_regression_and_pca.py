import numpy as np
from neuroCombat import neuroCombat
import pandas as pd
from CCA.utils import *
import pickle
import matplotlib.pyplot as plt

#This file regresses out site confounds and performs PCA

def main() :
    unfiltered_scores = pd.read_csv("/data/NIMH_scratch/zwallymi/behavioral/regressed_scores.csv", dtype=str)
    with open('/data/NIMH_scratch/zwallymi/gradients_and_behavior/regressed_subjects_final.txt', 'r') as file:
        subjects = [line.strip() for line in file]
    subjects_list = [x[4:8]+'_'+x[8:19] for x in subjects]   
    
    #only include subjects with both gradient and behavioral data
    scores_filtered = unfiltered_scores[unfiltered_scores[unfiltered_scores.columns[1]].isin(subjects_list)]
    scores_filtered[scores_filtered.columns[1]] = pd.Categorical(scores_filtered[scores_filtered.columns[1]], categories=subjects_list, ordered=True)
    unfiltered_scores_sorted = scores_filtered.sort_values(scores_filtered.columns[1])

    #remove non-numeric columns
    subjects = list(unfiltered_scores_sorted['src_subject_id'])
    unfiltered_scores_sorted.drop(['Unnamed: 0', 'src_subject_id'], axis=1, inplace=True)
    scores = np.array(unfiltered_scores_sorted, dtype=float)

    #obtain site and family values for each subject
    admin = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_y_lt.csv')
    age = admin['interview_age']
    admin = admin[admin.columns[:3]]
    admin = pd.concat([admin, age], axis=1)
    admin_baseline = admin[admin['eventname']=='baseline_year_1_arm_1']
    admin_filtered = admin_baseline[admin_baseline['src_subject_id'].isin(subjects)]
    admin_filtered['src_subject_id'] = pd.Categorical(admin_filtered['src_subject_id'], categories=subjects, ordered=True)
    sorted_admin = admin_filtered.sort_values('src_subject_id')

    #obtain sex
    demo = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/abcd-general/abcd_p_demo.csv')
    demo_baseline = demo[demo['eventname']=='baseline_year_1_arm_1']
    sex = pd.concat([demo_baseline['src_subject_id'], demo_baseline['demo_sex_v2']], axis=1)
    sex_filtered = sex[sex['src_subject_id'].isin(sorted_admin['src_subject_id'])]
    sex_filtered['src_subject_id'] = pd.Categorical(sex_filtered['src_subject_id'], categories=subjects, ordered=True)
    sorted_sex = sex_filtered.sort_values('src_subject_id')
    site_ids = list(sorted_admin['site_id_l'])
    age_ids = list(sorted_admin['interview_age'])
    sex_ids = list(sorted_sex['demo_sex_v2'])

    #regress out site
    covars = pd.DataFrame({'site': site_ids, 'age': age_ids, 'sex': sex_ids})
    combat = neuroCombat(dat=scores.T, covars=covars, batch_col='site', 
                            categorical_cols=['sex'], continuous_cols=['age'])

    #save new data and regression estimates
    np.save('/data/NIMH_scratch/zwallymi/behavioral/site_regressed_scores.npy', combat['data'].T)
    with open('/data/NIMH_scratch/zwallymi/behavioral/combat_estimates.pkl', 'wb') as file:
        pickle.dump(combat['estimates'], file)

    #compute PCA
    PCs, vari_explain, s = pca_wrap(combat['data'].T, method='sPCA', reg=1, ridge_reg=0.1)

    #visualize PCs
    fig, ax = plt.subplots()
    ax.plot(range(25), vari_explain, 'o-', linewidth=2, color='green')
    plt.title('Results from Principal Component Analysis')
    plt.xlabel('Principal Components')
    plt.ylabel('% Variance Explained')
    plt.show()
    
    #identify components that account for first 95% of variance
    cumulative_variance = np.cumsum(vari_explain)
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # +1 to include the component at the threshold

    #extract relevant PCs and save
    vars_to_keep = PCs[:, :n_components_95] #keep components that account for first 95% of variance
    np.save('/data/NIMH_scratch/zwallymi/behavioral/regressed_behavioral_pca.npy', vars_to_keep)

if __name__ == "__main__" :
    main()