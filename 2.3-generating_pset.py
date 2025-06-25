import numpy as np
import pandas as pd
import copy

from tqdm import tqdm
seed = 312123
np.random.seed(seed)

# Creates permutation sets by reordering subjects while keeping families together.

def main() :
    # obtain subjects' family ids from tabulated data
    gen = pd.read_csv('/data/NIMH_scratch/zwallymi/tabulated_release5/core/genetics/gen_y_pihat.csv')
    with open('/data/NIMH_scratch/zwallymi/gradients_and_behavior/regressed_subjects_final.txt', 'r') as file:
        sub_list = [line.strip() for line in file]
    subjects = [x[4:8]+'_'+x[8:19] for x in sub_list]
    
    one = gen[gen.columns[:3]]
    two = gen[gen.columns[6]]
    three = gen[gen.columns[12:16]]
    onetwo = pd.concat((one, two), axis=1)
    full = pd.concat((onetwo, three), axis=1)

    full_filtered = full[full['src_subject_id'].isin(subjects)]
    full_filtered['src_subject_id'] = pd.Categorical(full_filtered['src_subject_id'], categories=subjects, ordered=True)
    full_sorted = full_filtered.sort_values(by='src_subject_id')
    
    family_ids = pd.DataFrame(full_sorted['rel_family_id'])
    family_ids = family_ids.reset_index(drop=True)
    
    family_ids.to_csv('/data/NIMH_scratch/zwallymi/gradients_and_behavior/family_ids.csv')
    
    nP = 10000 #number of permutations
    
    pset = pd.DataFrame(generate_group_permutation(family_ids, 'rel_family_id', nP)).T
    np.save('10kperm_pset.npy', pset)

def generate_group_permutation(df, group_column, npermutation=1):

    pset = []
    df_list = []

    # compute family size
    group_sizes = df.groupby('rel_family_id').size().reset_index(name='size')
    # list of all observed family size
    size_list = np.unique(group_sizes['size'].values)
    size_list = np.sort(size_list)
    
    # merge the group size information
    df = df.merge(group_sizes, on='rel_family_id', how='left')

    # add index column to store permuted index
    df['index'] = np.arange(df.shape[0])

    # add orig_index column to store the original indexing
    df['orig_index'] = np.arange(df.shape[0])

    # sort dataframe based on family ID 
    df = df.sort_values(by='rel_family_id')

    # generate npermutation set
    for n in tqdm(range(npermutation)):

        df_with_size = copy.deepcopy(df)

        # permutation within each sets of subjects from families of the same size
        for i, size in enumerate(size_list):

            # extract dataframe with specific family size
            rows_size = df_with_size[df_with_size['size'] == size]
            # print('Number of family if size {}: {}'.format(size, len(rows_size['rel_family_id'].unique())))
            
            # Shuffle the family IDs (while keeping their rows together)
            # Shuffle subjects within each family was not done because same and identity map could happen between family ID
            # shuffling subjects within a family ID may cause sibling matching
            shuffled_family_ids = rows_size['rel_family_id'].unique()
            shuffled_family_ids = np.random.permutation(shuffled_family_ids)
            
            # Create a mapping from original family IDs to shuffled family IDs
            group_mapping = dict(zip(rows_size['rel_family_id'].unique(), shuffled_family_ids))
            
            # Add a new column with the shuffled family assignments
            df_with_size['family_id-{}'.format(size)] = df_with_size['rel_family_id'].map(group_mapping)
            
            # Sort the DataFrame by the shuffled family column
            # df_with_size = df_with_size.sort_values(by='family_id-{}'.format(size)).reset_index(drop=True)
            df_with_size = df_with_size.sort_values(by='family_id-{}'.format(size))
    
            if i > 0:
                df_with_size['family_id-{}'.format(size)] = df_with_size['family_id-{}'.format(size)].fillna(df_with_size['family_id-{}'.format(size_list[i-1])])
    
        # record permuted index (to 'index' column) after family ID shuffling, family by family
        id_list = np.unique(df_with_size['family_id-{}'.format(size_list[-1])].values)
        for id in id_list:
            # the corresponding subject index shuffling can be obtained from the dataframe's index
            new_index = list(df.loc[df[group_column]==id].index)

            # shuffling index inside a family
            shuffled_new_index = np.random.permutation(new_index)
            df_with_size.loc[df_with_size['family_id-{}'.format(size_list[-1])]==id,'index'] = shuffled_new_index

        # reorder the dataframe based on the original index
        df_with_size = df_with_size.sort_values(by='orig_index')

        for i, size in enumerate(size_list[:-1]):
            df_with_size = df_with_size.drop('family_id-{}'.format(size), axis=1)

        df_with_size = df_with_size.rename(columns={'family_id-{}'.format(size_list[-1]): 'permuted_family_id'})

        # extract the final permuted index to be the permutation index
        permuted_index = list(df_with_size['index'].values)
        
        pset.append(permuted_index)
        df_list.append(df_with_size)
        
    return pset

if __name__ == "__main__":
    main() 
