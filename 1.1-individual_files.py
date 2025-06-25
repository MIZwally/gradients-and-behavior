import argparse
from brainspace.gradient import GradientMaps
import Packages.IndividualMatrix as IndividualMatrix
import numpy as np
import pandas as pd
import os 
import shutil

################################################################
### this file is intended to be run as part of a bash script ###
################################################################

# takes in a list of subjects, the job id and the job number
# does qc for mean fd and volumes
# makes a fc matrix and gradient file for each subject
# saves matrices in matrices directory and creates a qc file

def main() :
    
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects_file", type=str, help="Description for arg1")
    parser.add_argument("job_id", type=str, help="Description for arg2")
    parser.add_argument("job_num", type=str, help="Description for arg3")
    args = parser.parse_args()

    # open subject file and convert into list of subjects
    with open(args.subjects_file, 'r') as file:
        subjects_list = [line.strip() for line in file]
    
    # get all runs for subject and checks mean framewise displacement
    # does not add run to overall list if mean fd >= 0.5
    subjects = []
    all_runs = []
    fd_exclusion_subs = []
    fd_exclusion_runs = []
    for sub in subjects_list :
        directory = '/data/ABCD_MBDU/abcd_bids/derivatives/fmriprep/fmriprep_20.2.0/'+sub+'/out/fmriprep/'+sub+'/ses-baselineYear1Arm1/func/'
        # determines if subject has image files
        if not os.path.exists(directory) :
            print("no directory")
            continue
        exclude_any, bad_runs, good_runs, run_nums = IndividualMatrix.check_runs(sub)
        # adds runs and subject to overall lists 
        if exclude_any :
            # adds sub and run to exclusion list if any run is excluded
            fd_exclusion_subs.append(sub)
            fd_exclusion_runs.append(bad_runs)
        if bad_runs != run_nums :
            # adds sub and run to normal list if not all runs are excluded
            subjects.append(sub)
            all_runs.append(good_runs)
        
    print("good: ", len(subjects), len(all_runs))
    print("bad: ", len(fd_exclusion_subs), len(fd_exclusion_runs))

    # evaluate runs to see if volumes pass qc    
    exclusion = []
    for sub, runs in enumerate(all_runs) :
        temp = []
        for i in runs :
            small, few = IndividualMatrix.check_volumes(subjects[sub], i)
            if not small :
                if not few :
                    temp.append(3) #3 = both problems
                else : 
                    temp.append(1) #1 = run is too short
            elif not few :
                temp.append(2) #2 = too many volumes removed in cleaning
            else :
                temp.append(0) #0 = no problems
        exclusion.append(temp)

    print(exclusion)

    # make inclusion list
    include = []
    for files, runs in zip(all_runs, exclusion) :
        temp = [file for file, run in zip(files, runs) if run == 0]
        include.append(temp)

    print(include)

    # remove subjects with no usable runs
    usable_subjects = []
    for h, set in enumerate(exclusion) :
        count = 0
        for item in set :
            if item != 0 :
                count+=1
        if count < len(set) :
            usable_subjects.append(subjects[h])

    # get timeseries for each run to be included
    all_timeseries = [None] * len(usable_subjects)
    for sub, list in zip(subjects, include) : 
        if list == [] :
            continue
        directory = '/data/ABCD_MBDU/abcd_bids/derivatives/fmriprep/fmriprep_20.2.0/'+sub+'/out/fmriprep/'+sub+'/ses-baselineYear1Arm1/func/'
        timeseries = []
        # make folder for this subject
        lscratch = f'/lscratch/{args.job_id}/{sub}/'
        os.mkdir(lscratch)
        # get gifti for smoothing
        IndividualMatrix.gifti_conversion(sub, lscratch)
        for run in list :
            temp = IndividualMatrix.clean_smooth_and_project(directory, sub, lscratch, run)
            timeseries.append(temp)
        all_timeseries[i] = timeseries

    # combine all runs for one subject
    concatenated = [None] * len(all_timeseries)
    for x, list in enumerate(all_timeseries) :
        if list == [] :
            continue
        temp = list[0]
        for i in range(len(list)-1) :
            temp = np.concatenate((temp, list[i+1]), axis=1)
        concatenated[x] = temp
    # remove empty timeseries entries    
    final_concat = [x for x in concatenated if x is not None]

    # make fc matrices for all individuals
    matrices = [None] * len(usable_subjects)
    for i, ts in enumerate(final_concat) :
        matrices[i] = IndividualMatrix.create_individual_fc_matrix(ts)

    # creates and saves a qc spreadsheet
    os.chdir(f'/lscratch/{args.job_id}/')
    all_all_runs = fd_exclusion_runs + all_runs
    all_subjects = fd_exclusion_subs + subjects
    labeled_runs = [[sub +"_run-" + num for num in run] for sub, run in zip(all_subjects, all_all_runs)]
    flat_runs = sum(labeled_runs, [])
    fd_exclusion_code = [[1 for _ in row] for row in fd_exclusion_runs]
    all_exclusion = fd_exclusion_code + exclusion
    flat_exclusion = sum(all_exclusion, [])
    qc_df = pd.DataFrame({'runs': flat_runs, 'exclusion_code': flat_exclusion})
    qc_df.to_csv(f'qc_data-{args.job_num}.csv', index=False)

    # saves fc matrices in their own directory
    os.mkdir(f'/lscratch/{args.job_id}/matrices-{args.job_num}')
    for i, matrix in enumerate(matrices) :
        np.save(f'/lscratch/{args.job_id}/matrices-{args.job_num}/{usable_subjects[i]}_connectivity_matrix.npy', matrix)
        
    # removes all temp files
    for sub in usable_subjects :
        shutil.rmtree(sub)
    
if __name__ == "__main__":
    main()
