import nibabel as nib
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn import datasets
import os
from nilearn.maskers import NiftiLabelsMasker
import argparse

# This code was run as a swarm job to create Schaefer-parcellated fc matrices from Goyal et al. 2022
# time series data. Matrices were made with both a 200 and 400 parcellation, but this file is ready
# for creating 200 parcel matrices.

def ts_to_fc(file, mask) :
    time_series = mask.fit_transform(file)
    correlation_measure = ConnectivityMeasure(kind="correlation")
    return correlation_measure.fit_transform([time_series])[0]

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("job_num", type=int, help="Description for arg1")
    args = parser.parse_args()
    
    with open('/data/ABCD_MBDU/goyaln2/abcd_cca_replication/data/5013/5013_subjects.txt', 'r') as file:
        nik_5013_list = [line.strip() for line in file]
    first = args.job_num * 50
    if args.job_num == 100 :
        second = 5014
    else :
        second = (args.job_num + 1) * 50
    subjects = nik_5013_list[first:second]
    print(subjects)
    
    atlas = datasets.fetch_atlas_schaefer_2018(yeo_networks=17, resolution_mm=2, n_rois=200)
    atlas_filename = atlas['maps']

    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        verbose=5,
    )

    dir = '/data/NIMH_scratch/zwallymi/nik_files/NIFTI'
    os.chdir(dir)
    files_list = os.listdir()
    print('Loading NIFTIs')
    niftis = [nib.load(f'{dir}/{x}') for x in files_list if x[:19] in subjects]

    print(f'{len(niftis)} NIFTIs loaded. Creating matrices')
    matrices = [ts_to_fc(ts, masker) for ts in niftis]
    print('Matrices created. Saving matrices')
    for i, sub in enumerate(subjects) :
        np.save(f'/data/NIMH_scratch/zwallymi/nik_files/matrices_200/{sub}_nik_200_matrix.npy', matrices[i])
        
if __name__ == "__main__":
    main()
