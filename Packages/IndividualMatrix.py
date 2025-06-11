from brainspace.utils.parcellation import reduce_by_labels
import copy
import glob
import nibabel as nib
from nilearn.interfaces.fmriprep import load_confounds
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import subprocess
import os
import pandas as pd

scratch = '/data/NIMH_scratch/zwallymi/gradients/'

#This module contains the functions called in 'individual_files.py' to create functional connectivity matrices for a list of subjects


def check_runs(sub, directory) :
    #determines if the subject's runs should be excluded for too high of a mean framewise displacement
    #returns exclusion boolean, two lists for the good and bad runs, and a list of the runs' numbers
    
    os.chdir(directory)
    #collects the run numbers of all resting state runs for this subject
    files = os.listdir()
    rest = [x for x in files if 'rest' in x]
    runs = [y for y in rest if 'dtseries.nii' in y]
    run_nums = [z[z.find('run-')+4:z.find('run-')+5] for z in runs] #the numbers assigned to each run
    #evaluates mean fd for each run and assigns as good or bad
    good_runs = []
    bad_runs = []
    exclude_any = False #indicates whether or not exclusions occur, to prevent empty lists in exclusion list
    for run in run_nums :
        cf_file =f'{sub}_ses-baselineYear1Arm1_task-rest_run-{run}_desc-confounds_timeseries.tsv'
        cfs = pd.read_csv(f'{directory}/{cf_file}', sep='\t')
        mean_fd = cfs['framewise_displacement'].mean()
        if mean_fd >= 0.5 :
            print('mean fd too high')
            bad_runs.append(run)
            exclude_any = True
            continue
        good_runs.append(run)
    return exclude_any, bad_runs, good_runs, run_nums

def check_volumes(sub, run) :
    #determines if this run has enough volumes that pass motion qualifications
    #returns booleans for if run is long enough and if less tha 50% volumes were removed
    
    pwd = f'/data/ABCD_MBDU/abcd_bids/derivatives/fmriprep/fmriprep_20.2.0/{sub}/out/fmriprep/{sub}/ses-baselineYear1Arm1/func/'
    file = f'{sub}_ses-baselineYear1Arm1_task-rest_run-{run}_space-fsLR_den-91k_bold.dtseries.nii'
    _, sample_mask = load_confounds(pwd+file, strategy=('motion','wm_csf', 'scrub'), motion='full', scrub=5,
                                                    fd_threshold=0.3, std_dvars_threshold=3, wm_csf='basic', 
                                                    compcor='anat_combined', n_compcor=5, demean=True)

    check_len = True
    check_num = True

    og = nib.load(pwd+file)
    og_vol = og.get_fdata().shape[0]
    if sample_mask is None :
        new_vol = og_vol
    else :
        new_vol = sample_mask.shape[0]
    
    perc = new_vol / og_vol
            
    if og_vol < 144 :
        check_len = False
    if perc < 0.5 :
        check_num = False
        
    return check_len, check_num

def regress_confounds(pwd, sub_id, lscratch, run) :
    #regresses out confounds from timeseries
    #uses nilearn.image.clean_image() which only takes nifits, converts cifti to and from nifti
    #confound tsv file must be in the same directory as the image file
    
    dtseries = f'{sub_id}_ses-baselineYear1Arm1_task-rest_run-{run}_space-fsLR_den-91k_bold.dtseries.nii'
    cleaned = f'{sub_id}_ses-baselineYear1Arm1_task-rest_run-{run}_desc-cleaned_bold.nii.gz'
    combo = f'{sub_id}_ses-baselineYear1Arm1_task-rest_run-{run}_desc-cleaned_bold.dtseries.nii'
    
    confounds_out, sample_mask = load_confounds(pwd+dtseries, strategy=('motion','wm_csf', 'scrub'), motion='full', scrub=5,
                                                    fd_threshold=0.3, std_dvars_threshold=3, wm_csf='basic', 
                                                    compcor='anat_combined', n_compcor=5, demean=True)
    
    #bash script to convert cifti to nifti
    to_nifti = f"""
    #!/bin/bash
    module load connectome-workbench
    wb_command -cifti-convert -to-nifti \
        {pwd}{dtseries} \
        {lscratch}{sub_id}_run-{run}_output.nii.gz
    """ 
    subprocess.run(to_nifti, shell=True)
    nifti = nib.load(lscratch+f'{sub_id}_run-{run}_output.nii.gz')
    
    #cleaning nifti
    clean_nifti = image.clean_img(nifti, detrend=True, standardize='zscore_sample', confounds=confounds_out, 
                                    standardize_confounds=True, filter='butterworth', low_pass=0.1, high_pass=0.01, 
                                    t_r=0.8, **{'clean__sample_mask':sample_mask})
    nib.save(clean_nifti, lscratch+cleaned)
    
    #making new template cifti with cleaned dimensions
    total_volumes = clean_nifti.get_fdata().shape[3]
    og = nib.load(pwd+dtseries)
    template_data = og.get_fdata()[sample_mask]
    header = og.header
    new_header = copy.deepcopy(header)
    new_header.get_index_map(0).number_of_series_points = total_volumes
    template = nib.Cifti2Image(template_data, header=new_header)
    nib.save(template, lscratch+f'{sub_id}_run-{run}_template.dtseries.nii')
    
    #bash script to convert nifti back to cifti
    from_nifti = f"""
    #!/bin/bash
    module load connectome-workbench
    wb_command -cifti-convert -from-nifti \
        {lscratch}{cleaned} \
        {lscratch}{sub_id}_run-{run}_template.dtseries.nii \
        {lscratch}{combo}
    """ 
    subprocess.run(from_nifti, shell=True)

def gifti_conversion(sub, lscratch) :
    #creates the surface giftis needed for smoothing
    #hemisphere giftis do not match the template, so they need to be resampled to match for smoothing
    
     #creating directory strings
    template_dir = '/data/MLDSST/templates/templateflow/tpl-fsLR/'
    templateL = 'tpl-fsLR_hemi-L_den-32k_sphere.surf.gii'
    templateR = 'tpl-fsLR_hemi-R_den-32k_sphere.surf.gii'
    subj_fs = f'/data/ABCD_MBDU/abcd_bids/derivatives/fmriprep/fmriprep_20.2.0/{sub}/out/freesurfer/{sub}/surf/'
    subj_anat = f'/data/ABCD_MBDU/abcd_bids/derivatives/fmriprep/fmriprep_20.2.0/{sub}/out/fmriprep/{sub}/ses-baselineYear1Arm1/anat/'
    targetL = glob.glob(subj_anat + '*hemi-L_midthickness.surf.gii')[0]
    targetR = glob.glob(subj_anat + '*hemi-R_midthickness.surf.gii')[0]
    
    #creates bash script to run wb_command
    script = f"""
    !#/bin/bash
    module load freesurfer
    module load connectome-workbench
    
    # convert freesurfer sphere to GIFTI
    mris_convert {subj_fs}lh.sphere {lscratch}lh.sphere.gii
    mris_convert {subj_fs}rh.sphere {lscratch}rh.sphere.gii

    # resample subjects midthickness to fsLR 32k
    wb_command -surface-resample {targetL} {lscratch}lh.sphere.gii {template_dir}{templateL} \
        BARYCENTRIC {lscratch}{sub}_ses-baselineYear1Arm1_hemi-L_midthickness.32k_fs_LR.surf.gii
    wb_command -surface-resample {targetR} {lscratch}rh.sphere.gii {template_dir}{templateR} \
        BARYCENTRIC {lscratch}{sub}_ses-baselineYear1Arm1_hemi-R_midthickness.32k_fs_LR.surf.gii
    """
    subprocess.run(script, shell=True)

def smooth(sub, lscratch, run, vertices) :
    #smoothes the cifti file by converting it to a gifti and running wb_command -cifti-smoothing
    #returns the smoothed image file in cifti format
 
    smooth = f"""
    #!/bin/bash
    module load connectome-workbench
    wb_command -cifti-smoothing "{lscratch}{sub}_ses-baselineYear1Arm1_task-rest_run-{run}_desc-cleaned_bold.dtseries.nii" \
        5 5 COLUMN "{lscratch}{sub}_ses-baselineYear1Arm1_task-rest_run-{run}_desc-smoothed_bold.dtseries.nii" \
        -fwhm -left-surface "{lscratch}{sub}_ses-baselineYear1Arm1_hemi-L_midthickness.32k_fs_LR.surf.gii" \
        -right-surface "{lscratch}{sub}_ses-baselineYear1Arm1_hemi-R_midthickness.32k_fs_LR.surf.gii" 
    """
    
    subprocess.run(smooth, shell=True, check=True)
    return nib.load(lscratch+sub+f'_ses-baselineYear1Arm1_task-rest_run-{run}_desc-smoothed_bold.dtseries.nii')

def surf_data_from_cifti(data, axis, surf_name):
    #gets surface data from cifti file
    #code from Christopher J. Markiewicz's notebook on Nibabel
    
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def decompose_cifti(img, ts):
    #takes in image and gives lh and rh surface data
    #code from Christopher J. Markiewicz's notebook on Nibabel
    
    data = ts
    brain_models = img.header.get_axis(1)  # Assume we know this
    return (surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"),
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT"))
    
def make_matrix(ts) :
    #returns a functional connectivity matrix
    
    correlation_measure = ConnectivityMeasure(kind='correlation')
    matrix = correlation_measure.fit_transform([ts.T])[0]
    return matrix

def clean_smooth_and_project(directory, subject, lscratch, run, vertices=False) :
    #cleans and smooths the data, then projects it to the surface
    #returns combined lh and rh surface data
    #use for each run of a subject, not yet concatenated
    
    #clean and smooth
    regress_confounds(directory, subject, lscratch, run)
    smooth_img = smooth(subject, lscratch, run, vertices)
    
    #get surface data
    fdata=smooth_img.get_fdata(dtype=np.float32)
    if vertices :
        return smooth_img
    surf_lh, surf_rh = decompose_cifti(smooth_img, fdata)
    
    return np.concatenate((surf_lh, surf_rh), axis=0)

def create_individual_fc_matrix(clean_ts, vertices=False) :
    #returns a functional connectivity matrix for a subject's concatenated runs
    #sets up atlas reference and labels data with atlas, then makes matrix
    #intended to run on concatenated subject time series
    
    #loads in scalar file with labels
    if vertices :
        print('vertices')
        scalar = np.load('/data/NIMH_scratch/zwallymi/gradients/vertices_scalar.npy').squeeze()
    else :
        scalar_file = nib.load('/data/MLDSST/parcellations/schaefer/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dscalar.nii')
        scalar = scalar_file.get_fdata(dtype='float64').squeeze()

    #removes 0 vales from scalar and corresponding vertices in time series
    mask = list(scalar != 0)
    extra_clean_ts = clean_ts[mask]

    #make the functional connectivity matrix
    if vertices :
        return make_matrix(extra_clean_ts)
    else :
        #maps labels to vertices in time series for parcellation
        clean_scalar = scalar[mask]
        seed_ts = reduce_by_labels(extra_clean_ts, clean_scalar, axis=1, red_op='mean')
        return make_matrix(seed_ts)

    

