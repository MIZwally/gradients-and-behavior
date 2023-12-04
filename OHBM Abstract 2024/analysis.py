import matplotlib.pyplot as plt
import numpy as np
import os
from pingouin.regression import _bias_corrected_ci
from scipy.stats import norm
import seaborn as sns

def main() :
    directories = os.listdir('/data/NIMH_scratch/zwallymi/gradients_and_behavior/bootstrap')
    directories.remove('logs')
    
    hyp1 = []
    hyp2 = []
    for dir in directories :
        if os.path.exists(f'/data/NIMH_scratch/zwallymi/gradients_and_behavior/bootstrap/{dir}/{dir}_abcd_and_hcp_correlations.npy') :
            hyp1.append(np.load(f'/data/NIMH_scratch/zwallymi/gradients_and_behavior/bootstrap/{dir}/{dir}_abcd_and_hcp_correlations.npy'))
        if os.path.exists(f'/data/NIMH_scratch/zwallymi/gradients_and_behavior/bootstrap/{dir}/{dir}_CCA_correlations.npy') :
            hyp2.append(np.load(f'/data/NIMH_scratch/zwallymi/gradients_and_behavior/bootstrap/{dir}/{dir}_CCA_correlations.npy'))
            
    prim_prim = []
    prim_sec = []
    sec_prim = []
    sec_sec = []
    for array in hyp1 :
        prim_prim.append(array[0, 0] * -1)
        prim_sec.append(array[0, 1] * -1)
        sec_prim.append(array[1, 0] * -1)
        sec_sec.append(array[1, 1] * -1)
    
    hyp1_og = np.load('/data/NIMH_scratch/zwallymi/gradients/abcd_and_hcp_correlations.npy')
    prim_prim_og = hyp1_og[0, 0] * -1
    prim_sec_og = hyp1_og[0, 1] * -1
    sec_prim_og = hyp1_og[1, 0] * -1
    sec_sec_og = hyp1_og[1, 1] * -1
    
    pp_interval = _bias_corrected_ci(np.array(prim_prim), prim_prim_og, alpha=0.01)
    ps_interval = _bias_corrected_ci(np.array(prim_sec), prim_sec_og, alpha=0.01)
    sp_interval = _bias_corrected_ci(np.array(sec_prim), sec_prim_og, alpha=0.01)
    ss_interval = _bias_corrected_ci(np.array(sec_sec), sec_sec_og, alpha=0.01)
    
    first = [array[0] for array in hyp2]
    ordered = sorted(first)
    CCA_lower = np.percentile(ordered, 2.5)
    CCA_upper = np.percentile(ordered, 97.5)
    
if __name__ == "__main__":
    main() 