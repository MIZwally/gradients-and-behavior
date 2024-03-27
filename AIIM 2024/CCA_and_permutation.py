import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from utils import *
from permcca import permcca
import matplotlib.pyplot as plt
import seaborn as sns

def main() :
    Xpc = np.load('/data/NIMH_scratch/zwallymi/gradients/new_regressed_spearman_correlations.npy')
    Ypc = np.load('/data/NIMH_scratch/zwallymi/behavioral/new_regressed_behavioral_pca.npy')

    #compute cca
    k = 6 #number of components for Xpc or Ypc, smaller of the two
    cca = CCA(n_components=k)
    cca = cca.fit(Xpc[:,:], Ypc[:,:k])
    Xc, Yc = cca.transform(Xpc[:,:], Ypc[:,:k])

    #visualize CCA components
    corrcoef = np.corrcoef(np.hstack((Xc, Yc)).T)
    corrcoef = corrcoef[0:k,:][:,k:].T
    sns.heatmap(corrcoef, annot=True, annot_kws={"fontsize":6}, cmap='Blues',
                xticklabels=[f"Xc-{i+1}" for i in range(k)],
                yticklabels=[f"Yc-{i+1}" for i in range(k)])

    #save correlation values of modes
    diag = np.array([corrcoef[i][i] for i in range(k)])
    np.save('new_regressed_CCA_correlations.npy', diag)

    #run permutation test, see permcca documentation for output info
    p, r, A, B, U, V, perm_dist = permcca(Xpc, Ypc, 1000)

    #create confidence intervals for each mode
    upper_bounds = []
    lower_bounds = []
    for i in range(6) :
        dist = perm_dist[:, i]
        ordered = sorted(dist)
        lower = np.percentile(ordered, 2.5)
        upper = np.percentile(ordered, 97.5)
        upper_bounds.append(upper)
        lower_bounds.append(lower)
        print(lower, upper)

    #visualize modes and confidence intervals of null distribution
    x = [1, 2, 3, 4, 5, 6]
    y = diag
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-', color='firebrick')

    #add rectangles with upper and lower bounds around each point
    for xi, yi, upper, lower in zip(x, y, upper_bounds, lower_bounds):
        width = 1 
        height = upper - lower
        ax.add_patch(plt.Rectangle((xi - width/2, lower), width, height, facecolor='lightgray'))

    ax.set_ylabel('Correlation Value', fontsize=12)
    ax.set_xlabel('CCA Modes', fontsize=12)
    ax.set_title('Results from CCA Permutation Test', fontsize=14)
    ax.legend(('CCA correlations', '95% confidence interval'), fontsize=10, bbox_to_anchor=(0.975, 0.975))
    plt.show()
    
if __name__ == "__main__":
    main() 