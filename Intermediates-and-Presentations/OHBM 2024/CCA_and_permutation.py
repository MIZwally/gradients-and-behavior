import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from CCA.utils import *
from CCA.permcca import permcca
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def seber_cca(Y, X, R, S):
    N = Y.shape[0]
    Qy, Ry, iY = scipy.linalg.qr((Y), mode='economic', pivoting=True)
    Qx, Rx, iX = scipy.linalg.qr((X), mode='economic', pivoting=True)
    K = min(np.linalg.matrix_rank(Y), np.linalg.matrix_rank(X))
    QyTQx = Qy.T @ Qx
    if K <= 6 or K == np.min(QyTQx.shape):
        L, D, MT = scipy.linalg.svd(QyTQx)
    else:
        L, D, MT = scipy.sparse.linalg.svds(QyTQx, k=K)
    
    cc = np.minimum(np.maximum(D[:K], 0), 1)
    A = np.linalg.pinv(Ry) @ (L[:, :K]) * np.sqrt(N - R)
    B = np.linalg.pinv(Rx) @ (MT[:K, :].T) * np.sqrt(N - S)
    A = A[iY]
    B = B[iX]
    return A, B, cc

def center(X):
    icte = np.sum(np.diff(X, axis=0) ** 2, axis=0) == 0
    X = np.delete(X, np.where(icte), axis=1)
    X = X - np.mean(X, axis=0)
    return X

def main() :
    #load in sides of CCA
    Xpc = np.load('/data/NIMH_scratch/zwallymi/gradients/CCA_regressed_spearman_correlations.npy')
    Ypc = np.load('/data/NIMH_scratch/zwallymi/behavioral/CCA_regressed_behavioral_pca.npy')

    #compute cca
    A, B, cc = seber_cca(center(Ypc), center(Xpc), 2, 2)
    print("CCA mode correlations: ", cc)     

    #run permutation test, see permcca documentation for output info
    p, r, A, B, U, V, perm_dist = permcca(Xpc, Ypc, 1000)
    print("Permutation test p-values: ", p)
    
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
    y = cc

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='-', color='firebrick')
    for xi, yi, upper, lower in zip(x, y, upper_bounds, lower_bounds):
        width = 1 
        height = upper - lower
        ax.add_patch(plt.Rectangle((xi - width/2, lower), width, height, facecolor='lightgray'))
        
    ax.set_ylabel('Correlation Value', fontsize=16)
    ax.set_xlabel('CCA Modes', fontsize=16)
    ax.set_title('Results from CCA Permutation Test', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.legend(('CCA correlations', '95% confidence interval'), fontsize=14, bbox_to_anchor=(0.975, 0.975))

    plt.show()

if __name__ == "__main__":
    main() 