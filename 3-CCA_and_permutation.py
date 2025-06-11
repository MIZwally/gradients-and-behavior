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
    Xpc = np.load('/data/NIMH_scratch/zwallymi/gradients/regressed_spearman_correlations.npy')
    Ypc = np.load('/data/NIMH_scratch/zwallymi/behavioral/regressed_behavioral_pca.npy')
    pset10k = np.load('/data/NIMH_scratch/zwallymi/gradients_and_behavior/10kperm/10kperm_pset.npy')
    #compute cca
    A, B, cc = seber_cca(center(Xpc[:,:9]), center(Ypc), 2, 2)
    print("CCA mode correlations: ", cc)     

    #run permutation test, see permcca documentation for output info
    p, r, A, B, U, V, perm_dist = permcca(Xpc[:,:9], Ypc, 10000, Pset=pset10k.T)
    print("Permutation test p-values: ", p)

if __name__ == "__main__":
    main() 