import numpy as np
import pandas as pd
from CCA.utils import *
from CCA.permcca import permcca
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# This code computes CCA using this study's fc and a combination of Nik's and ours subject measures
# See 'nik_mia_sm_4617.ipynb', 'nik_CCA_4617.ipynb', and 'nik_CCA_3766.ipynb' in this directory for 
# for reference, as well as 'nik_behavioral.ipynb' in the behavioral directory.

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
    print('Mia FC and Mia SM full (7196) sample')
    FC = np.load('/data/NIMH_scratch/zwallymi/gradients/group_files/all_fc_matrix.npy')
    FPCs, Fvari_explain, Fs = pca_wrap(FC, method='vanilla', reg=1, ridge_reg=0.1)
    Ypc = np.load('/data/NIMH_scratch/zwallymi/behavioral/new_regressed_behavioral_pca.npy')
    X = FPCs[:, :25]
    Y = Ypc
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
    print('\n Mia FC and Mia SM 4617 sample')
    M_4617_SM = np.load('/data/NIMH_scratch/zwallymi/behavioral/nik_testing/my_4617_PCA.npy')
    X = PCs[:, :25]
    Y = M_4617_SM
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
    print('\n Mia FC and Nik SM 4617 subject sample')
    FC4617 = np.load('/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/4617_fc_matrix.npy')
    PCs, vari_explain4617, s = pca_wrap(FC4617, method='vanilla', reg=1, ridge_reg=0.1)
    Ypc = np.load('/data/NIMH_scratch/zwallymi/behavioral/nik_testing/4617_PCA.npy')
    X = PCs[:, :70]
    Y = Ypc[:, :70]
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
    print('\n Mia FC and Nik SM through our pipeline (3766) sample')
    FC3766 = np.load('/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/3766_fc_matrix.npy')
    PCs3766, vari_explain3766, s = pca_wrap(FC3766, method='vanilla', reg=1, ridge_reg=0.1)
    Ypc3766 = np.load('/data/NIMH_scratch/zwallymi/behavioral/nik_testing/3766_PCA.npy')
    X = PCs3766[:, :70]
    Y = Ypc3766[:, :70]
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
if __name__ == "__main__":
    main()
