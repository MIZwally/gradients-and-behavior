import numpy as np
from CCA.utils import *
import scipy
import os

# This code replicates the original Goyal et al. CCA using our function, as well as with two
# different reduced samples and the Schaefer 200 parcellated gradients. See 'nik_CCA.ipynb', 
# 'nik_CCA_4617.ipynb' and 'nik_schaefer_CCA.ipynb' for reference

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
    print('Rerun of original Nik CCA with our CCA code')
    Xpc = np.loadtxt('/data/NIMH_scratch/abcd_cca/abcd_cca_replication/data/5013/N5.txt', delimiter=',')
    Ypc = np.loadtxt('/data/NIMH_scratch/abcd_cca/abcd_cca_replication/data/5013/S5.txt', delimiter=',')
    X = Xpc
    Y = Ypc
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
    print('\n Nik FC and Nik SM 4617 sample')
    Xpc4617 = np.load('/data/NIMH_scratch/zwallymi/gradients_and_behavior/nik_4617_N5.npy')
    Ypc4617 = np.load('/data/NIMH_scratch/zwallymi/behavioral/nik_testing/4617_PCA.npy')
    X = Xpc4617[:, :70]
    Y = Ypc4617[:, :70]
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
    print('\n  Nik FC and Nik SM 396 sample')
    Xpc396 = np.load('/data/NIMH_scratch/zwallymi/gradients_and_behavior/nik_396_N5.npy')
    Ypc396 = np.load('/data/NIMH_scratch/zwallymi/gradients_and_behavior/nik_396_N5.npy')
    X = Xpc396
    Y = Ypc396
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
    print('\n Nik Schaefer 200 gradients and Nik SM full (5013) sample')
    grads  = np.load('/data/NIMH_scratch/zwallymi/nik_files/schaefer_200_flat_gradients.npy')
    X = grads
    Y = Ypc
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
if __name__ == "__main__":
    main()
