import numpy as np
import pandas as pd
from CCA.utils import *
import scipy

# This code computes the CCA for Nik's Schaefer-parcellated fc matrices. 
# Comment in/out whichever combination of parcels and sample size you want
# see nik_schaefer_CCA.ipynb for reference

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
    parcels = '400'
    #parcels = '200'
    
    num_subs = '5013'
    #num_subs = '4617'
    #num_subs = '3766'
    
    print(f'Nik Schaefer {parcels} FC with Nik SM {num_subs} sample')
    
    FC = np.load(f'/data/NIMH_scratch/zwallymi/gradients/group_files/nik_testing/schaefer_{parcels}_{num_subs}_fc_matrix.npy')
    PCs, vari_explain400, s = pca_wrap(FC, method='vanilla', reg=1, ridge_reg=0.1)
    
    if num_subs == '5013' :
        Ypc = np.loadtxt('/data/NIMH_scratch/abcd_cca/abcd_cca_replication/data/5013/S5.txt', delimiter=',')
    else :
        Ypc = np.load(f'/data/NIMH_scratch/zwallymi/behavioral/nik_testing/{parcels}_PCA.npy')
        
    X = PCs[:, :70]
    Y = Ypc[:, :70]
    A, B, cc = seber_cca(center(Y), center(X), 2, 2)
    print('Top 5 mode correlations: ', cc[:5])
    
if __name__ == "__main__":
    main()
