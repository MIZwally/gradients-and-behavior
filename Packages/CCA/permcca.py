import numpy as np
import scipy
from tqdm import tqdm
import copy

# This is implmented based on
# https://github.com/andersonwinkler/PermCCA/blob/master/permcca.m

def permcca(Y, X, nP=1000, Z=None, W=None, Sel=None, partial=True, Pset=None):
    
 # Inputs:
 # - Y        : Left set of variables, size N by P.
 # - X        : Right set of variables, size N by Q.
 # - nP       : An integer representing the number of permutations.
 #              Default is 1000 permutations.
 # - Z        : (Optional) Nuisance variables for both (partial CCA) or left side (part CCA) only.
 # - W        : (Optional) Nuisance variables for the right side only (bipartial CCA).
 # - Sel      : (Optional) Selection vector to use Theil's residuals instead of Huh-Jhun's projection. 
 #              The vector should be made of integer indices.
 #              The R unselected rows of Z (S of W) must be full rank. Use -1 to randomly select N-R (or N-S) rows.
 # - partial  : (Optional) Boolean indicating whether
 #              this is partial (true) or part (false) CCA.
 #              Default is true, i.e., partial CCA.
 # - Pset     : Predefined set of permutations (e.g., that respect
 #              exchangeability blocks). For information on how to
 #              generate these, see:
 #              https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
 #              If a selection matrix is provided (for the Theil method),
 #              Pset will have to have fewer rows than the original N, i.e.,
 #              it will have as many rows as the effective number of
 #              subjects determined by the selection matrix.

 # Outputs:
 # - p   : p-values, FWER corrected via closure.
 # - r   : Canonical correlations.
 # - A   : Canonical coefficients, left side.
 # - B   : Canonical coefficients, right side.
 # - U   : Canonical variables, left side.
 # - V   : Canonical variables, right side.
    
    # Read input arguments
    Ny, P = Y.shape
    Nx, Q = X.shape
    N = Ny
    I = np.eye(N)

    # Handle optional arguments
    if Z is None:
        Qz = I
    else:
        Z = center(Z)
        Qz = semiortho(Z, Sel)

    if W is None:
        if partial:
            W = Z
            Qw = Qz
        else:
            Qw = I
    else:
        W = center(W)
        Qw = semiortho(W, Sel)

    # centering Y
    # remove constant column
    Y = center(Y)
    Y = Qz.T @ Y
    Ny = Y.shape[0]
    R = Z.shape[1] if Z is not None else 0

    # centering X
    # remove constant column
    X = center(X)
    X = Qw.T @ X
    Nx = X.shape[0]
    S = W.shape[1] if W is not None else 0

    # Initial CCA
    A, B, r = seber_cca(Qz @ Y, Qw @ X, R, S)
    K = len(r)
    U = Y @ np.hstack((A, scipy.linalg.null_space(A.T) ))
    V = X @ np.hstack((B, scipy.linalg.null_space(B.T) ))

    # Initialize counter
    cnt = np.zeros(K)
    lW = np.zeros(K)

    rperm_list = []

    # For each permutation
    for p in tqdm(range(1, nP + 1)):

        # If user didn't supply a set of permutations,
        # permute randomly both Y and X.
        # Otherwise, use the permutation set to shuffle
        # one side only.
        if Pset is None:
            idxY = np.random.permutation(Ny) if p > 1 else np.arange(Ny)
            idxX = np.random.permutation(Nx) if p > 1 else np.arange(Nx)
        else:
            idxY = Pset[:, p - 1]
            idxX = np.arange(Nx)

        # For each canonical variable
        for k in range(K):
            _, _, rperm = seber_cca(Qz @ U[idxY, k:], Qw @ V[idxX, k:], R, S)
            # why implement like this?
            lWtmp = -np.cumsum( np.log(1 - rperm ** 2)[::-1] )[::-1]
            lW[k] = lWtmp[0]
            # lW[k] = -np.sum(np.log(1 - rperm**2))

        if p == 1:
            lW1 = copy.deepcopy(lW)

        cnt += (lW - lW1 >= 0)*1.0

        _, _, rperm = seber_cca(Qz @ U[idxY], Qw @ V[idxX], R, S)
        rperm_list.append(rperm)

    punc = cnt / nP
    pfwer = np.maximum.accumulate(punc)
    return pfwer, r, A, B, Qz @ Y @ A, Qw @ X @ B, rperm_list


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


def semiortho(Z, Sel=None):

    N, R = Z.shape
    # Sel should be a vector containing selected row index
    if Sel is None:
        Q, D, _ = scipy.linalg.svd(scipy.linalg.null_space(Z.T), full_matrices=False)
        return Q @ np.diag(D)
    else:
        
        # Sel is either -1 or a vector of indices
        if isinstance(Sel, np.ndarray):
            unSel = np.setdiff1d(np.arange(N), Sel)
            if np.linalg.matrix_rank(Z[unSel, :]) < R:
                raise ValueError("Selected rows of nuisance not full rank")
        else:
            assert isinstance(Sel, int)
            assert Sel == -1
            rZ = np.linalg.matrix_rank(Z)
            if rZ < R:
                raise ValueError('Impossible to use the Theil method with this set of nuisance variables')

            # Find unique rows; since unique sorts outputs, shuffle to avoid trends
            _, iU, _ = np.unique(Z, axis=0, return_index=True)
            pidx = np.random.permutation(len(iU))
            iU = iU[pidx]

            # Go by trial and error
            unSel0 = []
            rnk0 = 0
            for u in iU:
                unSel = unSel0.append(u)
                Zout = Z[unSel]
                rnk = np.linalg.matrix_rank(Zout)
                if rnk > rnk0:
                    unSel0 = unSel
                    rnk0 = rnk
                if rnk == R:
                    break       

            Sel = np.setdiff1d(np.arange(N), unSel)
                
        S = np.eye(N)[:, Sel]
        Rz = np.eye(N) - Z @ np.linalg.pinv(Z)
        return Rz @ S @ scipy.linalg.sqrtm(np.linalg.inv(S.T @ Rz @ S))
