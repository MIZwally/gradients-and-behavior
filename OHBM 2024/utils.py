import numpy as np
import copy
from matplotlib import pyplot
import seaborn as sns

#from rPCA import R_pca
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA


def shuffle(array, percentage):
    n = len(array)
    index = np.random.choice(np.arange(n), size=int(percentage/100*n), replace=False)
    _array = copy.deepcopy(array)
    _array[index] = array[np.random.permutation(index)]
    return _array

def pca_wrap(X, method='vanilla', reg=1.0, ridge_reg=0.01):

    """
    Input
    X: input data matrix
    method: 'vanilla', 'rPCA', 'sPCA'
    reg: regularization parmaeter, only useful for 'sPCA' & 'rPCA'

    Output
    PCs: principal components
    vari_explain: variance explained by each component
    s: singular values
    """

    assert method in ['vanilla', 'rPCA', 'sPCA']

    ndim = min(X.shape[1], X.shape[0])
    
    if method == 'vanilla':

        pca = PCA(n_components=ndim)
        PCs = pca.fit_transform(X)
        vari_explain = pca.explained_variance_ratio_
        s = pca.singular_values_

        return PCs, vari_explain, s

    elif method == 'sPCA':
        pca = SparsePCA(n_components=ndim, alpha=reg, ridge_alpha=ridge_reg)
        PCs = pca.fit_transform(X)
        Q, R = np.linalg.qr(PCs)
        s = np.diag(R)
        vari_explain = s**2
        vari_explain = vari_explain / np.sum(vari_explain)

        orders = np.argsort(vari_explain)[::-1]
        vari_explain = vari_explain[orders]
        PCs = PCs[:,orders]

    elif method == 'rPCA':
        rpca = R_pca(X, lmbda=reg)
        background_PCs, PCs = rpca.fit(max_iter=10000, iter_print=10000)
        Q, R = np.linalg.qr(PCs)
        s = np.diag(R)
        vari_explain = s**2
        vari_explain = vari_explain / np.sum(vari_explain)
        orders = np.argsort(vari_explain)[::-1]
        vari_explain = vari_explain[orders]
        PCs = PCs[:,orders]
        
    else:
        raise ValueError('method unknown')
    return PCs, vari_explain, s

def plot_distribution(X, ncols=10, title_list=None):
    ndim = X.shape[1]
    if ndim <= ncols:
        ncols = ndim
        nrows = 1
    else:
        nrows = int(np.ceil(ndim/ncols))

    if title_list is not None:
        assert len(title_list) == ndim
        
    fig, axs = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2*nrows))
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
    for k in range(ncols*nrows):
        i, j = np.divmod(k, ncols)
        if k < ndim:
            # sns.histplot(X[:,i], ax=axs[i], log_scale=False)
            axs[i,j].hist(X[:,k])
            axs[i,j].set(ylabel='')
            axs[i,j].set(yticklabels=[])  
            axs[i,j].tick_params(left=False)
            if title_list is not None:
                axs[i,j].set_title(title_list[k])
        else:
            axs[i,j].axis('off')
            axs[i,j].set_title('')
    pyplot.subplots_adjust(wspace=0)
    pyplot.tight_layout()
    pyplot.show()


####################################################################################
# Adpoted from BrainSpace package
####################################################################################

from scipy import sparse as ssp
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from scipy.sparse.csgraph import connected_components

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

def is_symmetric(x, tol=1E-10):
    """Check if input is symmetric.

    Parameters
    ----------
    x : 2D ndarray or sparse matrix
        Input data.
    tol : float, optional
        Maximum allowed tolerance for equivalence. Default is 1e-10.

    Returns
    -------
    is_symm : bool
        True if `x` is symmetric. False, otherwise.

    Raises
    ------
    ValueError
        If `x` is not square.

    """

    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError('Array is not square.')

    if ssp.issparse(x):
        if x.format not in ['csr', 'csc', 'coo']:
            x = x.tocoo(copy=False)
        dif = x - x.T
        return np.all(np.abs(dif.data) < tol)

    return np.allclose(x, x.T, atol=tol)



def make_symmetric(x, check=True, tol=1E-10, copy=True, sparse_format=None):
    """Make array symmetric.

    Parameters
    ----------
    x : 2D ndarray or sparse matrix
        Input data.
    check : bool, optional
        If True, check if already symmetry first. Default is True.
    tol : float, optional
        Maximum allowed tolerance for equivalence. Default is 1e-10.
    copy : bool, optional
        If True, return a copy. Otherwise, work on `x`.
        If already symmetric, returns original array.
    sparse_format : {'coo', 'csr', 'csc', ...}, optional
        Format of output symmetric matrix. Only used if `x` is sparse.
        Default is None, uses original format.

    Returns
    -------
    sym : 2D ndarray or sparse matrix.
        Symmetrized version of `x`. Return `x` it is already
        symmetric.

    Raises
    ------
    ValueError
        If `x` is not square.

    """

    if not check or not is_symmetric(x, tol=tol):
        if copy:
            xs = .5 * (x + x.T)
            if ssp.issparse(x):
                if sparse_format is None:
                    sparse_format = x.format
                conversion = 'to' + sparse_format
                return getattr(xs, conversion)(copy=False)
            return xs
        else:
            x += x.T
            if ssp.issparse(x):
                x.data *= .5
            else:
                x *= .5
    return x

def _build_kernel(x, kernel, gamma=None):

    if kernel in {'pearson', 'spearman'}:
        if kernel == 'spearman':
            x = np.apply_along_axis(rankdata, 1, x)
        return np.corrcoef(x)

    if kernel in {'cosine', 'normalized_angle'}:
        x = 1 - squareform(pdist(x, metric='cosine'))
        if kernel == 'normalized_angle':
            x = 1 - np.arccos(x, x)/np.pi
        return x

    if kernel == 'gaussian':
        if gamma is None:
            gamma = 1 / x.shape[1]
        return rbf_kernel(x, gamma=gamma)

    if callable(kernel):
        return kernel(x)

    raise ValueError("Unknown kernel '{0}'.".format(kernel))

def dominant_set(s, k, is_thresh=False, norm=False, copy=True, as_sparse=True):
    """Keep the largest elements for each row. Zero-out the rest.

    Parameters
    ----------
    s : 2D ndarray
        Similarity/affinity matrix.
    k :  int or float
        If int, keep top `k` elements for each row. If float, keep top `100*k`
        percent of elements. When float, must be in range (0, 1).
    is_thresh : bool, optional
        If True, `k` is used as threshold. Keep elements greater than `k`.
        Default is False.
    norm : bool, optional
        If True, normalize rows. Default is False.
    copy : bool, optional
        If True, make a copy of the input array. Otherwise, work on original
        array. Default is True.
    as_sparse : bool, optional
        If True, return a sparse matrix. Otherwise, return the same type of the
        input array. Default is True.

    Returns
    -------
    output : 2D ndarray or sparse matrix
        Dominant set.

    """

    if not is_thresh:
        nr, nc = s.shape
        if isinstance(k, float):
            if not 0 < k < 1:
                raise ValueError('When \'k\' is float, it must be 0<k<1.')
            k = int(nc * k)

        if k <= 0:
            raise ValueError('Cannot select 0 elements.')

    if as_sparse:
        return _dominant_set_sparse(s, k, is_thresh=is_thresh, norm=norm)

    return _dominant_set_dense(s, k, is_thresh=is_thresh, norm=norm, copy=copy)

def _dominant_set_dense(s, k, is_thresh=False, norm=False, copy=True):
    """Compute dominant set for a dense matrix."""

    if is_thresh:
        s = s.copy() if copy else s
        s[s <= k] = 0

    else:  # keep top k
        nr, nc = s.shape
        idx = np.argpartition(s, nc - k, axis=1)
        row = np.arange(nr)[:, None]
        if copy:
            col = idx[:, -k:]  # idx largest
            data = s[row, col]
            s = np.zeros_like(s)
            s[row, col] = data
        else:
            col = idx[:, :-k]  # idx smallest
            s[row, col] = 0

    if norm:
        s /= np.nansum(s, axis=1, keepdims=True)

    return s


def compute_affinity(x, kernel=None, sparsity=.9, pre_sparsify=True,
                     non_negative=True, gamma=None):
    """Compute affinity matrix.

    Parameters
    ----------
    x : ndarray, shape = (n_samples, n_feat)
        Input matrix.
    kernel : str, None or callable, optional
        Kernel function. If None, only sparsify. Default is None.
        Valid options:

        - If 'pearson', use Pearson's correlation coefficient.
        - If 'spearman', use Spearman's rank correlation coefficient.
        - If 'cosine', use cosine similarity.
        - If 'normalized_angle': use normalized angle between two vectors. This
          option is based on cosine similarity but provides similarities
          bounded between 0 and 1.
        - If 'gaussian', use Gaussian kernel or RBF.

    sparsity : float or None, optional
        Proportion of smallest elements to zero-out for each row.
        If None, do not sparsify. Default is 0.9.
    pre_sparsify : bool, optional
        Sparsify prior to building the affinity. If False, sparsify the final
        affinity matrix.
    non_negative : bool, optional
        If True, zero-out negative values. Otherwise, do nothing.
    gamma : float or None, optional
        Inverse kernel width. Only used if ``kernel == 'gaussian'``.
        If None, ``gamma = 1./n_feat``. Default is None.

    Returns
    -------
    affinity : ndarray, shape = (n_samples, n_samples)
        Affinity matrix.
    """

    # Do not accept sparse matrices for now
    if ssp.issparse(x):
        x = x.toarray()

    if not pre_sparsify and kernel is not None:
        x = _build_kernel(x, kernel, gamma=gamma)

    if sparsity is not None and sparsity > 0:
        x = dominant_set(x, k=1-sparsity, is_thresh=False, as_sparse=False)

    if pre_sparsify and kernel is not None:
        x = _build_kernel(x, kernel, gamma=gamma)

    if non_negative:
        x[x < 0] = 0

    return x


def _graph_is_connected(graph):
    return connected_components(graph)[0] == 1


def diffusion_mapping(adj, n_components=10, alpha=0.5, diffusion_time=0,
                      random_state=None):
    """Compute diffusion map of affinity matrix.

    Parameters
    ----------
    adj : ndarray or sparse matrix, shape = (n, n)
        Affinity matrix.
    n_components : int or None, optional
        Number of eigenvectors. If None, selection of `n_components` is based
        on 95% drop-off in eigenvalues. When `n_components` is None,
        the maximum number of eigenvectors is restricted to
        ``n_components <= sqrt(n)``. Default is 10.
    alpha : float, optional
        Anisotropic diffusion parameter, ``0 <= alpha <= 1``. Default is 0.5.
    diffusion_time : int, optional
        Diffusion time or scale. If ``diffusion_time == 0`` use multi-scale
        diffusion maps. Default is 0.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    v : ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order.
    w : ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in descending order.

    References
    ----------
    * Coifman, R.R.; S. Lafon. (2006). "Diffusion maps". Applied and
      Computational Harmonic Analysis 21: 5-30. doi:10.1016/j.acha.2006.04.006
    * Joseph W.R., Peter E.F., Ann B.L., Chad M.S. Accurate parameter
      estimation for star formation history in galaxies using SDSS spectra.
    """

    rs = check_random_state(random_state)
    use_sparse = ssp.issparse(adj)

    # Make symmetric
    if not is_symmetric(adj, tol=1E-10):
        warnings.warn('Affinity is not symmetric. Making symmetric.')
        adj = make_symmetric(adj, check=False, copy=True, sparse_format='coo')
    else:  # Copy anyways because we will be working on the matrix
        adj = adj.tocoo(copy=True) if use_sparse else adj.copy()

    # Check connected
    if not _graph_is_connected(adj):
        warnings.warn('Graph is not fully connected.')

    ###########################################################
    # Step 2
    ###########################################################
    # When α=0, you get back the diffusion map based on the random walk-style
    # diffusion operator (and Laplacian Eigenmaps). For α=1, the diffusion
    # operator approximates the Laplace-Beltrami operator and for α=0.5,
    # you get Fokker-Planck diffusion. The anisotropic diffusion
    # parameter: \alpha \in \[0, 1\]
    # W(α) = D^{−1/\alpha} W D^{−1/\alpha}
    if alpha > 0:
        if use_sparse:
            d = np.power(adj.sum(axis=1).A1, -alpha)
            adj.data *= d[adj.row]
            adj.data *= d[adj.col]
        else:
            d = adj.sum(axis=1, keepdims=True)
            d = np.power(d, -alpha)
            adj *= d.T
            adj *= d

    ###########################################################
    # Step 3
    ###########################################################
    # Diffusion operator
    # P(α) = D(α)^{−1}W(α)
    if use_sparse:
        d_alpha = np.power(adj.sum(axis=1).A1, -1)
        adj.data *= d_alpha[adj.row]
    else:
        adj *= np.power(adj.sum(axis=1, keepdims=True), -1)

    ###########################################################
    # Step 4
    ###########################################################
    if n_components is None:
        n_components = max(2, int(np.sqrt(adj.shape[0])))
        auto_n_comp = True
    else:
        auto_n_comp = False

    # For repeatability of results
    v0 = rs.uniform(-1, 1, adj.shape[0])

    # Find largest eigenvalues and eigenvectors
    w, v = eigsh(adj, k=n_components + 1, which='LM', tol=0, v0=v0)

    # Sort descending
    w, v = w[::-1], v[:, ::-1]

    ###########################################################
    # Step 5
    ###########################################################
    # Force first eigenvector to be all ones.
    v /= v[:, [0]]

    # Largest eigenvalue should be equal to one too
    w /= w[0]

    # Discard first (largest) eigenvalue and eigenvector
    w, v = w[1:], v[:, 1:]

    if diffusion_time <= 0:
        # use multi-scale diffusion map, ref [4]
        # considers all scales: t=1,2,3,...
        w /= (1 - w)
    else:
        # Raise eigenvalues to the power of diffusion time
        w **= diffusion_time

    if auto_n_comp:
        # Choose n_comp to coincide with a 95 % drop-off
        # in the eigenvalue multipliers, ref [4]
        lambda_ratio = w / w[0]

        # If all eigenvalues larger than 0.05, select all
        # (i.e., sqrt(adj.shape[0]))
        threshold = max(0.05, lambda_ratio[-1])
        n_components = np.argmin(lambda_ratio > threshold)

        w = w[:n_components]
        v = v[:, :n_components]

    # Rescale eigenvectors with eigenvalues
    v *= w[None, :]

    # Consistent sign (s.t. largest value of element eigenvector is pos)
    v *= np.sign(v[np.abs(v).argmax(axis=0), range(v.shape[1])])
    return v, w
