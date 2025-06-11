import numpy as np
import copy
from matplotlib import pyplot

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

    ndim = X.shape[1]
    
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

