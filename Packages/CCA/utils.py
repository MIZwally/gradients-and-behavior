import numpy as np
import copy
from matplotlib import pyplot
import seaborn as sns

#from rPCA import R_pca
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.cross_decomposition import CCA

import nilearn
from nilearn.image import load_img, new_img_like
from nilearn import plotting
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

### For regressing out demographic information
def regress_out_factor( dependent_variable, factor_to_regress_out, discrete=False ):

    if discrete:
        residuals = copy.deepcopy(dependent_variable)
        for i in range(dependent_variable.shape[1]):
            X = dependent_variable[:,i]
            encoder = OneHotEncoder(sparse_output=False)
            y_one_hot = encoder.fit_transform(factor_to_regress_out.reshape(-1, 1))
            y_one_hot = sm.add_constant(y_one_hot)
            model = sm.OLS( X, y_one_hot )
            result = model.fit()
            predicted_X_class = result.predict(y_one_hot)
            residuals_X = X - predicted_X_class
            residuals[:,i] = residuals_X
            
    else:
        residuals = copy.deepcopy(dependent_variable)
        for i in range(dependent_variable.shape[1]):
            X = dependent_variable[:,i]
            y = copy.deepcopy(factor_to_regress_out)
            y = sm.add_constant(y)
            model = sm.OLS( X, y ).fit()
            residuals[:,i] = model.resid
    return residuals


### PCA wrap
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


### For visualuzation

### for gradient visualization
def export_modelcoeff_nii(parcellation_imgpath, model_coef):

    ref_img = load_img(parcellation_imgpath)
    parcellation = ref_img.get_fdata().copy()
    nROI = int(np.max(parcellation))
    assert nROI == model_coef.shape[0]
    for i in range(nROI):
        parcellation[np.where(parcellation==i+1)] = model_coef[i]

    new_img = new_img_like(ref_img, 
                           parcellation,
                           affine=ref_img.affine,
                           copy_header=ref_img.header)
    return new_img, parcellation

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

### for CV fold visualization

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    
    cmap_data = pyplot.cm.Paired
    cmap_cv = pyplot.cm.coolwarm
    
    """Create a sample plot for indices of a cross-validation object."""
    use_groups = "Group" in type(cv).__name__
    groups = group if use_groups else None
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["sex", "sex * age"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    
    return ax

    
### MISC
def shuffle(array, percentage):
    n = len(array)
    index = np.random.choice(np.arange(n), size=int(percentage/100*n), replace=False)
    _array = copy.deepcopy(array)
    _array[index] = array[np.random.permutation(index)]
    return _array
