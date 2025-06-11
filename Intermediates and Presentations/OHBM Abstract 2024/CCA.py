import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import CCA
from Packages.CCA.utils import *

def main() :
    #load behavioral and gradient data as numpy arrays
    Ypc = np.load('behavioral_pca.npy')
    Xpc = np.load('gradient_corrs.npy')

    #run CCA
    k = 8
    cca = CCA(n_components=k)
    cca = cca.fit(Xpc[:,:], Ypc[:,:k])
    Xc, Yc = cca.transform(Xpc[:,:], Ypc[:,:k])

    #plot correlations
    corrcoef = np.corrcoef(np.hstack((Xc, Yc)).T)
    corrcoef = corrcoef[0:k,:][:,k:].T
    sns.heatmap(corrcoef, annot=True, annot_kws={"fontsize":6}, cmap='Blues',
                xticklabels=[f"Xc-{i+1}" for i in range(k)],
                yticklabels=[f"Yc-{i+1}" for i in range(k)])

    #grab only useful correlations and save as numpy file
    diag = np.array([corrcoef[i][i] for i in range(k)])
    np.save('CCA_correlations.npy', diag)

if __name__ == "__main__":
    main() 