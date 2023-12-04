import pandas as pd
import numpy as np
from CCA.utils import *
import matplotlib.pyplot as plt

def main() :
    
    unfiltered_scores = pd.read_csv("/data/NIMH_scratch/zwallymi/behavioral/no_na_scores.csv", dtype=str)
    no_subject = unfiltered_scores[unfiltered_scores.columns[2:]]
    no_cash = pd.concat([no_subject[no_subject.columns[:10]], no_subject[no_subject.columns[11:]]], axis=1)
    
    cashless = pd.concat([unfiltered_scores[unfiltered_scores.columns[1:12]], unfiltered_scores[unfiltered_scores.columns[13:]]], axis=1)
    cashless.to_csv('/data/NIMH_scratch/zwallymi/gradients_and_behavior/behavior_with_subjects.csv')

    arrays = []
    for row in no_cash.index :
        numeric = pd.to_numeric(no_cash.iloc[row], errors='coerce')
        arrays.append(np.array(numeric))
    numpy = np.array(arrays)
    numpy.shape
    
    PCs, vari_explain, s = pca_wrap(numpy, method='sPCA', reg=1, ridge_reg=0.1)
    
    plt.plot(range(25), vari_explain, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()
    
    vars_to_keep = PCs[:, :8]
    np.save('behavioral_pca.npy', vars_to_keep)

if __name__ == "__main__":
    main() 