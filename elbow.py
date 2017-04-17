import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
import sys

if len(sys.argv) >= 2:
    name = sys.argv[1]
else:
    raise "no name given"


def main():
    eigenvalues = 10
    sparse_matrix = load_sparse_csr(name + "_reshaped_data.npz")

    U, Sigma, VT = randomized_svd(sparse_matrix, n_components=eigenvalues, n_iter=5, random_state=None)
    data = pd.concat([pd.Series(range(eigenvalues)), pd.Series(Sigma)], axis=1)
    sns.pointplot(x=0, y=1, data=data)
    sns.plt.show()


def load_sparse_csr(name):
    loader = np.load(name)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

if __name__ == '__main__':
    main()
