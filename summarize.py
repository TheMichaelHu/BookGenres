import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse

data = pd.read_csv('./ratings_Books_small.csv')
data = data.drop(['timestamp'], 1)
count = 0


def main():
    print "users", len(data['user'].unique())
    print "books", len(data['book'].unique())
    name = "small"

    sparse_matrix = get_reshaped_data(name + "_reshaped_data")
    # sparse_matrix = load_sparse_csr(name + "_reshaped_data.npz")

    svd = TruncatedSVD(n_components=5)
    svd.fit(sparse_matrix)

    print svd.explained_variance_ratio_
    print svd.explained_variance_ratio_.sum()

    # X_hat = pd.DataFrame(pca.fit_transform(X))
    # sns.jointplot(x=0, y=1, data=X_hat, xlim=(-.68, -.66), ylim=(-.25, -.2))
    # sns.plt.show()

def get_reshaped_data(name):
    sparse_matrix = sparse.vstack(reshape_data().values)
    save_sparse_csr(name, sparse_matrix)
    return sparse_matrix


def save_sparse_csr(name, matrix):
    np.savez(name, data=matrix.data, indices=matrix.indices,
             indptr=matrix.indptr, shape=matrix.shape)


def load_sparse_csr(name):
    loader = np.load(name)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def reshape_data():
    books = pd.Series(data['book'].unique())
    return books.apply(get_ratings)


def get_ratings(book):
    global count
    if count % 1000 == 0:
        print count
    count += 1
    users = pd.Series(data['user'].unique())
    ratings = data.loc[data['book'] == book].reset_index()

    zeros = pd.DataFrame(np.zeros((len(users), 1)), index=users)

    for i in range(len(ratings['user'])):
        zeros.loc[ratings['user'][i]][0] = normalize_rating(ratings['rating'][i], book, ratings['user'][i])

    return sparse.csr_matrix(zeros.values.T)


def normalize_rating(rating, book, user):
    return rating

if __name__ == '__main__':
    main()
