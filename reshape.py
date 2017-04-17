import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
import sys

if len(sys.argv) >= 2:
    name = sys.argv[1]
else:
    raise "no name given"

data = pd.read_csv('./ratings_Books_' + name + '.csv')
data = data.drop(['timestamp'], 1)
count = 0


def main():
    print "users", len(data['user'].unique())
    print "books", len(data['book'].unique())

    sparse_matrix = get_reshaped_data(name + "_reshaped_data")

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
