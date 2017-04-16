import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./ratings_Books.csv')
data = data.drop(['timestamp'], 1)


def main():
    print "users", len(data['user'].unique())
    print "books", len(data['book'].unique())

    df = reshape_data()

    X = df.drop(['book'], 1).as_matrix()
    svd = TruncatedSVD(n_components=5)
    svd.fit(X)

    print svd.explained_variance_ratio_
    print svd.explained_variance_ratio_.sum()
    # X_hat = pd.DataFrame(pca.fit_transform(X))
    # sns.jointplot(x=0, y=1, data=X_hat, xlim=(-.68, -.66), ylim=(-.25, -.2))
    # sns.plt.show()


def reshape_data():
    books = pd.Series(data['book'].unique())
    return books.apply(get_ratings)


def get_ratings(book):
    users = pd.Series(data['user'].unique())
    ratings = data.loc[data['book'] == book].reset_index()

    zeros = pd.DataFrame(np.zeros((1, len(users))), columns=users)
    zeros['book'] = [book]

    for i in range(len(ratings['user'])):
        zeros[ratings['user'][i]] = normalize_rating(ratings['rating'][i], book, ratings['user'][i])

    return zeros.loc[0]


def normalize_rating(rating, book, user):
    return rating

if __name__ == '__main__':
    main()
