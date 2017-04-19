import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
import sys
import urllib2
import json

if len(sys.argv) >= 2:
    name = sys.argv[1]
else:
    raise "no name given"


def main():
    data = pd.read_csv('./ratings_Books_' + name + '.csv')
    books = pd.Series(data['book'].unique(), name="book")
    sparse_matrix = load_sparse_csr(name + "_reshaped_data.npz")

    svd = TruncatedSVD(n_components=7)
    svd.fit(sparse_matrix)
    data = pd.DataFrame(svd.fit_transform(sparse_matrix))

    # keep dropping lowest of 3 clusters
    num_clusters = 3
    clusters_dropped = 0
    while clusters_dropped < 50:
        hac = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
        hac.fit(data)
        clusters = hac.fit_predict(data)
        cluster_counts = {}
        for cluster in clusters:
            if cluster in cluster_counts:
                cluster_counts[cluster] += 1
            else:
                cluster_counts[cluster] = 1
        count_order = np.argsort(cluster_counts.values())[::-1]

        if cluster_counts[count_order[-1]] <= 100:
            num_clusters = 3
            clusters_dropped += 1
            print "iter ", clusters_dropped, ": dropped ", cluster_counts[count_order[-1]]
            data, books = keep_clusters(clusters, count_order[:2], data, books)
        else:
            num_clusters += 1

    for num_clusters in range(2, 5):
        print "_" * 100
        hac = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
        hac.fit(data)
        clusters = hac.fit_predict(data)

        print_clusters(clusters, num_clusters, data, books)

    labels = hac.labels_
    # labels = get_custom_labels(clusters, books)

    plot_dendrogram(hac, labels=labels)
    plt.show()

    # plot_variance(data, 21)

    plot_dendrogram(hac, labels=labels, truncate_mode='lastp', p=4)
    plt.show()


def get_custom_labels(clusters, books):
    labels = pd.Series([str(x) for x in clusters], index=books.index)
    mappings = {"0439064864": "Harry Potter 2", "0439136350": "Harry Potter 3", "0590997297": "Animorphs 9", "0590997289": "Animorphs 8"}

    for isbn in mappings:
        if len(books[books == isbn]) != 0:
            idx = books[books == isbn].index[0]
            labels[idx] = mappings[isbn] + " " + labels[idx]

    return labels.values


def plot_2d_data(matrix):
    svd = TruncatedSVD(n_components=2)
    svd.fit(matrix)
    data = svd.fit_transform(matrix)
    df = pd.DataFrame(data)
    sns.jointplot(x=0, y=1, data=df)
    sns.plt.show()


def plot_variance(data, limit):
    variances = []
    for i in range(2, limit):
        hac = AgglomerativeClustering(n_clusters=i, affinity='cosine', linkage='average')
        hac.fit(data)
        clusters = hac.fit_predict(data)
        variances.append(get_within_cluster_variation(clusters, i, data))

    data = pd.concat([pd.Series(range(2, limit)), pd.Series(variances)], axis=1)
    sns.pointplot(x=0, y=1, data=data)
    sns.plt.show()


def get_within_cluster_variation(clusters, n_clusters, data):
    total_var = 0
    for cluster_num in range(n_clusters):
        df = pd.concat([data, pd.Series(clusters, name="cluster", index=data.index)], axis=1)
        df = df[df["cluster"] == cluster_num]
        dists = pdist(df.drop(['cluster'], 1).values, metric="cosine")
        variance = dists.sum()/len(dists)

        if not np.isnan(variance):
            total_var += variance

    #     print "cluster", cluster_num, "variation: ", variance
    # print "Total variance:", total_var
    return total_var


def print_clusters(clusters, n_clusters, data, books):
    for cluster_num in range(n_clusters):
        print "Cluster", cluster_num, ":"
        df = pd.concat([books, pd.Series(clusters, name="cluster", index=books.index)], axis=1)
        df = df.loc[df["cluster"] == cluster_num]
        cluster_books = df["book"].values

        sample_isbns = np.random.choice(cluster_books, 20)
        print [(x + ": " + get_title(x)) for x in sample_isbns]


def get_title(isbn):
    url = "http://openlibrary.org/api/books?bibkeys=ISBN:" + isbn + "&jscmd=data"
    res = urllib2.urlopen(url).read()
    try:
        data = json.loads(res[18:-1])['ISBN:' + isbn]
    except KeyError:
        return "n/a"

    return json.loads(res[18:-1])['ISBN:' + isbn]['title']


def keep_clusters(clusters, cluster_nums, data, books):
    # whitelist = ["0439064864", "0439136350", "0590997297", "0590997289"]
    df = pd.concat([data, pd.Series(clusters, name="cluster", index=data.index), books], axis=1)
    # df_whitelist = df[df["book"].isin(whitelist)]
    df = df[df["cluster"].isin(cluster_nums)]
    # df = df_whitelist.combine_first(df)
    return df.drop(['cluster', 'book'], 1), df['book']


def remove_clusters(clusters, cluster_nums, data, books):
    # whitelist = ["0439064864", "0439136350", "0590997297", "0590997289"]
    df = pd.concat([data, pd.Series(clusters, name="cluster", index=data.index), books], axis=1)
    # df_whitelist = df[df["book"].isin(whitelist)]
    df = df[~df["cluster"].isin(cluster_nums)]
    # df = df_whitelist.combine_first(df)
    return df.drop(['cluster', 'book'], 1), df['book']


def plot_dendrogram(hac, **kwargs):
    children = hac.children_
    distance = np.arange(children.shape[0])
    observations = np.arange(2, children.shape[0]+2)
    linkage_matrix = np.column_stack([children, distance, observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def load_sparse_csr(name):
    loader = np.load(name)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

if __name__ == '__main__':
    main()
