import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
import sys

if len(sys.argv) >= 2:
    name = sys.argv[1]
else:
    raise "no name given"


def main():
    name = "small"
    sparse_matrix = load_sparse_csr(name + "_reshaped_data.npz")

    svd = TruncatedSVD(n_components=2)
    svd.fit(sparse_matrix)
    data = svd.fit_transform(sparse_matrix)

    hac = AgglomerativeClustering(n_clusters=40, affinity='cosine', linkage='average')
    hac.fit(data)
    clusters = hac.fit_predict(data)

    # # remove outliers
    # data = remove_outliers(clusters, 0, data)
    # hac = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='average')
    # hac.fit(data)
    # clusters = hac.fit_predict(data)

    # # remove outliers
    # data = remove_outliers(clusters, 0, data)
    # hac = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='average')
    # hac.fit(data)
    # clusters = hac.fit_predict(data)

    # # remove outliers
    # data = remove_outliers(clusters, 2, data)
    # hac = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='average')
    # hac.fit(data)
    # clusters = hac.fit_predict(data)

    # # remove outliers
    # data = remove_outliers(clusters, 0, data)
    # hac = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
    # hac.fit(data)
    clusters = hac.fit_predict(data)

    print clusters

    plot_dendrogram(hac, labels=hac.labels_)
    plt.show()


def remove_outliers(clusters, cluster_num, data):
    df = pd.concat([pd.DataFrame(data), pd.Series(clusters, name="cluster")], axis=1)
    df = df.loc[df["cluster"] != cluster_num]
    return df.drop(['cluster'], 1).values


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
