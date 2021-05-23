import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense as lloyd_iter
from threadpoolctl import threadpool_limits
from sklearn.cluster._k_means_fast import _inertia_dense as _inertia
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import os


def shape_data(pd_data):
    data = pd_data[
        ['Serving Size', 'Calories', 'Total Fat',
            'Carbohydrates', 'Sugars', 'Protein']
    ]

    data = data.rename(
        columns={'Serving Size': 'Size', 'Total Fat': 'Fat'})

    data['Size'] = data['Size'].str.split().str[0]
    data = data.astype(float)

    data_normalized = (data - data.mean()) / (data.max() - data.min())
    return data_normalized


def kmeans_single_lloyd(X, sample_weight, centers_init, max_iter=100,
                        x_squared_norms=None, n_threads=1):

    n_clusters = centers_init.shape[0]
    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    ret = np.zeros(max_iter)

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers_new,
                       weight_in_clusters, labels, center_shift, n_threads)
            # print('new labels:', labels)
            centers, centers_new = centers_new, centers
            ret[i] = davies_bouldin_score(X, labels)
            labels_old[:] = labels

    return ret


def spy_kmeans(X, centers_init):
    sample_weight = np.ones(X.shape[0])
    return kmeans_single_lloyd(
        X,
        sample_weight,
        centers_init
    )


def get_one_spy_kmeans(X, k, centers_generator):
    X = X.to_numpy().copy(order='C')
    centers = centers_generator(X, k)
    return spy_kmeans(X, centers)


def k_p_p(X, k):
    centers, _ = kmeans_plusplus(X, k)
    return centers


def k_random(X, k):
    return np.random.permutation(X)[:k]


def k_total_random(X, k):
    return np.array([np.random.uniform(X.min(axis=0), X.max(axis=0)) for _ in range(k)])


def measure_init_method(X, k=5, init_method='k-means++'):
    method = init_method
    centers_generator = None

    if init_method == 'k-means++':
        centers_generator = k_p_p
    if init_method == 'random':
        centers_generator = k_random
    if init_method == 'completely-random':
        centers_generator = k_total_random

    results = np.array([get_one_spy_kmeans(X, k, centers_generator)
                       for _ in range(20)])

    std = np.std(results, axis=0)
    mean_results = np.mean(results, axis=0)

    plt.errorbar(
        np.arange(mean_results.shape[0]),
        mean_results,
        std,
        linestyle='None',
        marker='o',
        ecolor='black',
        capsize=3
    )
    plt.title(f'Init method = {init_method}')
    plt.xlabel('Algorithm step')
    plt.ylabel('Calinski Harabasz Score of clustering')
    plt.show()


def get_score_for_k(X, k):
    results = [davies_bouldin_score(
        X,
        KMeans(
            n_clusters=k,
            init=k_total_random(X.to_numpy().copy(order='C'), k),
            n_init=1
        ).fit(X).labels_
    ) for _ in range(20)]
    return results


def find_best_k(X):
    start_k = 3
    end_k = 20
    k_range = range(start_k, end_k)
    results = np.array(
        [get_score_for_k(X, k) for k in k_range]
    )
    std = np.std(results, axis=1)
    mean_results = np.mean(results, axis=1)

    plt.errorbar(
        np.array(k_range),
        mean_results,
        std,
        linestyle='None',
        marker='o',
        ecolor='black',
        capsize=3
    )
    plt.title(f'Score depending on clusters count')
    plt.xlabel('k')
    plt.ylabel('Calinski Harabasz Score of clustering')
    plt.show()

    return mean_results.argmin() + start_k


def get_categories_map(data):
    categories_list = list(set(data['Category']))
    return data.apply(lambda row: categories_list.index(row['Category']), axis=1)


if __name__ == '__main__':
    data = pd.read_csv('menu.csv')
    shaped_data = shape_data(data)
    measure_init_method(shaped_data, init_method='k-means++')
    measure_init_method(shaped_data, init_method='random')
    measure_init_method(shaped_data, init_method='completely-random')
    k = find_best_k(shaped_data)
    X = shaped_data.to_numpy().copy(order='C')
    kmeans = KMeans(
        n_clusters=k,
        init=k_total_random(X, k),
        n_init=1
    ).fit(shaped_data)

    print(f'Number of clusters: {k}')
    print(f'Cluster centers: {kmeans.cluster_centers_}')

    plt.hist(kmeans.labels_,
             kmeans.cluster_centers_.shape[0], align='right', rwidth=0.5)
    plt.show()

    new_dt = pd.DataFrame({'cluster': kmeans.labels_,
                          'category': data['Category'], 'name': data['Item']})
    print(new_dt)

    pca2 = decomposition.PCA(n_components=2)
    X = pca2.fit_transform(shaped_data)
    X_centers = pca2.fit_transform(kmeans.cluster_centers_)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(X[:, 0], X[:, 1], s=5, c=kmeans.labels_,
                marker="s", label='data')
    ax1.scatter(X_centers[:, 0], X_centers[:, 1], s=25,
                marker="o", label='cluster centers')
    plt.legend(loc='upper left')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(X[:, 0], X[:, 1], s=5, c=get_categories_map(data),
                marker="s", label='data')
    ax2.scatter(X_centers[:, 0], X_centers[:, 1], s=25,
                marker="o", label='cluster centers')
    plt.legend(loc='upper left')
    plt.show()
