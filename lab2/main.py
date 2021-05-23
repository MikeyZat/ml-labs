import numpy as np
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

N_SAMPLES = 1000
cm_bright = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def add_points(centers, stds, X=None, y=None, n_samples=N_SAMPLES//3):
    X1, y1 = make_blobs(n_samples=n_samples, centers=centers,
                        shuffle=False, cluster_std=stds)

    if X is not None and y is not None:
        return np.concatenate((X, X1)), np.concatenate((y, y1))

    return X1, y1


def generate_data():
    X, y = add_points(centers=[(-90, -9), (0, 0),
                      (45, 0)], stds=[12.0, 9.0, 7.0])
    X, y = add_points(centers=[(-60, 0), (-20, 8),
                      (18, -7)], stds=[9.0, 7.0, 8.0], X=X, y=y)
    X, y = add_points(centers=[(-40, 12), (-35, 20),
                      (-22, -13)], stds=[5.0, 7.0, 8.0], X=X, y=y)
    # islands
    X, y = add_points(centers=[(-20, 2), (-60, 15), (-60, -10)],
                      stds=[2.0, 2.0, 2.0], X=X, y=y, n_samples=50)
    return X, y


def zad1():
    X, y = generate_data()
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Input data')
    plt.show()
    return X, y


def zad2(X, y):
    h = 0.5  # step size in the mesh
    figure = plt.figure(figsize=(8, 20))
    titles = [
        "k=1, metric=Euclides, weights=uniform",
        "k=13, metric=Euclides, weights=uniform",
        "k=1, metric=Mahalanobis, weights=uniform",
        "k=9, metric=Euclides, weights=distance",
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=13),
        KNeighborsClassifier(n_neighbors=1, algorithm='brute',
                             metric='mahalanobis', metric_params={'V': np.cov(X)}),
        KNeighborsClassifier(n_neighbors=9, weights='distance'),
    ]

    x_min, x_max = X[:, 0].min() - 5.0, X[:, 0].max() + 5.0
    y_min, y_max = X[:, 1].min() - 5.0, X[:, 1].max() + 5.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    i = 1
    for title, clf in zip(titles, classifiers):
        ax = plt.subplot(len(classifiers), 1, i)
        clf.fit(X, y)

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1).reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=.8)

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(title)
        i += 1


def get_k_classifcator_results(X, y, k, **kwargs):
    clf = KNeighborsClassifier(n_neighbors=k, **kwargs)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=.2)
    clf.fit(X_train, y_train)
    score = clf.score(X_validation, y_validation)
    return score


def get_k_classifcator_mean_results(X, y, k, **kwargs):
    N = 20
    results = np.array([
        [get_k_classifcator_results(X, y, k, **kwargs) for _ in range(N)]
    ])
    return np.mean(results)


def get_classificator_results(X, y, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    k_best_results = np.array(
        [get_k_classifcator_mean_results(
            X_train, y_train, k, **kwargs) for k in range(1, 21)]
    )
    best_k = np.argmax(k_best_results)
    clf = KNeighborsClassifier(n_neighbors=best_k, **kwargs)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score


def test_classifier(X, y, **kwargs):
    N = 10
    best_results = np.array(
        [get_classificator_results(X, y, **kwargs) for _ in range(N)]
    )
    return np.mean(best_results), np.std(best_results)


def zad3(X, y):
    best_score, best_score_std = test_classifier(X, y)
    worst_score, worst_score_std = test_classifier(
        X,
        y,
        algorithm='brute',
        metric='mahalanobis',
        metric_params={'V': np.cov(X)})
    scores = [best_score, worst_score]
    stds = [best_score_std, worst_score_std]
    x_pos = range(len(scores))
    x_labels = [
        'metric=Euclides,\nweights=uniform',
        'metric=Mahalanobis,\nweights=uniform'
    ]
    figure, ax = plt.subplots()
    bar = ax.bar(
        x_pos,
        [best_score, worst_score],
        yerr=[best_score_std, worst_score_std],
        align='center',
        ecolor='black',
        capsize=6)
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Score')
    ax.set_title('Classificators\' precision')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    for x in x_pos:
        ax.text(x-0.1, scores[x]/2, ('%.4f' % scores[x]), fontweight='bold')
    plt.show()


if __name__ == '__main__':
    X, y = zad1()
    zad2(X, y)
    zad3(X, y)
