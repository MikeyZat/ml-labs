import numpy as np
from matplotlib import pylab as plt

POINTS_SAMPLES_NUMBER = 1000
ATTEMPTS_NUMBER = 10
DIMENSIONS = range(1, 15)


def with_mean_std(func):
    def wrapper(results, *args, **kwargs):
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
        return func(mean, std, *args, **kwargs)
    return wrapper


def get_hypercube(a, n, points_number=POINTS_SAMPLES_NUMBER):
    return np.random.uniform(low=-(a/2), high=a/2, size=(points_number, n))


def get_inside_hypersphere_points(points, a):
    return points[np.linalg.norm(points, axis=1) <= a]


def get_inside_sphere_ratio(cube_edge, sphere_radius, dimension):
    points = get_hypercube(cube_edge, dimension)
    inside_sphere_points = get_inside_hypersphere_points(points, sphere_radius)
    percent_ratio = inside_sphere_points.shape[0] / points.shape[0]
    return percent_ratio


def zad1_subtask():
    return np.array(
        [get_inside_sphere_ratio(2.0, 1.0, n) * 100.0 for n in DIMENSIONS],
    )


@with_mean_std
def show_ratio_bar_plot(mean, std, title):
    fig, ax = plt.subplots()
    x_pos = range(len(DIMENSIONS))
    ax.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xlabel('Number of dimensions')
    ax.set_ylabel('Ratio [%]')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DIMENSIONS)
    plt.show()


def zad1():
    results = np.array([
        zad1_subtask() for _ in range(ATTEMPTS_NUMBER)
    ])
    show_ratio_bar_plot(results, 'Ratio of points from hypercube a=2.0 inside a sphere with r=1.0')
    outside_results = np.full(shape=(ATTEMPTS_NUMBER, len(DIMENSIONS)), fill_value=100.0) - results
    show_ratio_bar_plot(
        outside_results, 'Ratio of points from hypercube a=2.0 outside a sphere with r=1.0'
    )


def get_all_points_distance(cube_edge, dimension):
    points = get_hypercube(cube_edge, dimension)
    distances = np.array([
        np.linalg.norm(points - P, axis=1) for P in points
    ])
    mean = np.mean(distances)
    std = np.std(distances)
    std_mean_ratio = (std/mean) * 100.0
    return mean, std, std_mean_ratio


def zad2_subtask():
    return np.array(
        [get_all_points_distance(2.0, n) for n in DIMENSIONS],
    )


@with_mean_std
def show_scatter_plot(mean, std, ylabel, title):
    fig, ax = plt.subplots()
    x_pos = range(len(DIMENSIONS))
    ax.errorbar(x_pos, mean, yerr=std, linestyle="None", fmt='o', ecolor="black")
    ax.set_xlabel('Number of dimensions')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DIMENSIONS)
    plt.show()


def zad2():
    results = np.array([
        zad2_subtask() for _ in range(ATTEMPTS_NUMBER)
    ])
    means = results[:, :, 0]
    stds = results[:, :, 1]
    ratios = results[:, :, 2]
    show_scatter_plot(means, ylabel="distance", title="Avarage distances between points")
    show_scatter_plot(stds, ylabel="standard deviations", title="Standard deviations of distances between points")
    show_ratio_bar_plot(ratios, "Ratio of std compared to mean value")


def get_angle(points):
    u = points[:2]
    v = points[2:]
    u_abs = u[1] - u[0]
    v_abs = v[1] - v[0]
    return np.arccos(np.sum(u_abs * v_abs) / (np.linalg.norm(u_abs) * np.linalg.norm(v_abs)))


def get_all_points_radius(cube_edge, dimension):
    points = get_hypercube(cube_edge, dimension, 100000)
    SAMPLES_NUMBER = 10000
    angles = np.array([
        get_angle(points[np.random.choice(len(points), size=4, replace=False)]) for _ in range(SAMPLES_NUMBER)
    ])
    return angles


def zad3():
    dimensions = [2, 3, 5, 10]
    for n in dimensions:
        results = get_all_points_radius(2.0, n)
        plt.hist(results, bins=15, edgecolor='black')
        plt.title(f"Histogram of angles between 2 random vectors for {n} dimensions")
        plt.show()


if __name__ == '__main__':
    zad1()
    zad2()
    zad3()
