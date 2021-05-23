from PIL import Image
import numpy as np
from sklearn import decomposition
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import os

IMAGE_SHAPE = (100, 100)
cm = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#000000'])

# 1 - fork - red, 2 - knife - green, 3 - spoon - blue, 4 - teaspoon - black
C = [
    2,
    1,
    1,
    3,
    4,
    1,
    1,
    3,
    2,
    4,
    2,
    3,
    2,
    4,
    3,
    3,
    4,
    4,
    2,
    2,
    3,
    4,
    1,
    3,
    1,
    1,
    1,
    4,
    4,
    2,
    3,
    2,
    3,
    4,
    2,
    2,
    4,
    4
]


# images shaping
def load_and_shape_images():
    file_list = os.listdir(('images'))

    images = []

    for file in file_list:
        img = Image.open(f'images/{file}').convert(mode='L')
        img.thumbnail(IMAGE_SHAPE)
        images.append(np.array(img).flatten())

    return np.array(images)


def plot_head_images(img_array):
    fig = plt.figure(figsize=(16, 9))
    columns = 5
    rows = 2

    # first 5
    for i in range(0, 10):
        ax = fig.add_subplot(rows, columns, i+1)
        plt.imshow(img_array[i].reshape(IMAGE_SHAPE), cmap='gray')

    plt.show()


def center_images(img_array):
    mean_img = np.mean(img_array, axis=0)
    img_array = np.subtract(img_array, mean_img)
    Image.fromarray(mean_img.reshape(IMAGE_SHAPE)).show()
    return img_array


def show_components_and_variance(pca, img_array):
    pca.fit(img_array)

    fig = plt.figure(figsize=(16, 9))
    columns = 5
    rows = 2

    # first 5
    for i in range(0, 5):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_title(f"component {i}")
        plt.imshow(pca.components_[i].reshape(IMAGE_SHAPE), cmap='gray')

    # last 5
    offset = pca.components_.shape[0] - 5
    for i in range(offset, offset + 5):
        ax = fig.add_subplot(rows, columns, i - offset + 6)
        ax.set_title(f"component {i}")
        plt.imshow(pca.components_[i].reshape(IMAGE_SHAPE), cmap='gray')

    plt.show()

    plt.scatter(
        np.arange(pca.explained_variance_ratio_.shape[0]),
        pca.explained_variance_ratio_,
    )
    plt.title("Variance ratio for each component")
    plt.show()


def apply_pca(pca, img_array):
    tmp = pca.fit_transform(img_array)
    new_images = pca.inverse_transform(tmp)
    plot_head_images(new_images)


if __name__ == '__main__':
    images_array = load_and_shape_images()
    # plot example images
    plot_head_images(images_array)
    # center and show mean img
    centered_images = center_images(images_array)
    # create pca components
    pca = decomposition.PCA()
    pca2 = decomposition.PCA(n_components=2)
    pca4 = decomposition.PCA(n_components=4)
    pca16 = decomposition.PCA(n_components=16)
    # show components and variance ratio
    show_components_and_variance(pca, centered_images.copy())
    # test pca4 and pca16
    apply_pca(pca4, centered_images)
    apply_pca(pca16, centered_images)
    # test pca2 and plot on 2D map
    X = pca2.fit_transform(centered_images)
    plt.scatter(X[:, 0], X[:, 1], c=C, cmap=cm)
