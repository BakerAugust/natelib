import pandas as pd
import numpy as np
from numpy.core.numeric import allclose
from sklearn.decomposition import PCA as skPCA
from sklearn import datasets
import matplotlib.pyplot as plt
from pedagogy.pca import svd, PCA

np.set_printoptions(suppress=True)


def test_svd() -> None:
    A = np.array([[1, 2, 1],
                  [3, 4, 1],
                  [5, 6, 1]])
    U, S, V = svd(A)
    # U_, S_, V_ = LA.svd(A, full_matrices=True)
    # smat = np.diag(S)
    print(U.shape, S.shape, V.shape)
    # print(U_.shape, S_.shape, V_.shape)
    print(V)
    print(U)
    print(A, np.dot(U[:, :3] * S, V))


def test_PCA() -> None:
    # Test against SKlearn using iris data
    iris = datasets.load_iris()
    y = iris.target
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    pca = PCA()
    pca.fit(data)

    new_X = pca.transform(2)
    fig, axarr = plt.subplots(1,2)
    axarr[0].scatter(new_X[:, 0], new_X[:, 1], c=y)
    print(pca.singular_values)
    print(pca.components)

    skpca = skPCA(n_components=2)
    sk_new_X = skpca.fit_transform(pca.data)
    print(skpca.singular_values_)
    print(skpca.components_)
    axarr[1].scatter(sk_new_X[:, 0], sk_new_X[:, 1], c=y)
   
    plt.show()
    assert allclose(sk_new_X, new_X[:, :2])


if __name__ == '__main__':
    # test_svd()
    test_PCA()
    

