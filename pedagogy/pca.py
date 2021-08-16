"""Principal component analysis"""
from preprocess.preprocess import center, minmax_scale
from typing import Tuple
import numpy as np
from numpy.random import default_rng
from numpy import linalg as LA
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.utils.extmath import svd_flip


def svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Singular value decomposition.

    Params
    --------
    A : np.ndarray

    Returns
    --------
    U : np.ndarray - left singular vectors
    Sigma: np.ndarray - matrix of singular values on the diagonal
    V : np.ndarray - right singular vectors
    """
    # _, U = LA.eig(np.matmul(A, A.transpose()))
    lamdas, V = LA.eig(np.matmul(A.transpose(), A))
    S = np.sqrt(lamdas)
    
    # Get U based on V and S
    U = np.dot(A, LA.inv(V.transpose())) / S  # inefficient, but ¯\_(ツ)_/¯
    return U, S, V.transpose()


class PCA:
    @staticmethod
    def _calc_exlpained_variance_ratio(singular_values: np.ndarray) -> np.ndarray:
        """Estimates explained variance for the singular values"""
        n = singular_values.shape[0]
        explained_variance = (singular_values ** 2) / (n - 1)
        total_var = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_var

        return explained_variance_ratio

    def fit(self, df: DataFrame, centered: bool = False, scaled: bool = False) -> None:
        """fits data"""
        self.labels = df.columns
        self.data = df.values

        if not centered:
            self.data = center(self.data)
        if not scaled:
            self.data = minmax_scale(self.data)

        U, S, Vt = svd(self.data)
        U, Vt = svd_flip(U, Vt)  # Ensures deterministic output

        self.components = Vt
        self.singular_values = np.flip(np.sort(S))
        self.explained_variance_ratio = PCA._calc_exlpained_variance_ratio(
            self.singular_values
        )

    def transform(self, n_components: int) -> np.ndarray:
        """
        Transforms data into specified number of principal components
        """
        return np.dot(self.data, self.components[:n_components].transpose())

    def loading_plot(self, index: int) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a chart with the weights for the specified component.
        """
        fig, ax = plt.subplots(1, 1)
        ax.bar(self.labels, self.components[index])
        return fig, ax

    def explained_variance_plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a plot with the cumulative explained variance.
        """
        fig, ax = plt.subplots(1, 1)
        n = self.explained_variance_ratio.shape[0]
        cumu_explained_var = [
            np.sum(self.explained_variance_ratio[0:x]) for x in np.arange(0, n)
        ]

        ax.plot(np.arange(0, n), cumu_explained_var)
        ax.set_xlabel("N components")
        ax.set_ylabel("Explained variance")

        return fig, ax
