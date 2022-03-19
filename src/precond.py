
import numpy as np
import torch
from math import log

from src.chol import pivoted_chol
from src.utils import Array


class Preconditionner:
    def __init__(self, k: int, sigma2: float):
        """
        :param k: rank of preconditionner
        :param sigma2: noise variance
        """
        self.k = k
        self.sigma2 = sigma2

    def __matmul__(self, other):
        raise NotImplementedError

    def inv_fun(self, y: Array):
        raise NotImplementedError


class PartialCholesky(Preconditionner):
    def __init__(self, K: torch.Tensor, k: int, sigma2: float):
        super(PartialCholesky, self).__init__(k, sigma2)
        # Compute partial pivoted Cholesky
        self.Lk = pivoted_chol(K, k)
        # Precompute useful term
        self.LTL = self.Lk.T @ self.Lk

    def inv_fun(self, y: torch.Tensor):
        M = torch.eye(self.k) + self.LTL / self.sigma2
        z = torch.linalg.solve(M, self.Lk.T @ y)
        return y / self.sigma2 - self.Lk @ z / self.sigma2**2

    def logdet(self):
        eigs = torch.linalg.eigvalsh(self.LTL)
        eigs = (1 + eigs / self.sigma2).log().sum()
        n = self.Lk.shape[0]
        logdet = n * log(self.sigma2) + eigs
        return float(logdet)

    def sample_gaussian(self, size: int):
        n = self.Lk.shape[0]
        M1 = torch.randn((self.k, size), dtype=self.Lk.dtype, device=self.Lk.device)
        M2 = torch.randn((n, size), dtype=self.Lk.dtype, device=self.Lk.device)
        Y = self.Lk @ M1 + self.sigma2 ** 0.5 * M2
        return Y


def sample_gaussian(Lk: Array, size: int, sigma2: float): # TODO WARNING sigma not sigma2
    """
    Sample from the Gaussian distribution N(0, P_k^hat), P_k^hat = L_k L_k^T + sigma2 I,
    using the re-parametrization trick for multivariate Gaussian.
    :param Lk: low-rank Cholesky factor
    :param size: number of vectors to sample
    :return: array of shape n x size
    """
    # For single vector, use re-parametrization trick:
    # Y = Lk X_1 + sigma X_2 ~ N(0, P_k^hat) if X_1 ~ N(0, I_k), X_2(0, I_n)
    # Get get all samples at once, just sample two matrices:
    # M1 of shape k x size, M2 of shape n x size, each with iid standard normal entries
    # Then Y = Lk M + sigma M2 has each column vectors distributed as desired
    n, k = Lk.size()
    M1 = torch.randn((k, size), dtype=Lk.dtype, device=Lk.device)
    M2 = torch.randn((n, size), dtype=Lk.dtype, device=Lk.device)
    Y = Lk @ M1 + sigma2**0.5 * M2
    return Y


def precond_logdet(Lk, sigma2):
    """
    Use generalized matrix determinant lemma to compute determinant of preconditionner Pk^hat = L_k L_k^T + sigma2 I.
    :param Lk: rank k Cholesky factor
    :param sigma2: variance of the noise
    :return:
    """
    n, k = Lk.shape
    M = Lk.T @ Lk
    eigs = torch.linalg.eigvalsh(M)
    eigs = (1 + eigs/sigma2).log().sum()
    return n * log(sigma2) + eigs
    # n, k = Lk
    # detk = np.linalg.det(np.eye(k) + Lk.T @ Lk)
    # return np.log(detk) + n*np.log(sigma2)


def pinv_fun(Lk: torch.Tensor, sigma2: float):
    """
    Returns a function x |-> P^-1 x for P = Lk Lk^T + sigma2 I,
    i.e. the preconditionner P is a rank k partial cholesky decomposition plus a noise term.
    Uses the Woodbury formula to compute in O(nk^2) time instead of O(n^3).

    :param Lk: the rank k Cholesky factor
    :param sigma2: variance of the noise
    """
    k = Lk.shape[1]
    # Precompute
    M = torch.eye(k) + Lk.T @ Lk / sigma2

    def inner(y: torch.Tensor):
        z = torch.linalg.solve(M, Lk.T @ y)
        return y / sigma2 - Lk @ z / sigma2**2

    return inner


if __name__ == '__main__':
    # Lk = np.diag([3., 2., 1.])[:, :2]
    # sigma2 = 1
    # Pk = Lk @ Lk.T + sigma2 * np.eye(3)
    # Pkinv = np.diag(1/np.diag(Pk))
    # y = np.random.randn(3)#np.ones(3)
    # fun = pinv_fun(Lk, sigma2)
    # x = fun(y)
    # xtrue = Pkinv @ y

    n, k = 1000, 30
    Lk = np.tril(np.random.randn(n, n))[:, :k]
    sigma2 = 2.
    Pk = Lk @ Lk.T + np.eye(n) * sigma2
    y = np.random.randn(n, 1)
    xtrue = np.linalg.solve(Pk, y)
    x = pinv_fun(Lk, sigma2)(y)
    xerr = np.abs(xtrue - x).max()

    Lk = torch.tril(torch.randn(n, k, dtype=torch.double))
    Pkhat = Lk @ Lk.T + torch.eye(n) * sigma2
    logdet_torch = torch.linalg.slogdet(Pkhat).logabsdet
    logdet = precond_logdet(Lk, sigma2)
    logdet_err = abs(logdet - logdet_torch)
    # Timeit! precond_logdet much faster

    # Sampling
    Y = sample_gaussian(Lk, size=10000, sigma2=sigma2)
    # Empirical covariance matrix
    S = torch.cov(Y)
    S_err = torch.abs(S - Pkhat).max()

