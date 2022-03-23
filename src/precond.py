
import numpy as np
import torch
from math import log

from src.chol import pivoted_chol
from src.utils import Array


class Preconditionner:
    """Abstract base class for preconditionners."""
    def __init__(self, K: torch.Tensor, k: int, sigma2: float):
        """
        :param k: rank of preconditionner
        :param sigma2: noise variance
        """
        self.n = K.shape[0]
        self.k = k
        self.sigma2 = sigma2

    def __matmul__(self, other):
        """Matrix-matrix multiplication with preconditionner matrix."""
        # Child classes must implement this
        raise NotImplementedError

    def inv_fun(self, y: Array):
        """Matrix-matrix multiplication oracle y |-> P^-1 y, i.e. with the *inverse* of the preconditionner matrix."""
        # Child classes must implement this
        raise NotImplementedError

    def logdet(self):
        """Exact computation of the log-determinant of the precondionner matrix."""
        # Child classes must implement this
        raise NotImplementedError

    def sample_gaussian(self, size: int):
        """Sample from the multivariate Gaussian distribution N(0, P) with P the preconditionner matrix."""


class PartialCholesky(Preconditionner):
    def __init__(self, K: torch.Tensor, max_rank: int, sigma2: float, tol=1e-12):
        """
        Initialize partial pivoted Cholesky preconditionner of rank at most k.
        If pivoted Cholesky converges before k steps, then the rank will be smaller.
        Pivoted cholesky computes the approximation L L^T ~= K,
        then the preconditionner is Pk_hat = L L^T + sigma2 I, I the identity.

        Important note: Pk_hat will precondition K_hat = K + sigma2 I, and *not* just K alone.

        :param K: input PSD matrix without added diagonal
        :param max_rank: maximum rank of the preconditionner
        :param sigma2: noise variance,
        """
        # Initialize parent object
        super().__init__(K, max_rank, sigma2)

        # Compute partial pivoted Cholesky
        self.Lk = pivoted_chol(K, max_rank, tol)

        # If pivoted Cholesky converged earlier than max_rank, need to update the rank
        self.k = self.Lk.shape[-1]

        # Precompute useful term (small matrix: k x k)
        self.LTL = self.Lk.T @ self.Lk

    def __matmul__(self, other):
        return torch.linalg.multi_dot((self.Lk, self.Lk.T, other)) + self.sigma2 * other

    def Pk_hat(self):
        """Construct explicitly the full preconditionner matrix Pk_hat = L L^T + sigma2 I"""
        return self.Lk @ self.Lk.T + self.sigma2 * torch.eye(self.n)

    def inv_fun(self, y: torch.Tensor):
        """See docstrings of parent class"""
        # Leverage Woodbury formula to solve a k x k system instead of n x n, k << n
        M = torch.eye(self.k) + self.LTL / self.sigma2
        z = torch.linalg.solve(M, self.Lk.T @ y)
        return y / self.sigma2 - self.Lk @ z / self.sigma2**2

    def logdet(self):
        """See docstrings of parent class"""
        # Leverage the lemma for matrix determinant (analogous to Woodbury)
        # by computing eigenvalues of k x k matrix instead of n x n, k << n
        eigs = torch.linalg.eigvalsh(self.LTL)
        eigs = (1 + eigs / self.sigma2).log().sum()
        logdet = self.n * log(self.sigma2) + eigs
        return float(logdet)

    def sample_gaussian(self, size: int):
        """See docstrings of parent class"""
        # Leverage re-parametrization trick: if covariance matrix has decomposition S = MM^T,
        # then easy to sample from N(0, S)
        M1 = torch.randn((self.k, size), dtype=self.Lk.dtype, device=self.Lk.device)
        M2 = torch.randn((self.n, size), dtype=self.Lk.dtype, device=self.Lk.device)
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

