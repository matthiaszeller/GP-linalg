

from __future__ import annotations

from copy import deepcopy
from math import log

import numpy as np
import torch

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
        # Store a reference to the kernel matrix
        self.K = K

    def __matmul__(self, other):
        """Matrix-matrix multiplication with preconditionner matrix."""
        # Child classes must implement this
        raise NotImplementedError

    def compute_precond_eigs(self):
        """
        Compute the eigenvalues of the preconditionned matrix
                    Phat^-1/2 Khat Phat^-1/2
        in O(n^3) time. This is **not** used for the inference process (because of the complexity!),
        rather a utility function to analyze numerical properties of the algorithms.
        """
        # The spectrum of those two matrices are the same:
        # - Phat^-1/2 Khat Phat^-1/2
        # - Phat^-1 Khat
        # => eigenvalues are real

        # We rather compute the second matrix
        precond_mx = self.inv_fun(self.K + torch.eye(self.n) * self.sigma2)
        # Compute explicitly (i.e, inefficiently) the eigenvalues
        precond_eigs = torch.linalg.eigvals(precond_mx)
        # We know the eigenvalues are real, see the above remark about equality of spectrum
        precond_eigs = torch.real(precond_eigs)

        return precond_eigs

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
        raise NotImplementedError


class PartialCholesky(Preconditionner):
    def __init__(self, K: torch.Tensor, max_rank: int, sigma2: float, tol=1e-12,
                 use_tensorflow_algorithm: bool = False, verbose: bool = False):
        """
        Initialize partial pivoted Cholesky preconditionner of rank at most k.
        If pivoted Cholesky converges before k steps, then the rank will be smaller.
        Pivoted cholesky computes the approximation L L^T ~= K,
        then the preconditionner is Pk_hat = L L^T + sigma2 I, I the identity.

        Important note: Pk_hat will precondition K_hat = K + sigma2 I, and *not* just K alone.

        :param K: input PSD matrix without added diagonal
        :param max_rank: maximum rank of the preconditionner
        :param sigma2: noise variance
        :param tol: tolerance for pivoted Cholesky algorithm, see docstring of `pivoted_chol`
        :param use_tensorflow_algorithm: for numerical experiments, use fast implementation
        """
        # Initialize parent object
        super().__init__(K, max_rank, sigma2)

        # Compute partial pivoted Cholesky, store the trace errors at each step as made available by the algo
        self.trace_errors = []
        if use_tensorflow_algorithm is False:
            self.Lk, self.pivots = pivoted_chol(K, max_rank, tol,
                                                callback=lambda err_k: self.trace_errors.append(err_k),
                                                return_pivots=True, verbose=verbose)
            self.trace_errors = torch.tensor(self.trace_errors)
        else:
            import tensorflow as tf
            import tensorflow_probability as tfp
            self.Lk = tfp.math.pivoted_cholesky(tf.convert_to_tensor(K), max_rank, diag_rtol=tol)
            self.Lk = torch.from_numpy(self.Lk.numpy())

        # If pivoted Cholesky converged earlier than max_rank, need to update the rank
        self.k = self.Lk.shape[-1]

        # Precompute useful term (small matrix: k x k)
        self.LTL = self.Lk.T @ self.Lk

    def __matmul__(self, other):
        return torch.linalg.multi_dot((self.Lk, self.Lk.T, other)) + self.sigma2 * other

    def truncate(self, k: int) -> PartialCholesky:
        """
        Decrease the rank of the pivoted Cholesky approximation,
        i.e, return a copy of the PartialCholesky object and process it as if only k steps were done.
        Used to speedup numerical experiments (avoid rerunning the algo).
        :param k: the desired lower rank, must be <= current rank
        :return: new PartialCholesky object
        """
        assert k <= self.k
        P = deepcopy(self)
        if k == self.k:
            return P # just return an unaltered copy

        # Update new rank
        P.k = k
        P.Lk = P.Lk[:, :k]
        P.pivots = P.pivots[:k]
        P.trace_errors = P.trace_errors[:k+1]
        # Recompute k x k matrix
        P.LTL = P.Lk.T @ P.Lk
        return P

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

