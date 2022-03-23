from typing import Callable

import numpy as np
import torch

from src.cg import mbcg
from src.precond import PartialCholesky
from src.quadrature import lanczos_quadrature


def inference(K: torch.Tensor, k: int, sigma2: float, N: int, m: int):
    """
    Gaussian process inference engine. Given a kernel matrix K, compute:
        - log det(K_hat)
        -

    :param K:
    :param k:
    :param m:
    :param sigma2:
    :return:
    """
    n = K.shape[0]

    # Compute partial Cholesky preconditionner
    P = PartialCholesky(K, k, sigma2)

    # Matrix with added diagonal
    Khat = K + torch.eye(n) * sigma2

    # Probe vectors sampled from gaussian with covariance matrix Pk_hat
    #Z = torch.randn(n, N)
    Z = P.sample_gaussian(size=N)

    # Run mBCG to compute partial tridiagonalizations
    _, Ts = mbcg(lambda X: Khat @ X, P.inv_fun, Z, torch.zeros_like(Z), m)

    # For each probe vector zi, compute lanczos quadrature to estimate zi^T log(Khat) zi
    estimates = np.array([
        lanczos_quadrature(f=torch.log, Tm=Ts[i], z=Z[:, i], matrix_size=n)
        for i in range(N)
    ])

    # Average the trace estimators to get approximate of log det(M),
    # where M is the preconditionned SPD matrix
    precond_logdet = np.mean(estimates)

    # Retrieve the logdet of the original matrix
    logdet = precond_logdet + P.logdet()

    return logdet


def compute_logdet(A: torch.Tensor, N: int, m: int, return_avg: bool = True):
    """
    Compute the log determinant of the SPD matrix A with stochastic lanczos quadrature.
    This is mainly for illustrative purpose as this does not use preconditionning.

    :param A: input matrix
    :param N: number of probe vectors
    :param m: number of CG and Lanczos steps
    :return:
    """
    # Generate standard normal probe vectors
    n = A.shape[0]
    Z = torch.randn(n, N, dtype=A.dtype, device=A.device)
    X0 = torch.zeros(n, N, dtype=A.dtype, device=A.device)

    # Compute partial Lanczos tridiagonalization for each probe vector
    _, Ts = mbcg(lambda X: A@X, lambda X: X, Z, X0, m)

    # Compute Lanczos quadrature for each probe vector
    estimates = np.array([
        lanczos_quadrature(f=torch.log, Tm=Tm_i, z=z_i, matrix_size=n)
        for Tm_i, z_i in zip(Ts, Z.T)
    ])

    if return_avg is False:
        return estimates
    estimate = np.mean(estimates)
    return estimate


if __name__ == '__main__':
    from scipy.stats import ortho_group
    torch.set_default_dtype(torch.double)

    n, m, N = 1000, 50, 50
    eigs = torch.linspace(1, 1000, n)
    Q = torch.from_numpy(ortho_group.rvs(n))
    A = Q @ torch.diag(eigs) @ Q.T
    logdet = eigs.log().sum()
    sanity_check = torch.logdet(A)

    estimate = compute_logdet(A, N, m)
    err = abs(logdet - estimate)
    relerr = err / logdet

    # n = 100
    # Q = torch.from_numpy(ortho_group.rvs(n))
    # eigs = torch.linspace(1, 10, n)
    # A = Q @ torch.diag(eigs) @ Q.T
    # sigma2 = 2.0
    # Ahat = A + sigma2 * torch.eye(n)
    # logdet_true = torch.logdet(Ahat)
    # print('true', logdet_true)
    #
    # k, N, m = 30, 10, 10
    # logdet = inference(A, k, sigma2, N, m)
    #
    # relerr = abs(logdet_true - logdet) / logdet_true
    #

