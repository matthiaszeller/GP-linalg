from typing import Callable

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
    # Compute partial Cholesky
    P = PartialCholesky(K, k, sigma2)
    # Sample from preconditionner
    Z = P.sample_gaussian(size=N)
    # Compute kernel matrix with diagonal
    Khat = K + sigma2 * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    # mBCG
    X0 = torch.zeros(K.shape[0], N, dtype=K.dtype, device=K.device)
    X, Ts = mbcg(lambda X: Khat @ X, P.inv_fun, Z, X0, m)
    # Compute Lanczos quadrature and average
    precond_logdet = torch.tensor([
        lanczos_quadrature(torch.log, Tm_i, zi) for Tm_i, zi in zip(Ts, Z.T)
    ]).mean()
    # Retrieve the logdet of the original matrix
    logdet = precond_logdet + P.logdet()

    return logdet


def compute_logdet(Khat: torch.Tensor, N: int, m: int):
    """
    Compute the log determinant of the SPD kernel matrix Khat with stochastic lanczos quadrature.
    This is mainly for debugging purpose.

    :param Khat_fun: matrix-matrix multiplication oracle X |-> Khat @ X
    :param pinv_fun: matrix-matrix multiplication oracle with preconditionner
    :param N: number of probe vectors
    :param m: number of Lanczos steps
    :return:
    """
    n = Khat.shape[0]
    # Probe vectors
    Z = torch.randn(n, N, dtype=Khat.dtype, device=Khat.device)
    X0 = torch.zeros(n, N, dtype=Khat.dtype, device=Khat.device)
    # mBCG call
    X, Ts = mbcg(lambda X: Khat @ X, lambda X: X, Z, X0, m)
    # Lanczos quadrature
    estimates = torch.tensor([
        lanczos_quadrature(torch.log, Tm_i, zi) for Tm_i, zi in zip(Ts, Z.T)
    ])

    estimate = torch.mean(estimates)
    return estimate


if __name__ == '__main__':
    from scipy.stats import ortho_group
    torch.set_default_dtype(torch.double)

    # n, m, N = 100, 50, 20
    # eigs = torch.linspace(1, 1000, n)
    # Q = torch.from_numpy(ortho_group.rvs(n))
    # A = Q @ torch.diag(eigs) @ Q.T
    # logdet = eigs.log().sum()
    # sanity_check = torch.logdet(A)
    #
    # estimate = compute_logdet(A, N, m)
    # err = abs(logdet - estimate)
    # relerr = err / logdet

    n, m, k, N = 100, 20, 30, 20
    eigs = 10**torch.linspace(0, 5, n)
    Q = torch.from_numpy(ortho_group.rvs(n))
    A = Q @ torch.diag(eigs) @ Q.T

    sigma2 = 2.0
    logdet = inference(A, k, sigma2, N, m)
    true = torch.logdet(A)
    relerr = abs(logdet - true) / true

