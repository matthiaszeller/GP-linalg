from typing import Callable, Union, List

import numpy as np
import torch

from src.cg import mbcg
from src.kernel import SquaredExponentialKernel, Kernel
from src.precond import PartialCholesky
from src.quadrature import lanczos_quadrature

#
# def inference(Xtrain, y: torch.Tensor, kernel: Kernel, k: int, N: int, m: int, return_avg: bool = True, **kwargs):
#     K, grad = kernel.compute_kernel_and_grad(**kwargs)
#     n = K.shape[0]
#
#     # Compute partial Cholesky preconditionner
#     P = PartialCholesky(K, k, sigma2)
#
#     # Matrix with added diagonal
#     Khat = K + torch.eye(n) * sigma2
#
#     # Probe vectors sampled from gaussian with covariance matrix Pk_hat
#     Z = P.sample_gaussian(size=N)
#     # Add the y vector
#     if y.ndim == 1:
#         y = y.view(-1, 1)
#     Z = torch.concat((y, Z), dim=1)
#
#     # Run mBCG to compute partial tridiagonalizations
#     X, Ts = mbcg(lambda X: Khat @ X, P.inv_fun, Z, torch.zeros_like(Z), m)
#
#     # Discard the first tridiag matrix: related to y, must not be used for logdet
#     Ts.pop(0)
#     # Get the solve for y
#     ysolve = X[:, 0]
#
#     # For each probe vector zi, compute lanczos quadrature to estimate zi^T log(Khat) zi
#     estimates = np.array([
#         lanczos_quadrature(f=torch.log, Tm=Ts[i], z=Z[:, i], matrix_size=n)
#         for i in range(N)
#     ])
#
#     # Average the trace estimators to get approximate of log det(M),
#     # where M is the preconditionned SPD matrix
#     if return_avg:
#         precond_logdet = np.mean(estimates)
#     else:
#         precond_logdet = estimates
#
#     # Retrieve the logdet of the original matrix
#     logdet = precond_logdet + P.logdet()
#
#     return logdet


def inference(y: torch.Tensor, K: Kernel, k: int, N: int, m: int,
              mbcg_tol: float = 1e-10, return_avg=True, info=None):
    """
    Gaussian process inference engine. Given a kernel matrix K, compute:
        - log det(K_hat)
        - linear solve Khat^{-1} y
        - trace term

    :param K: kernel matrix
    :param k: max rank of preconditionner
    :param m: number of conjugate gradient (and Lanczos) steps
    :param sigma2: noise variance
    :return:
    """
    n = K.K.shape[0]

    # Compute partial Cholesky preconditionner
    P = PartialCholesky(K.K, k, K.sigma2)

    # Probe vectors sampled from gaussian with covariance matrix Pk_hat
    #Z = torch.randn(n, N)
    Z = P.sample_gaussian(size=N)
    # Add the y vector
    Z = torch.concat((y.reshape(-1, 1), Z), dim=1)

    # Run mBCG to compute partial tridiagonalizations
    X, Ts = mbcg(K.Khat_fun, P.inv_fun, Z, torch.zeros_like(Z), m, tol=mbcg_tol)

    # For each probe vector zi, compute lanczos quadrature to estimate zi^T log(Khat) zi
    # the first tridiag matrix corresponds to y, we discard it
    Ts = Ts[1:]
    estimates = np.array([
        lanczos_quadrature(f=torch.log, Tm=Ts[i], z=Z[:, i], matrix_size=n)
        for i in range(N)
    ])

    if info is not None:
        info['niter'] = Ts[0].shape[0]

    # Average the trace estimators to get approximate of log det(M),
    # where M is the preconditionned SPD matrix
    if return_avg:
        precond_logdet = np.mean(estimates)
    else:
        precond_logdet = estimates

    # Retrieve the logdet of the original matrix
    logdet = precond_logdet + P.logdet()

    # Retrieve the y solve
    ysolve = X[:, 0]

    # Compute the trace term for each kernel hyperparameter
    traces = []
    for dKdthetai in K.grad:
        # Compute product of grad matrix with probe vectors
        V = dKdthetai @ P.inv_fun(Z[:, 1:]) # discard first column which is the y vector
        # Perform dot products
        dots = (X[:, 1:] * V).sum(dim=0) # discard first column of X corresponding to y vector
        # Compute monte carlo estimator, i.e. mean of dot products
        traces.append(dots.mean().item())
    traces = torch.tensor(traces)

    return ysolve, logdet, traces


def compute_logdet(A: torch.Tensor, N: int, m: Union[int, List[int]], return_avg: bool = True):
    """
    Compute the log determinant of the SPD matrix A with stochastic lanczos quadrature.
    This is mainly for illustrative purpose as this does not use preconditionning.

    :param A: input matrix
    :param N: number of probe vectors
    :param m: number of CG and Lanczos steps, can be a list
    :return: logdet estimator. If return_avg is False, then return the N estimates.
             If in addition m is a list, returns m lists of N estimates
    """
    # Generate standard normal probe vectors
    n = A.shape[0]
    Z = torch.randn(n, N, dtype=A.dtype, device=A.device)
    X0 = torch.zeros(n, N, dtype=A.dtype, device=A.device)

    # Compute partial Lanczos tridiagonalization for each probe vector
    if isinstance(m, int):
        # Get tridiagonal matrices form mBCG
        _, Ts = mbcg(lambda X: A@X, lambda X: X, Z, X0, m)

        # Compute Lanczos quadrature for each probe vector
        estimates = np.array([
            lanczos_quadrature(f=torch.log, Tm=Tm_i, z=z_i, matrix_size=n)
            for Tm_i, z_i in zip(Ts, Z.T)
        ])
    else:
        # m is a list: run mBCG only once with the max number of steps, then estimators for smaller m values
        # are obtained by truncating the Hessenberg matrix
        m_mbcg = max(m)
        _, Ts = mbcg(lambda X: A@X, lambda X: X, Z, X0, m_mbcg)

        estimates = []
        # Iteration over truncation points
        for m_ in m:
            estimates.append(np.array([
                lanczos_quadrature(f=torch.log, Tm=Tm_i[:m_, :m_], z=z_i, matrix_size=n)
                for Tm_i, z_i in zip(Ts, Z.T)
            ]))
        estimates = np.array(estimates)

    if return_avg is False:
        return estimates

    estimate = np.mean(estimates, axis=-1)
    return estimate


if __name__ == '__main__':
    from scipy.stats import ortho_group
    torch.set_default_dtype(torch.double)

    n = 100
    y = torch.zeros(n)
    x = torch.linspace(0, 1, n)

    sigma2, lengthscale = 0.1, 1.0

    K = SquaredExponentialKernel(x)
    K.compute_kernel_and_grad(torch.tensor([sigma2, lengthscale]))

    ysolve, logdet, traces = inference(y, K, k=10, N=10, m=10)

    Khatinv = torch.linalg.inv(K.Khat())
    matmul = Khatinv @ K.grad

    traces_true = torch.tensor([
        M.trace() for M in matmul
    ])
