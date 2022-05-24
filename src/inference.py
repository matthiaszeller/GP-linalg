from typing import Callable, Union, List, Dict, Any

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
    if k > 0:
        P = PartialCholesky(K.K, k, K.sigma2)
        Pinvfun = P.inv_fun
        Plogdet = P.logdet()
        sampler = P.sample_gaussian
        rank = P.k
    else:
        Pinvfun = lambda x: x
        Plogdet = 0.0
        sampler = lambda N: torch.randn(n, N)
        rank = 0

    # Probe vectors sampled from gaussian with covariance matrix Pk_hat
    Z = sampler(N)
    # Add the y vector
    Z = torch.concat((y.reshape(-1, 1), Z), dim=1)

    # Run mBCG to compute partial tridiagonalizations
    X, Ts = mbcg(K.Khat_fun, Pinvfun, Z, torch.zeros_like(Z), m, tol=mbcg_tol)

    # For each probe vector zi, compute lanczos quadrature to estimate zi^T log(Khat) zi
    # the first tridiag matrix corresponds to y, we discard it
    Ts = Ts[1:]
    estimates = np.array([
        lanczos_quadrature(f=torch.log, Tm=Ts[i], z=Z[:, i+1], matrix_size=n)
        for i in range(N)
    ])

    if info is not None:
        info['niter'] = Ts[0].shape[0]
        info['rank'] = rank

    # Average the trace estimators to get approximate of log det(M),
    # where M is the preconditionned SPD matrix
    if return_avg:
        precond_logdet = np.mean(estimates)
    else:
        precond_logdet = estimates

    # Retrieve the logdet of the original matrix
    logdet = precond_logdet + Plogdet

    # Retrieve the y solve
    ysolve = X[:, 0]

    # Compute the trace term for each kernel hyperparameter
    traces = []
    for dKdthetai in K.grad:
        # Compute product of grad matrix with probe vectors
        V = dKdthetai @ Pinvfun(Z[:, 1:]) # discard first column which is the y vector
        # Perform dot products
        dots = (X[:, 1:] * V).sum(dim=0) # discard first column of X corresponding to y vector
        # Compute monte carlo estimator, i.e. mean of dot products
        traces.append(dots.mean().item())
    traces = torch.tensor(traces)

    return ysolve, logdet, traces



def inference_experiments(y: torch.Tensor, K: Kernel, k: int, Ntot: int, ms: List[int],
                          mbcg_tol, config: Dict[str, Any]):
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
    if k > 0:
        P = PartialCholesky(K.K, k, K.sigma2)
        logdet_precond = P.logdet()
        Pinvfun = P.inv_fun
        # Probe vectors sampled from gaussian with covariance matrix Pk_hat
        Z = P.sample_gaussian(size=Ntot)
    else:
        Pinvfun = lambda x: x
        logdet_precond = 0
        Z = torch.randn(n, Ntot)

    # Add the y vector
    mbcg_RHS = torch.concat((y.reshape(-1, 1), Z), dim=1)

    # Run mBCG to compute partial tridiagonalizations
    max_iter = max(ms)
    Xs = []
    callback = lambda Xk: Xs.append(Xk)
    _, Ts = mbcg(K.Khat_fun, Pinvfun, mbcg_RHS, torch.zeros_like(mbcg_RHS), max_iter, tol=mbcg_tol, callback=callback)
    # Xs is tensor of size (mbcg_iter+1) x n x Ntot
    Xs = torch.stack(Xs)

    # Index zero of last dimension is for the y solve
    ysolves, Xs = Xs[:, :, 0], Xs[:, :, 1:]
    # Groundtruth
    ysolve_true = torch.linalg.solve(K.Khat(), y)

    results_ysolves = []
    for m in ms:
        results_ysolves.append({
            **config,
            'niter': m,
            'k': k,
            'ysolve_estim': ysolves[m],
            'ysolve_error': ((ysolve_true - ysolves[m]).norm() / ysolve_true.norm()).item()
        })

    # Logdet groundtruth
    logdet_true = torch.logdet(K.Khat())
    # For each probe vector zi, compute lanczos quadrature to estimate zi^T log(Khat) zi
    # the first tridiag matrix corresponds to y, we discard it
    Ts = Ts[1:]
    results_logdet = []
    for m in ms:
        logdet_estimates = []
        for i in range(Ntot):
            T = Ts[i]
            z = Z[:, i]
            Tsub = T[:m, :m]
            logdet_estimates.append(lanczos_quadrature(torch.log, Tsub, z, n))

        logdet_estimates = torch.tensor(logdet_estimates)
        logdet_errors = ((logdet_estimates + logdet_precond - logdet_true) / logdet_true).abs()
        results_logdet.append({
            **config,
            'niter': m,
            'k': k,
            'logdet_estim': logdet_estimates,
            'logdet_precond': logdet_precond,
            'logdet_error': logdet_errors
        })

    # Traces groundtruth
    A = torch.linalg.solve(K.Khat(), K.grad)
    traces_true = torch.tensor([M.trace() for M in A]).reshape(-1, 1)
    # Compute the trace term for each kernel hyperparameter
    PinvZ = Pinvfun(Z)
    # Gradient 3D tensor times preconditioned probe vectors
    V = K.grad @ PinvZ
    results_traces = []
    for m in ms:
        # Get the CG iterate corresponding to m-th iteration
        # Because first dimension of Xs starts with initial guess, no need for m-1 indexing
        Xk = Xs[m, :, :]
        # Perform dot products
        trace_estimators = (Xk * V).sum(dim=1)
        # Trace estimators is of shape n_hyperparams x Ntot
        results_traces.append({
            **config,
            'niter': m,
            'traces_estim': trace_estimators,
            'traces_error': ((trace_estimators - traces_true) / traces_true).abs()
        })

    return results_ysolves, results_logdet, results_traces


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

    n = 1000
    y = torch.zeros(n)
    x = torch.randn(n, 2)

    sigma2, lengthscale = 0.1, 0.1

    K = SquaredExponentialKernel(x)
    K.compute_kernel_and_grad(torch.tensor([sigma2, lengthscale]))

    info = dict()
    ysolve, logdet, traces = inference(y, K, k=50, N=10, m=50, return_avg=False, info=info)
    niter=  info['niter']

    Khatinv = torch.linalg.inv(K.Khat())
    matmul = Khatinv @ K.grad

    traces_true = torch.tensor([
        M.trace() for M in matmul
    ])

    logdet_true = torch.logdet(K.Khat())
    err_logdet = ((logdet.mean() - logdet_true) / logdet_true).abs()
