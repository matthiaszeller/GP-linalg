

from typing import Callable

import torch

from src.cg import mbcg


def lanczos_quadrature(f: Callable, Tm: torch.Tensor, z: torch.Tensor):
    """
    Approximation of quadratic form z^T f(A) z with Lanczos quadrature, A symmetric matrix where the function
    f is analytic in the spectral interval of f(A).

    :param Tm: Lanczos tridiagonal matrix of A with starting vector z
    :param z: probe vector used to start Lanczos iteration
    :return: approximation norm(z)^2 * e1^T f(Tm) e1
    """
    # Spectral decomposition of Tm # TODO use specialized algorithm for tridiag matrices
    eigs, Q = torch.linalg.eigh(Tm)
    # Compute function of eigenvalues
    feigs = f(eigs)
    # Get first row of eigenvector matrix
    v1 = Q[0, :]
    # Compute quadratic form v1^T @ diag(f(eigs)) @ v1
    approx = (v1 * feigs).dot(v1).sum()
    # Scale by squared norm of z
    approx *= z.dot(z)
    return approx


def slq(Afun: Callable, N: int, m: int):
    """
    Stochastic Lanczos Quadrature algorithm.
    :param Afun: matrix-matrix multiplication oracle X |-> AX
    :param N: number of probe vectors
    :param m: number of Lanczos iterations
    :return:
    """
    pass


if __name__ == '__main__':
    from src.lanczos import lanczos_linear_system
    from scipy.stats import ortho_group
    torch.set_default_dtype(torch.double)

    n, m = 100, 10
    # Create random symmetric matrix
    eigs = 10**torch.linspace(1, 5, n)
    f = torch.log

    Q = torch.from_numpy(ortho_group.rvs(n))
    A = Q @ torch.diag(eigs) @ Q.T
    fA = Q @ torch.diag(f(eigs)) @ Q.T
    # Lanczos
    z = torch.randn(n)
    x0 = torch.zeros(n)
    _, _, Tm = lanczos_linear_system(lambda x: A@x, x0, z, m)

    # mBCG
    Xm, Ts = mbcg(lambda X: A@X, lambda X: X, z.reshape(-1, 1), x0.reshape(-1, 1), m)
    Tm = Ts[0]

    # Quadratic form estimation
    quad = lanczos_quadrature(f, Tm, z)
    quad_true = z.T @ fA @ z
    err = torch.abs(quad - quad_true)
    relerr = torch.abs( (quad - quad_true) / quad_true )
