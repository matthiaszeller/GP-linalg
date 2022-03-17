

from typing import Callable, Tuple

import torch

from src import utils
from src.utils import Array


def lanczos_linear_system(Afun: Callable, x0: Array, b: Array, m: int) -> Tuple[Array, Array, Array]:
    """
    Lanczos solver for linear systems Ax=b with A symmetric.
    This function is actually indented to be used to recover the T and V matrices from the Arnoldi decomposition:
        A V_m = V_m T_m + beta_m+1 * v_m+1 e^T

    :param Afun: matrix-vector multiplication oracle x |-> Ax with A symmetric
    :param x0: starting vector
    :param b: vector such that Ax=b
    :param m: number of steps
    :return: xm, Vm, Tm
    """
    # Initial residuals
    r = b - Afun(x0)
    # Initialize basis of Krylov subspace
    r0norm = r.dot(r)**0.5
    v = r/r0norm
    V = [v]

    beta = 0.
    v_prev = 0. # will later be vector but needed for first iteration
    betas, alphas = [], []
    for _ in range(m):
        w = Afun(v) - beta * v_prev
        alpha = w.dot(v)
        wtilde = w - alpha * v
        beta = wtilde.dot(wtilde)**0.5
        v_prev = v
        v = wtilde / beta
        V.append(v)
        alphas.append(alpha)
        betas.append(beta)

    V = torch.stack(V[:-1]).T
    T = utils.build_sym_tridiag_matrix(torch.tensor(alphas), torch.tensor(betas[:-1]))
    e1 = torch.zeros(m)
    e1[0] = 1.
    ym = torch.linalg.solve(T, r0norm * e1)
    xm = x0 + V @ ym

    return xm, V, T


if __name__ == '__main__':
    from scipy.stats import ortho_group
    torch.set_default_dtype(torch.double)

    n, m = 100, 30
    Q = torch.from_numpy(ortho_group.rvs(n))
    eigs = torch.linspace(1, 1000, n)
    A = Q @ torch.diag(eigs) @ Q.T
    Ainv = Q @ torch.diag(1/eigs) @ Q.T

    b = torch.randn(n)
    x0 = torch.zeros(n)
    xtrue = Ainv @ b

    xm, V, T = lanczos_linear_system(lambda x: A@x, x0, b, m)
    # Check solutions
    relerr = torch.norm(xtrue - xm) / torch.norm(xtrue)
    # Check orthonormality
    errorV = torch.abs(torch.eye(m) - V.T @ V).max()

    condA = eigs.max() / eigs.min()
    # Check convergence of relative error (with A-norms)
    upperbound = 2 * ( (condA**0.5 - 1) / (condA**0.5 + 1) )**m
    diff = xtrue - xm
    relerr_A = (diff.T @ A @ diff)**0.5 / (xtrue.T @ A @ xtrue) ** 0.5

    # n = 100
    # eigs = np.arange(1, n+1) * 10
    # Q = ortho_group.rvs(n)
    # A = Q @ np.diag(eigs) @ Q.T
    # Ainv = Q @ np.diag(1/eigs) @ Q.T
    # b = np.random.randn(n)
    # xtrue = Ainv @ b
    # x0 = np.zeros(n)
    # m = 30
    # xm, V, T = lanczos_linear_system(lambda x: A@x, x0, b, m)
    # relerr_x = np.linalg.norm(xtrue - xm) / np.linalg.norm(xtrue)
    # # How far is V from having orthonormal columns ?
    # error_V = np.abs(np.eye(m) - V.T @ V).max()

