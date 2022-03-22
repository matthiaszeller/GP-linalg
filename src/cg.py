"""
Conjugate Gradient Method.
"""

from typing import Callable, Tuple, List

import torch

from src import utils
from src.lanczos import lanczos_linear_system
from src.utils import Array


def cg_vanilla(Afun: Callable, b: Array, x0: Array, k: int, callback: Callable = None) -> Array:
    """
    Vanilla Conjugate Gradient. Both torch- and numpy-compatible.

    :param Afun: matrix-vector multiplication oracle x |-> Ax, with A positive semi-definite matrix
    :param b: vector such that Ax=b
    :param x0: starting vector
    :param k: number of iterations
    :param callback: function called at the end of each iteration, provided with current solution as input
    :return: x, approximate solution after k steps
    """
    # Residuals of starting point
    r = b - Afun(x0)
    # Initialize search direction
    d = r
    # Initialize loop
    xprev = x0
    # Precompute squared norm of residuals
    rnorm2 = [-1] * (k+1)
    rnorm2[0] = r.dot(r)
    for i in range(1, k+1):
        # Precompute matrix-vector product A x d
        Ad = Afun(d)
        # Solution of the line search: optimal step size in direction d
        alpha = rnorm2[i-1] / d.dot(Ad)
        # Update solution approximation: linear combination of previous approx and step in direction d
        x = xprev + alpha * d
        # Update the residuals of the new approximation
        r = r - alpha * Ad
        rnorm2[i] = r.dot(r)
        # Conjugate Gram Schmidt: coefficient multiplying last search direction such that when it is added to the new
        #                         residuals, the new search direction is A-orthogonal to previous directions
        beta = rnorm2[i] / rnorm2[i-1]
        # Compute next search direction with Conjugate Gram Schmidt
        d = r + beta * d
        # Prepare next loop iteration
        xprev = x
        if callback is not None:
            callback(x)

    return x


def pcg_vanilla(Afun: Callable, Pinv: Callable, b: Array, x0: Array, k: int, callback: Callable = None) -> Array:
    """
    Preconditionned Conjugate Gradient Algorithm. Both torch- and numpy-compatible.

    :param Afun: matrix-vector multiplication oracle x |-> Ax, with A positive semi-definite matrix
    :param Pinv: matrix-vector multiplication oracle x |-> P^{-1} x
    :param b: vector such that Ax=b
    :param x0: starting vector
    :param k: number of iterations
    :param callback: function called at the end of each iteration, provided with current solution as input
    :return: x, approximate solution after k steps
    """
    # Residuals of starting point
    r = b - Afun(x0)
    # Preconditioned residuals
    z = Pinv(r)
    # Initialize search direction
    d = z
    # Initialize loop
    xprev = x0
    # Precompute residual dot product with preconditioned residuals
    rTz_prev = r.dot(z)
    for i in range(1, k+1):
        # Precompute matrix-vector product A x d
        Ad = Afun(d)
        # Solution of the line search: optimal step size in direction d
        alpha = rTz_prev / d.dot(Ad)
        # Update solution approximation: linear combination of previous approx and step in direction d
        x = xprev + alpha * d
        # Update the residuals of the new approximation
        r = r - alpha * Ad
        z = Pinv(r)
        rTz = r.dot(z)
        # Conjugate Gram Schmidt: coefficient multiplying last search direction such that when it is added to the new
        #                         residuals, the new search direction is A-orthogonal to previous directions
        beta = rTz / rTz_prev
        # Compute next search direction with Conjugate Gram Schmidt
        d = z + beta * d
        # Prepare next loop iteration
        xprev = x
        rTz_prev = rTz
        if callback is not None:
            callback(x)

    return x


def mbcg(Afun: Callable, Pinv: Callable, B: Array, X0: Array, k: int,
         callback: Callable = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if X0.ndim == 1:
        X0 = X0.reshape(-1, 1)

    # Residuals of starting point
    R = B - Afun(X0)
    # Preconditioned residuals
    Z = Pinv(R)
    # Initialize search direction
    D = Z
    # Initialize loop
    Xprev = X0
    # Precompute residual dot product with preconditioned residuals
    RTZ_prev = (R * Z).sum(0)
    # Initialize alpha_0, beta_0 used to later build Hessenberg matrices
    # Those are ghost values to ease implementation since T_11 = 1/alpha1 = 1/alpha1 + beta0/alpha0, beta0=0, alpha0!=0
    t = B.shape[-1]
    alphas, betas = [[1.] * t], [[0.] * t]
    for i in range(1, k+1):
        # Precompute matrix-matrix product A x D
        AD = Afun(D)
        # Solution of the line search: optimal step size in direction d
        alpha = RTZ_prev / (D * AD).sum(0)
        alphas.append(alpha)
        # Update solution approximation: linear combination of previous approx and step in direction d
        # By both Torch's and numpy's broadcasting rules, alpha being a 1D tensor will multiply D column-wise
        X = Xprev + alpha * D
        # Update the residuals of the new approximation
        R = R - alpha * AD
        Z = Pinv(R)
        RTZ = (R * Z).sum(0)
        # Conjugate Gram Schmidt: coefficient multiplying last search direction such that when it is added to the new
        #                         residuals, the new search direction is A-orthogonal to previous directions
        beta = RTZ / RTZ_prev
        betas.append(beta)
        # Compute next search direction with Conjugate Gram Schmidt
        D = Z + beta * D
        # Prepare next loop iteration
        Xprev = X
        RTZ_prev = RTZ
        if callback is not None:
            callback(X)

    # Recover Lanczos Hessenberg matrix from CG coefficients alpha, beta
    # t = number of simultaneous equations, i.e. size of 2nd dimension of B, X0
    # First build matrix Alphas of size k x t containing (1/alpha) coefficients
    Alphas_inv = 1 / utils.build_array_like(alphas, B)
    # Then the beta coefficients
    Betas = utils.build_array_like(betas, B)
    # Build the Hessenberg matrices, each column of Alphas, Betas correspond to coeffs in one Hessenberg matrix
    # Thus, we iterate over the transposed matrices
    Ts = []
    for ainv, b in zip(Alphas_inv.T, Betas.T):
        # Make use of the ghost values for the relations:
        # T_ii = 1/alpha_i + beta_{i-1}/alpha_{i-1}
        diag = ainv[1:] + b[:-1] * ainv[:-1]
        offdiag = b[1:-1]**0.5 * ainv[1:-1]
        Ts.append(utils.build_sym_tridiag_matrix(diag, offdiag))

    return X, Ts

#%%


if __name__ == '__main__':
    import numpy as np
    # Compare T matrix got by MBCG and Lanczos
    n = 100
    M = np.random.randn(n, n)
    A = M @ M.T
    b = np.random.randn(n)
    m = 30
    x0 = np.zeros(n)
    _, _, T_lanczos = lanczos_linear_system(lambda x: A@x, x0, b, m)

    b = b.reshape(-1, 1)
    x0 = x0.reshape(-1, 1)
    _, T_mbcg = mbcg(lambda x: A@x, lambda x: x, b, x0, m)

    err_T = np.abs(T_lanczos - T_mbcg).max()
