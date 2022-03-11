"""
Conjugate Gradient Method.
"""

from typing import Callable

from src.utils import Array


def cg_vanilla(Afun: Callable, b: Array, x0: Array, k: int, callback: Callable = None):
    """
    Vanilla Conjugate Gradient.

    :param Afun: matrix-vector multiplication oracle x |-> Ax, with A positive semi-definite matrix
    :param b: vector such that Ax=b
    :param x0: starting vector
    :param k: number of iterations
    :param callback: function called at the end of each iteration, provided with current solution as input
    :return: x, residuals
             approximate solution after k steps, (k+1)-dim array of norm of residuals
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


def pcg_vanilla(Afun: Callable, Pinv: Callable, b: Array, x0: Array, k: int, callback: Callable = None):
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
