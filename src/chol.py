import logging
from typing import Callable

import torch


def pivoted_chol(A: torch.Tensor, k: int, tol: float = 1e-12, callback: Callable = None) -> torch.Tensor:
    """
    Partial pivoted Cholesky factorization, see https://dl.acm.org/doi/10.1016/j.apnum.2011.10.001.
    Returns the partial Cholesky factor Lk such that Lk@Lk.R ~= A.
    Note that Lk is *not* lower triangular.

    :param A: nxn PSD matrix
    :param k: rank of approximation, must be <= rank(A)
    :param tol: tolerance on trace norm of the error matrix E = A - L L^T
    :param callback: function called at each iteration with input being the error at current step
    :return: Lm, nxm matrix, the "Cholesky factor" where m <= k, equality m=k depends on convergence w.r.t. tolerance
    """
    # TODO if lazy tensors use get_diag and get_row instead
    # Initialize diagonal on which we'll search pivots
    d = torch.diag(A).clone()
    n = A.shape[0]

    # Check necessary condition for positive semi-definiteness: all diagonal elements must be >= 0
    # easy to show by considering quadratic forms with canonical basis vectors: e_i^T A e_i = A_ii >= 0 by def of SPD
    if not (d >= 0).all():
        raise ValueError('the matrix is not SPD')

    # Initialize error (trace-norm)
    err = d.sum()

    # Initialize permutation vector
    pi = torch.arange(n)
    # Initialize partial Cholesky factor
    R = torch.zeros(k, n, dtype=A.dtype, device=A.device)

    # Loop while not reached tolerance or until max number of iter was reached
    m = 0
    while (m < k) and (err > tol):
        # Find pivot among remaining (permuted) diagonal elements
        pivot = torch.argmax(d[pi[m:]]).item() + m
        # Swap with pivot
        pi[[m, pivot]] = pi[[pivot, m]]

        # Check if negative definite
        if d[pi[m]] < 0:
            raise ValueError(f'the matrix is not PSD, pivot is {d[pi[m]]}')

        # Cholesky factorization of Schur complement
        R[m, pi[m]] = d[pi[m]] ** 0.5
        # TODO handle: if A PSD *and* k > rank of A, then R[m, pi[m]] zero and division by zero
        row = A[pi[m], :]
        if m > 0:
            for i in range(m+1, n):
                dot = R[:m, pi[m]].dot(R[:m, pi[i]]) # if m >= 1
                R[m, pi[i]] = (row[pi[i]] - dot) / R[m, pi[m]]
                d[pi[i]] -= R[m, pi[i]]**2
        else:
            for i in range(m+1, n):
                R[m, pi[i]] = row[pi[i]] / R[m, pi[m]]
                d[pi[i]] -= R[m, pi[i]]**2

        # Compute error
        err = d[pi[m+1:]].sum()

        if callback is not None:
            callback(err)

        # Prepare next iteration
        m += 1

    # Notify user
    if err < tol and m < k:
        print(f'pivoted cholesky converged after {m} steps')

    # Return "lower triangular" instead of "upper triangular"
    L = R.T
    # Also, truncate the matrix in case convergence was reached because max number of iterations
    L = L[:, :m]
    return L


if __name__ == '__main__':
    import tensorflow_probability as tfp
    import tensorflow as tf
    from scipy.stats import ortho_group

    torch.set_default_dtype(torch.double)

    # Full rank
    n, k = 5, 5
    A = torch.randn(n, n)
    A = A @ A.T
    L = pivoted_chol(A, k)
    err_full = (A-L@L.T).max()

    # Partial compare with tensorflow implementation
    n, k = 10, 8
    A = torch.randn(n, n)
    A = A @ A.T
    Ltfp = tfp.math.pivoted_cholesky(tf.convert_to_tensor(A), k)
    Ltfp = torch.from_numpy(Ltfp.numpy())
    Lk = pivoted_chol(A, k)
    err = torch.abs(Lk - Ltfp).max()

    sigma2 = 1.0
    Ahat = A + sigma2 * torch.eye(n)
    Phat = Lk @ Lk.T + sigma2 * torch.eye(n)
    condAhat = torch.linalg.cond(Ahat)
    precond_mx = torch.linalg.solve(Phat, Ahat)
    condPinvAhat = torch.linalg.cond(precond_mx)

    # Compute partial cholesky for rank-deficient PSD matrix
    n, r = 20, 5
    Q = torch.from_numpy(ortho_group.rvs(n))
    eigs = torch.linspace(1, n, n)
    eigs[r:] = 0
    A = Q @ torch.diag(eigs) @ Q.T

    L = pivoted_chol(A, n)
