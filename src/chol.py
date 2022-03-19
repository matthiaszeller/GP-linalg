

import torch


def pivoted_chol(A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Partial pivoted Cholesky factorization, see https://dl.acm.org/doi/10.1016/j.apnum.2011.10.001.
    Returns the partial Cholesky factor Lk such that Lk@Lk.R ~= A.
    Note that Lk is *not* lower triangular, but it is lower triangular up to some permutation of rows.

    :param A: nxn PSD matrix
    :param k: rank of approximation, must be <= rank(A)
    :return: Lk, nxk matrix
    """
    # TODO if lazy tensors use get_diag and get_row instead
    # Initialize diagonal on which we'll search pivots
    d = torch.diag(A).clone()
    n = A.shape[0]
    # Initialize permutation vector
    pi = torch.arange(n)
    # Initialize partial Cholesky factor
    R = torch.zeros(k, n, dtype=A.dtype, device=A.device)
    for m in range(k):
        # Find pivot among remaining (permuted) diagonal elements
        pivot = torch.argmax(d[pi[m:]]).item() + m
        pi[[m, pivot]] = pi[[pivot, m]]

        # Check if negative definite
        if d[pi[m]] < 0:
            raise ValueError('the matrix is not PSD')

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

    # Return "lower triangular" instead of "upper triangular"
    L = R.T
    return L


if __name__ == '__main__':
    import tensorflow_probability as tfp
    import tensorflow as tf

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

