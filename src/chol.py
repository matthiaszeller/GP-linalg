import torch
from math import sqrt

import numpy
import numpy as np


def find_pivot(S, i):
    return np.argmax(np.diag(S)[i:]) + i


def pivoted_chol(A: torch.Tensor, k: int):
    # TODO if lazy tensors use get_diag and get_row instead
    # https://www.sciencedirect.com/science/article/pii/S0168927411001814
    d = torch.diag(A).clone()
    n = A.shape[0]
    pi = torch.arange(n)
    R = torch.zeros(k, n, dtype=A.dtype, device=A.device)
    for m in range(k):
        pivot = torch.argmax(d[pi[m:]]).item() + m
        pi[[m, pivot]] = pi[[pivot, m]]

        if d[pi[m]] < 0:
            raise ValueError('the matrix is not PSD')
        R[m, pi[m]] = d[pi[m]] ** 0.5

        # TODO handle psd case L[m, pi[m]] is zero
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

    L = R.T
    return L


# def pivoted_chol(A: np.ndarray, k: int):
#     # After step i, the matrix S[:, :k] represents the i-step partial pivoted Cholesky decomposition of A
#     #               the matrix S[k+1:, k+1:] represents the Schur complement
#     S = A.copy()
#     ls = []
#     pi = np.arange(A.shape[0])
#     for i in range(k):
#         # Find permutation col/row
#         p = find_pivot(S, i)
#         # Permute 1st row and col with pth row and col
#         S[[i, p], [i, p]] = S[[p, i], [p, i]]
#         pi[[i, p]] = pi[[p, i]]
#         # Cholesky decomposition of Schur complement S
#         #l = S[:, 0] / sqrt(S[0, 0])
#         S[i:, i] /= sqrt(S[i, i])
#         # Compute new Schur complement (of smaller size)
#         S[i+1:, i+1:] -= np.outer(S[i+1:, i], S[i+1:, i])
#
#     S = np.tril(S)
#     return S
#
#     # Build the matrix L
#     L = numpy.zeros(A.shape)
#     # Accumulate permutations
#     pi = np.arange(A.shape[0])
#     for i in range(k):
#         # Permutation performed at step i was only considering diagonal of Schur complement
#         # thus must shift the permutation index
#         # p_tilde = pivots[i] + i
#         # # Update the accumulative permutation vector
#         # print(pi)
#         # pi[[i, p_tilde]] = pi[[p_tilde, i]]
#         # print(pi)
#         # Take into account the zeros when doing recursion
#         l = ls[i]
#         l[[0, pivots[i]]] = l[[pivots[i], 0]]
#         pi[[i, pivots[i]]] = pi[[pivots[i], i]]
#         ltilde = np.concatenate((np.zeros(i), l))
#         # Apply accumulative permutation vector
#         #ltilde = ltilde[pi]
#         L += np.outer(ltilde, ltilde)
#
#     return L


# %%

if __name__ == '__main__':
    import tensorflow_probability as tfp
    import tensorflow as tf

    torch.set_default_dtype(torch.double)

    # Full rank
    # n, k = 5, 5
    # A = torch.randn(n, n)
    # A = A @ A.T
    # L = pivoted_chol(A, k)
    # err_numpy = (A-L@L.T).max()

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
#     A = np.random.randn(n, n)
#     A = A @ A.T
#     L = pivoted_chol(A, k)
#
#     err = np.abs(A - L @ L.T).max()
#
#     # Partial
#     n, k = 10, 5
#     A = np.random.randn(n, n)
#     A = A @ A.T
#     L = pivoted_chol(A, k)
#     condA = np.linalg.cond(A)
#     P = L @ L.T
#
#     sigma = 1.0
#     Ahat = A + np.eye(n) * sigma
#     Phat = P + np.eye(n) * sigma
#     condPinvA = np.linalg.cond(np.linalg.inv(Phat) @ Ahat) # TODO better test
# # %%
