
import unittest
import torch
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import ortho_group

from src.chol import pivoted_chol

torch.set_default_dtype(torch.double)


class TestCholesky(unittest.TestCase):

    def test_full_rank(self):
        n = 10
        for _ in range(10):
            M = torch.randn(n, n)
            A = M @ M.T
            L = pivoted_chol(A, n)
            torch.testing.assert_allclose(A, L @ L.T)

    def test_full_rank_gpu(self):
        n = 10
        for _ in range(10):
            M = torch.randn(n, n, device='cuda')
            A = M @ M.T
            L = pivoted_chol(A, n)
            torch.testing.assert_allclose(A, L @ L.T)

    def test_partial_vs_tensorflow(self):
        n, k = 20, 10
        for _ in range(10):
            M = torch.randn(n, n)
            A = M @ M.T
            L = pivoted_chol(A, k)
            Ltf = tfp.math.pivoted_cholesky(tf.convert_to_tensor(A), k)
            Ltf = torch.from_numpy(Ltf.numpy())
            torch.testing.assert_allclose(L, Ltf)

    def test_low_rank(self):
        n, r = 20, 2
        Q = torch.from_numpy(ortho_group.rvs(n))
        eigs = torch.linspace(1, n, n)
        eigs[r:] = 0
        A = Q @ torch.diag(eigs) @ Q.T
        L = pivoted_chol(A, n)
        self.assertEqual(L.shape, (n, r))

        torch.testing.assert_allclose(A, L @ L.T)

    def test_condition_number(self):
        n, k = 50, 30
        sigma2 = 1.0
        for _ in range(10):
            M = torch.randn(n, n)
            A = M @ M.T
            L = pivoted_chol(A, k)
            #
            Ahat = A + torch.eye(n) * sigma2
            Phat = L @ L.T + torch.eye(n) * sigma2
            PinvA = torch.linalg.solve(Phat, Ahat)
            eigs_PinvA = torch.linalg.eigvals(PinvA)
            torch.testing.assert_allclose(torch.imag(eigs_PinvA), torch.zeros(n))
            eigs_PinvA = torch.real(eigs_PinvA)

            condA = torch.linalg.cond(Ahat)
            condPinvA = eigs_PinvA.max() / eigs_PinvA.min()
            self.assertGreater(condA, condPinvA)

    def test_callback(self):
        n = 10
        M = torch.randn(n, n)
        A = M @ M.T + torch.eye(n) # add digonal to avoid early convergence
        lst = []
        P = pivoted_chol(A, 5, callback=lambda err_k: lst.append(err_k))
        self.assertEqual(len(lst), 5)


if __name__ == '__main__':
    unittest.main()
