
import unittest
import torch
import tensorflow as tf
import tensorflow_probability as tfp

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

    def test_condition_number(self):
        # TODO this isn't right ?
        n, k = 10, 3
        sigma2 = 1.0
        for _ in range(10):
            M = torch.randn(n, n)
            A = M @ M.T
            L = pivoted_chol(A, k)
            #
            Ahat = A + torch.eye(n) * sigma2
            Phat = L @ L.T + torch.eye(n) * sigma2
            PinvA = torch.linalg.solve(Phat, Ahat)
            condA = torch.linalg.cond(Ahat)
            condPinvA = torch.linalg.cond(PinvA)
            self.assertGreater(condA, condPinvA)


if __name__ == '__main__':
    unittest.main()
