

import unittest

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from scipy import stats

from src.cg import cg_vanilla, pcg_vanilla, mbcg


class TestCG(unittest.TestCase):
    n = 100
    Q = stats.ortho_group.rvs(n)
    eigs = np.arange(1, n+1).astype(float)
    A = Q @ np.diag(eigs) @ Q.T
    Ainv = Q @ np.diag(1 / eigs) @ Q.T

    def test_vanilla_cg(self):
        b = np.random.randn(self.n)
        b /= b.dot(b)**0.5
        m = 50
        xk = cg_vanilla(lambda x: self.A@x, b, np.zeros(self.n), m)
        np.testing.assert_almost_equal(xk, self.Ainv@b, decimal=8)

    def test_vanilla_pcg(self):
        b = np.random.randn(self.n)
        b /= b.dot(b)**0.5
        m = 30
        eigs_pinv = np.concatenate((np.ones(self.n-m), 1/self.eigs[-m:]))
        Pinv = self.Q @ np.diag(eigs_pinv) @ self.Q.T
        xk = pcg_vanilla(lambda x: self.A@x, lambda x: Pinv@x, b, np.zeros(self.n), 50)
        np.testing.assert_almost_equal(xk, self.Ainv@b, decimal=8)

    def test_mbcg(self):
        t = 10
        B = np.random.randn(self.n, t)
        B /= (B * B).sum(0)
        Xk, _ = mbcg(lambda X: self.A@X, lambda X: X, B, np.zeros((self.n, t)), 50)
        np.testing.assert_almost_equal(Xk, self.Ainv@B)

        # TODO test T returned by mbcg

    def test_torch(self):
        A = torch.from_numpy(self.A)
        Ainv = torch.from_numpy(self.Ainv)
        b = torch.randn(self.n)
        b /= b.dot(b)**0.5

        m = 50
        xk = cg_vanilla(lambda x: A @ x, b, torch.zeros(self.n), m)
        torch.testing.assert_allclose(xk, Ainv@b)
        xk = pcg_vanilla(lambda x: A @ x, lambda x: x, b, torch.zeros(self.n), m)
        torch.testing.assert_allclose(xk, Ainv@b)

        t = 10
        B = torch.randn(self.n, t)
        B /= (B * B).sum(0)
        Xk, _ = mbcg(lambda X: A @ X, lambda X: X, B, torch.zeros((self.n, t)), m)
        torch.testing.assert_allclose(Xk, Ainv@B)


if __name__ == '__main__':
    unittest.main()
