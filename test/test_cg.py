

import unittest

import numpy as np
import torch
from scipy import stats

from src.cg import cg_vanilla, pcg_vanilla, mbcg

torch.set_default_dtype(torch.float64)


class TestCG(unittest.TestCase):
    n = 100
    Q = stats.ortho_group.rvs(n)
    eigs = np.arange(1, n+1).astype(float)
    A = Q @ np.diag(eigs) @ Q.T
    Ainv = Q @ np.diag(1 / eigs) @ Q.T

    t = 10 # for batched operations

    def test_vanilla_cg(self):
        b = np.random.randn(self.n)
        b /= b.dot(b)**0.5
        m = 90
        xk = cg_vanilla(lambda x: self.A@x, b, np.zeros(self.n), m)
        np.testing.assert_allclose(xk, self.Ainv@b, rtol=1e-10, atol=1e-16)

    def test_vanilla_pcg(self):
        b = np.random.randn(self.n)
        b /= b.dot(b)**0.5
        m = 20
        eigs_pinv = np.concatenate((np.ones(self.n-m), 1/self.eigs[-m:]))
        Pinv = self.Q @ np.diag(eigs_pinv) @ self.Q.T
        k = 75
        xk = pcg_vanilla(lambda x: self.A@x, lambda x: Pinv@x, b, np.zeros(self.n), k)
        np.testing.assert_allclose(xk, self.Ainv@b, rtol=1e-10, atol=1e-16)

    def test_mbcg_solution(self):
        B = np.random.randn(self.n, self.t)
        #B /= (B * B).sum(0)
        k = 80
        Xk, _ = mbcg(lambda X: self.A@X, lambda X: X, B, np.zeros((self.n, self.t)), k)
        np.testing.assert_allclose(Xk, self.Ainv@B, rtol=1e-10, atol=1e-16)
        # Test full rank
        Xk, _ = mbcg(lambda X: self.A@X, lambda X: X, B, np.zeros((self.n, self.t)), self.n)
        np.testing.assert_allclose(Xk, self.Ainv@B, rtol=1e-10, atol=1e-16)

        # Test with vectors (non-batched version)
        b = np.random.randn(self.n)
        xk, _ = mbcg(lambda X: self.A@X, lambda X: X, b, np.zeros(self.n), self.n)
        np.testing.assert_allclose(xk, self.Ainv@b, rtol=1e-10, atol=1e-16)

    def test_mbcg_2(self):
        B = np.random.randn(self.n, self.t)
        #B /= (B * B).sum(0)
        k = 80
        Xk, _ = mbcg(lambda X: self.A @ X, lambda X: X, B, np.zeros((self.n, self.t)), k)
        for i in range(self.t):
            xtrue = self.Ainv @ B[:, i]
            xk = Xk[:, i]
            np.testing.assert_allclose(xk, xtrue, rtol=1e-10, atol=1e-16)

    def test_precond_mbcg_solution(self):
        B = np.random.randn(self.n, self.t)
        m = 50
        eigs_pinv = np.concatenate((np.ones(self.n - m), 1 / self.eigs[-m:]))
        Pinv = self.Q @ np.diag(eigs_pinv) @ self.Q.T

        k = 50
        Xk, _ = mbcg(lambda X: self.A@X, lambda X: Pinv @ X, B, np.zeros((self.n, self.t)), k)
        np.testing.assert_allclose(Xk, self.Ainv @ B, rtol=1e-10, atol=1e-16)

    def test_mbcg_T(self):
        t = 10
        B = np.random.randn(self.n, t)
        B /= (B * B).sum(0)
        _, Ts = mbcg(lambda X: self.A @ X, lambda X: X, B, np.zeros((self.n, t)), 50)
        for T in Ts:
            np.testing.assert_allclose(T, T.T, atol=0)
            np.testing.assert_allclose(np.tril(T, -2), 0.0, atol=0)

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
