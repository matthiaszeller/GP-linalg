
import unittest

import numpy as np
import torch
from scipy.stats import ortho_group

from src.precond import pinv_fun, precond_logdet, PartialCholesky

torch.set_default_dtype(torch.double)


class TestPrecondRoutines(unittest.TestCase):

    def test_pinv_fun_simple(self):
        Lk = torch.diag(torch.tensor([3., 2., 1.]))[:, :2]
        sigma2 = 1.0
        Pk = Lk @ Lk.T + sigma2 * np.eye(3)
        Pkinv = torch.diag(1 / torch.diag(Pk))
        fun = pinv_fun(Lk, sigma2)

        for _ in range(10):
            y = torch.randn(3)
            x = fun(y)
            xtrue = Pkinv @ y
            torch.testing.assert_allclose(x, xtrue)

    def test_pinv_fun(self):
        n, k = 10, 3
        sigma2 = 2.
        for _ in range(10):
            Lk = torch.randn(n, k)
            P = Lk @ Lk.T + torch.eye(n) * sigma2
            y = torch.randn(n)
            xtrue = torch.linalg.solve(P, y)
            x = pinv_fun(Lk, sigma2)(y)
            torch.testing.assert_allclose(x, xtrue, atol=1e-15, rtol=1e-5)

    def test_precond_logdet(self):
        n, k = 10, 3
        sigma2 = 1.0
        for _ in range(10):
            Lk = torch.randn(n, k)
            Phat = Lk @ Lk.T + torch.eye(n) * sigma2
            logdet_torch = torch.linalg.slogdet(Phat).logabsdet
            logdet = precond_logdet(Lk, sigma2)
            torch.testing.assert_close(logdet, logdet_torch, atol=1e-14, rtol=1e-5)


class TestPreconditionner(unittest.TestCase):
    n = 100
    Q = torch.from_numpy(ortho_group.rvs(n))
    eigs = 10**torch.linspace(0, 5, n)
    A = Q @ torch.diag(eigs) @ Q.T

    def test_matmul(self):
        k, sigma2 = 10, 2
        P = PartialCholesky(self.A, k, sigma2)
        Pkhat = P.Lk @ P.Lk.T + sigma2 * torch.eye(self.n)
        X = torch.randn(self.n, 10)
        mul = P @ X
        true = Pkhat @ X
        torch.testing.assert_allclose(mul, true, atol=0, rtol=1e-10)

    def test_invfun(self):
        k, sigma2 = 30, 2.
        P = PartialCholesky(self.A, k, sigma2)
        Phat = P.Lk @ P.Lk.T + sigma2 * torch.eye(self.n)
        for _ in range(10):
            y = torch.randn(self.n)
            x = P.inv_fun(y)
            xtrue = torch.linalg.solve(Phat, y)
            torch.testing.assert_allclose(x, xtrue, rtol=1e-8, atol=1e-10)

    def test_invfun_batch(self):
        k, sigma2 = 30, 2.
        P = PartialCholesky(self.A, k, sigma2)
        Phat = P.Lk @ P.Lk.T + sigma2 * torch.eye(self.n)

        Y = torch.randn(self.n, 10)
        X = P.inv_fun(Y)
        Xtrue = torch.linalg.solve(Phat, Y)
        torch.testing.assert_allclose(X, Xtrue)

    def test_logdet(self):
        sigma2 = 2.0
        for k in (3, 5, 10, 20, 50):
            P = PartialCholesky(self.A, k, sigma2)
            Phat = P.Lk @ P.Lk.T + sigma2 * torch.eye(self.n)
            logdet = P.logdet()
            true = torch.logdet(Phat).item()
            self.assertAlmostEqual(logdet, true, places=10)


if __name__ == '__main__':
    unittest.main()
