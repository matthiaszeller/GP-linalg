

import unittest

import torch

from src.kernel import SquaredExponentialKernel, MaternKernel
from sklearn.gaussian_process.kernels import Matern as SklearnMatern

torch.set_default_dtype(torch.double)


class TestKernel(unittest.TestCase):

    def test_distance_matrix(self):
        n = 100
        x = torch.randn(n)
        k = SquaredExponentialKernel(x)
        D = k.D

        Dtrue = torch.empty(n, n)
        for i in range(n):
            for j in range(n):
                Dtrue[i, j] = (x[i] - x[j]).norm()

        torch.testing.assert_allclose(D, Dtrue, atol=1e-10, rtol=1e-9)

    def test_squared_exponential_kernel(self):
        n = 100
        x = torch.randn(n)
        l = 2.0
        K = SquaredExponentialKernel(x).compute_kernel(torch.tensor([0.1, l]))

        # Compute squared distance matrix
        D2 = torch.empty(n, n)
        for i in range(n):
            for j in range(n):
                dist2 = ((x[i] - x[j])**2).sum()
                D2[i, j] = dist2

        # Compute actual kernel
        Ktrue = torch.exp(-D2 / l)

        torch.testing.assert_allclose(K, Ktrue, atol=1e-14, rtol=1e-14)

    def test_matern_kernel(self):
        n = 100
        x = torch.randn(n).reshape(-1, 1)
        ls = (0.5, 1.0, 2.0)
        nus = (0.5, 1.0, 1.5, 2.5)
        k = MaternKernel(x)
        for l in ls:
            for nu in nus:
                k_sklearn = SklearnMatern(length_scale=l, nu=nu)

                K = k.compute_kernel(torch.tensor([0.1, l, nu]))
                K_sk = k_sklearn(x)
                torch.testing.assert_allclose(K, torch.from_numpy(K_sk))


if __name__ == '__main__':
    unittest.main()
