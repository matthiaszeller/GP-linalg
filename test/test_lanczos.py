

import unittest
import torch
from scipy.stats import ortho_group

from src.lanczos import lanczos_linear_system

torch.set_default_dtype(torch.double)


class TestLanczos(unittest.TestCase):

    def test_lanczos_orthonormality(self):
        n, m = 100, 30
        for _ in range(10):
            M = torch.randn(n, n)
            A = M + M.T
            b, x0 = torch.randn(n), torch.zeros(n)
            _, V, _ = lanczos_linear_system(lambda x: A@x, x0, b, m)
            torch.testing.assert_allclose(V.T @ V, torch.eye(m), rtol=0, atol=1e-10)

    def test_lanczos_convergence_spd(self):
        n = 100
        ms = [2, 10, 20, 50]
        Q = torch.from_numpy(ortho_group.rvs(n))
        eigs = torch.linspace(1, 1000, n)
        A = Q @ torch.diag(eigs) @ Q.T
        Ainv = Q @ torch.diag(1 / eigs) @ Q.T
        condA = eigs.max() / eigs.min()

        for _ in range(10):
            b = torch.randn(n)
            x0 = torch.zeros(n)
            xtrue = Ainv @ b
            for m in ms:
                upperbound = 2 * ((condA ** 0.5 - 1) / (condA ** 0.5 + 1)) ** m
                xm, V, T = lanczos_linear_system(lambda x: A @ x, x0, b, m)
                # Convergence: check upperbound of erros with A-norms
                diff = xtrue - xm
                relerr_A = (diff.T @ A @ diff) ** 0.5 / (xtrue.T @ A @ xtrue) ** 0.5
                self.assertGreater(upperbound, relerr_A)


if __name__ == '__main__':
    unittest.main()
