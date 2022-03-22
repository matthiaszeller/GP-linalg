

import unittest

import torch

from src.cg import mbcg
from src.lanczos import lanczos_linear_system

torch.set_default_dtype(torch.double)


class TestLanczosFromCg(unittest.TestCase):

    def test_lanczos_from_cg_single(self):
        n, m = 100, 30
        for _ in range(10):
            M = torch.randn(n, n)
            A = M @ M.T
            b, x0 = torch.randn(n), torch.randn(n)

            _, _, T_lanczos = lanczos_linear_system(lambda x: A@x, x0, b, m)

            b = b.reshape(-1, 1)
            x0 = x0.reshape(-1, 1)
            _, (T_mbcg, ) = mbcg(lambda X: A@X, lambda X: X, b, x0, m)

            torch.testing.assert_allclose(T_mbcg, T_lanczos, atol=1e-10, rtol=1e-8)

    def test_lanczos_from_cg_batched(self):
        n, m = 100, 30
        M = torch.randn(n, n)
        A = M @ M.T
        B = torch.randn(n, 10)
        X0 = torch.zeros(n, 10)

        Ts_lanczos = [
            lanczos_linear_system(lambda x: A@x, x0, b, m)[2]
            for x0, b in zip(X0.T, B.T)
        ]
        _, Ts = mbcg(lambda X: A@X, lambda X: X, B, X0, m)

        for T_lanczos, T_mbcg in zip(Ts_lanczos, Ts):
            torch.testing.assert_allclose(T_lanczos, T_mbcg)


if __name__ == '__main__':
    unittest.main()
