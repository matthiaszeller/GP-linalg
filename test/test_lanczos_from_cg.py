

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

            torch.testing.assert_allclose(T_mbcg, T_lanczos, atol=1e-10, rtol=1e-10)

    def test_lanczos_from_cg_batched(self):
        n, m = 100, 30
        M = torch.randn(n, n)
        A = M @ M.T

        n_batch = 10
        B = torch.randn(n, n_batch)
        X0 = torch.randn(n, n_batch)

        X, Ts_mbcg = mbcg(lambda X: A@X, lambda X: X, B, X0, m)
        self.assertEqual(len(Ts_mbcg), n_batch)

        for i, T_mbcg in enumerate(Ts_mbcg):
            b, x0 = B[:, i], X0[:, i]
            xm, _, T_lanczos = lanczos_linear_system(lambda x: A@b, x0, b, m)
            torch.testing.assert_allclose(xm, X[:, i], rtol=1e-10, atol=1e-10)
            torch.testing.assert_allclose(T_mbcg, T_lanczos, atol=1e-10, rtol=1e-10)


        Ts_lanczos = []
        for i in range(n_batch):
            b, x0 = B[:, i], X0[:, i]
            _, _, T = lanczos_linear_system(lambda x: A@b, x0, b, m)
            Ts_lanczos.append(T)

        torch.testing.assert_allclose(
            torch.stack(Ts_mbcg),
            torch.stack(Ts_lanczos), rtol=1e-10, atol=1e-10
        )


if __name__ == '__main__':
    unittest.main()
