from math import ceil
from typing import Tuple

import numpy as np
import torch
from scipy.special import gamma
from scipy.special import kv as modified_bessel_2nd


class Kernel:
    """
    Base abstract class for kernels.
    """
    def __init__(self, train_x: torch.Tensor):
        self.train_x = self._sanitize_input(train_x)

    def compute_kernel(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_kernel_and_grad(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def _sanitize_input(cls, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(dim=1)
        return X


class RadialKernel(Kernel):
    """
    Base abstract class for radial kernels, i.e. k(xi, xj) = f(|xi - xj|_2^2) for some function f.
    """
    def __init__(self, train_x: torch.Tensor):
        super(RadialKernel, self).__init__(train_x)

        self.D2 = self.get_squared_distance_matrix(self.train_x)

    @classmethod
    def get_squared_distance_matrix(cls, X: torch.Tensor, force_split=None):
        """
        Given an n x d data matrix, compute the n x n matrix of euclidean squared distances,
        i.e. D_ij = D_ji = |xi - xj|_2^2, with xi the ith row of X.
        """
        n, d = X.shape
        if force_split is None:
            if not X.is_cuda:
                # Assumes we won't split
                return cls._compute_distance_matrix(X, X)

            free_memory, _ = torch.cuda.mem_get_info()
            to_allocate = X.element_size() * n**2 * d
            available_memory = free_memory - X.element_size() * n**2 # account for final result
            available_memory *= 0.95
            factor = available_memory / to_allocate
            if factor > 1.0:
                return cls._compute_distance_matrix(X, X)

            n_splits = ceil(1 / factor) + 1
            print(f'{free_memory*1e-6:.5} MB of free memory, need '
                  f'{(to_allocate + free_memory - X.element_size() * n**2)*1e-6:.5} MB, '
                  f'splitting computations in {n_splits} parts')
        else:
            n_splits = force_split

        # Split calculations: take subsets sxd and nxd, diff matrix is s x n x d
        D = torch.empty(n, n, device=X.device, dtype=X.dtype)
        for id_start, id_end in cls.get_split_indices(n, n_splits):
            D[:, id_start:id_end] = cls._compute_distance_matrix(X[id_start:id_end], X)

        return D

    @classmethod
    def _compute_distance_matrix(cls, X1: torch.Tensor, X2: torch.Tensor):
        D = X1 - X2.unsqueeze(dim=1)
        # Compute squared 2-norm of differences
        D = (D ** 2).sum(-1)
        return D

    @staticmethod
    def get_split_indices(n, n_splits):
        ids = torch.linspace(0, n, n_splits+1, dtype=torch.int)
        ids = ids.tolist() + [-1]
        split_indices = list(zip(ids[:-1], ids[1:]))
        return split_indices[:-1]


class SquaredExponentialKernel(RadialKernel):

    def __init__(self, train_x: torch.Tensor):
        super(SquaredExponentialKernel, self).__init__(train_x)

    def compute_kernel(self, lengthscale: float) -> torch.Tensor:
        K = torch.exp(-self.D2 / lengthscale)
        return K

    def compute_kernel_and_grad(self, lengthscale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        K = self.compute_kernel(lengthscale)
        # Gradient w.r.t. lengthscale: K * D2 / l^2, elemwise
        grad = K * self.D2 / lengthscale**2
        # Only one hyperparam -> add dimension zero of size 1
        grad = grad.unsqueeze(dim=0)

        return K, grad


class MaternKernel(RadialKernel):

    def __init__(self, train_x: torch.Tensor):
        super(MaternKernel, self).__init__(train_x)

    def compute_kernel(self, lengthscale: float, nu: float) -> torch.Tensor:
        # Implementing this formula:
        # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

        # Compute Euclidean distances
        D = self.D2 ** 0.5
        # Factor with gamma function
        factor = 1 / gamma(nu) / 2 ** (nu - 1)
        # Compute the term inside nu exponent and inside bessel function
        M = D * (2 * nu) ** 0.5 / lengthscale
        # Evaluate bessel function
        B = modified_bessel_2nd(nu, M.numpy())

        # Put pieces together
        K = factor * (M ** nu) * torch.from_numpy(B)

        # When distance is zero, nan is computed because of Kv(0) / gamma(0), put one instead
        K[D == 0] = 1.0

        return K

    def compute_kernel_and_grad(self, lengthscale: float, nu: float) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


if __name__ == '__main__':
    N, d = 1000, 1000
    X = torch.randn(N, d, device='cuda', dtype=torch.double)
    l = 1.0
    k = RadialKernel()
    D = k.get_squared_distance_matrix(X)

    # Ktrue = torch.empty(N, N)
    # for i in range(N):
    #     for j in range(N):
    #         xi = X[i, :]
    #         xj = X[j, :]
    #         d = xi - xj
    #         z = np.exp(- d.dot(d) / l)
    #         Ktrue[i, j] = z
    #
    # err = np.abs(Ktrue - K).max()

#%%
