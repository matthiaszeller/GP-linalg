from math import ceil

import numpy as np
import torch


class Kernel:
    """
    Base abstract class for kernels.
    """

    def compute(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix.
        :param X: n x d input matrix, n is number of samples, d number of features
        :return: n x n kernel matrix K with K_ij = K_ji = k(xi, xj), k the kernel function
        """
        # --- Preprocessing
        # If the input is a 1D tensor, make it an n x 1 matrix (i.e only one feature, d=1)
        if X.ndim == 1:
            X = X.unsqueeze(dim=1)
        # --- Make child class compute with sanitized input
        return self._compute(X)

    def _compute(self, X: torch.Tensor) -> torch.Tensor:
        # Child classes must implement this
        pass


class RadialKernel(Kernel):
    """
    Base abstract class for radial kernels, i.e. k(xi, xj) = f(|xi - xj|_2^2) for some function f.
    """

    @classmethod
    def get_distance_matrix(cls, X: torch.Tensor, force_split=None):
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


class RBFKernel(RadialKernel):

    def __init__(self, charact_length: float = 1.0):
        self.charact_length = charact_length

    def _compute(self, X: torch.Tensor) -> torch.Tensor:
        D = self.get_distance_matrix(X)
        K = torch.exp(-D / self.charact_length)
        return K


if __name__ == '__main__':
    N, d = 1000, 1000
    X = torch.randn(N, d, device='cuda', dtype=torch.double)
    l = 1.0
    k = RadialKernel()
    D = k.get_distance_matrix(X)

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
