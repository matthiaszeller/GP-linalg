from math import ceil
from typing import Tuple, List

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
        self.sigma2 = 1.0
        self.K = None
        self.grad = None

    def Khat(self):
        return self.K + self.sigma2 * torch.eye(self.K.shape[0])

    def Khat_fun(self, y: torch.Tensor):
        return self.K @ y + self.sigma2 * y

    def compute_kernel(self, hyperparams: torch.Tensor) -> torch.Tensor:
        self.set_hyperparams(hyperparams)
        self.K = self._compute_kernel()
        return self.K

    def _compute_kernel(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_kernel_and_grad(self, hyperparams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute kernel
        self.compute_kernel(hyperparams)
        # Let child class compute gradient with respect to its own hyperparams
        grad = self._compute_gradient()
        # Compute the gradient w.r.t. noise variance: d Khat / dsigma2 = 2*sigma*I
        grad_noise = 2 * self.sigma2**0.5 * \
                     torch.eye(self.train_x.shape[0], dtype=self.train_x.dtype, device=self.train_x.device)

        # Prepend the gradient w.r.t. sigma2
        self.grad = torch.concat((grad_noise.unsqueeze(dim=0), grad))
        return self.K, self.grad

    def _compute_gradient(self) -> torch.Tensor:
        raise NotImplementedError

    def set_hyperparams(self, hyperparams: torch.Tensor):
        raise NotImplementedError

    def get_hyperparams(self) -> Tuple[torch.Tensor, List[str]]:
        theta, desc = self._get_hyperparams()
        theta = torch.concat((torch.tensor([self.sigma2]), theta))
        desc = ['sigma2'] + desc
        return theta, desc

    def _get_hyperparams(self) -> Tuple[torch.Tensor, List[str]]:
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

        self.D = self.get_distance_matrix(self.train_x)
        self.lengthscale = 1.0

    def set_hyperparams(self, hyperparams: torch.Tensor):
        self.sigma2, self.lengthscale = hyperparams
        if self.lengthscale <= 0.0:
            raise ValueError

    @staticmethod
    def get_distance_matrix(X: torch.Tensor) -> torch.Tensor:
        return torch.cdist(X, X)


class SquaredExponentialKernel(RadialKernel):

    def __init__(self, train_x: torch.Tensor):
        super(SquaredExponentialKernel, self).__init__(train_x)

        self.lengthscale = 1.0

    def _get_hyperparams(self) -> Tuple[torch.Tensor, List[str]]:
        return torch.tensor([self.lengthscale]), ['lengthscale']

    def _compute_kernel(self) -> torch.Tensor:
        K = torch.exp(-self.D**2 / self.lengthscale)
        return K

    def _compute_gradient(self) -> torch.Tensor:
        # Gradient w.r.t. lengthscale: K * D2 / l^2, elemwise
        grad = self.K * self.D**2 / self.lengthscale**2
        # Only one hyperparam -> add dimension zero of size 1
        grad = grad.unsqueeze(dim=0)

        return grad


class MaternKernel(RadialKernel):

    def __init__(self, train_x: torch.Tensor):
        super(MaternKernel, self).__init__(train_x)
        self.nu = 1.5

    def set_hyperparams(self, hyperparams: torch.Tensor):
        self.sigma2, self.lengthscale, self.nu = hyperparams
        self.nu = self.nu.item()
        self.lengthscale = self.lengthscale.item()

    def _compute_kernel(self) -> torch.Tensor:
        # Implementing this formula:
        # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

        # Compute Euclidean distancesself.
        # Factor with gamma function
        factor = 1 / gamma(self.nu) / 2 ** (self.nu - 1)
        # Compute the term inside nu exponent and inside bessel function
        M = self.D * (2 * self.nu) ** 0.5 / self.lengthscale
        # Evaluate bessel function
        B = modified_bessel_2nd(self.nu, M.numpy())

        # Put pieces together
        K = factor * (M ** self.nu) * torch.from_numpy(B)

        # When distance is zero, nan is computed because of Kv(0) / gamma(0), put one instead
        K[self.D == 0] = 1.0

        return K

    def _compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


if __name__ == '__main__':
    N, d = 100, 1000
    X = torch.linspace(0, 1, N)
    l = 1.0
    k = SquaredExponentialKernel(X)
    theta = torch.tensor([0.5, 1.])
    K, grad = k.compute_kernel_and_grad(theta)

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
