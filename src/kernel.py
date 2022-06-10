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
    # Class-level settings for *constrained* optimization of hyperparameters
    HYPERPARAMS_BOUNDS = None # child classes must define this

    def __init__(self, train_x: torch.Tensor):
        self.train_x = self._sanitize_input(train_x)
        # Hyperparameter: noise variance, child classes will define other parameters
        self.sigma2 = None
        # Kernel matrix and its gradient
        self.K = None
        self.grad = None

    def Khat(self):
        """Explicitly compute the kernel matrix with an added diagonal scaled by noise variance.
        compute_kernel or compute_kernel_and_grad must have been be called before."""
        return self.K + self.sigma2 * torch.eye(self.K.shape[0])

    def Khat_fun(self, y: torch.Tensor):
        """Oracle y |-> Khat y.
        compute_kernel or compute_kernel_and_grad must have been called before."""
        return self.K @ y + self.sigma2 * y

    def compute_kernel(self, hyperparams: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix with the given hyperparameters.
        If you wish to compute the gradient as well, call instead `compute_kernel_and_grad`."""
        self.set_hyperparams(hyperparams)
        self.K = self._compute_kernel()
        return self.K

    def _compute_kernel(self) -> torch.Tensor:
        """Protected method for computing the kernel"""
        # Child classes must implement this protected method
        raise NotImplementedError

    def compute_kernel_and_grad(self, hyperparams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the kernel matrix and its gradient."""
        # Compute kernel first
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
        # Child classes must implement this protected method
        raise NotImplementedError

    def set_hyperparams(self, hyperparams: torch.Tensor):
        # Child classes must implement this public method
        raise NotImplementedError

    def get_hyperparams(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_test_kernel(self, test_x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix with entries k(x, x'), x train sample, x' test sample"""
        raise NotImplementedError

    def new_data(self, X: torch.Tensor):
        """Return of copy of the kernel computed on the new data X"""
        # Instanciate new object
        other = self.__class__(X)
        # Compute its kernel matrix
        other.compute_kernel(self.get_hyperparams())
        return other

    @classmethod
    def _sanitize_input(cls, X: torch.Tensor) -> torch.Tensor:
        """Cast to a pytorch tensor if input is a numpy array, cast to matrix if input is vector."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
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
        """Update the kernel hyperparameters: a tensor of (sigma2, lengthscale)."""
        self.sigma2, self.lengthscale = hyperparams
        if self.lengthscale <= 0.0:
            raise ValueError

    @classmethod
    def get_distance_matrix(cls, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """All radial kernels depend on the euclidean distance matrix."""
        if Y is None:
            Y = X
        else:
            X = cls._sanitize_input(X)
            Y = cls._sanitize_input(Y)
        return torch.cdist(X, Y)

    def compute_test_kernel(self, test_x: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix with train and test data, k(X_train, X_test).
        Useful for predicting test samples that were not in the training set."""
        # Temporarily change the distance matrix
        backup = self.D
        self.D = self.get_distance_matrix(test_x, self.train_x)
        # Compute kernel
        K = self._compute_kernel()
        # Restore distance matrix
        self.D = backup
        return K


class SquaredExponentialKernel(RadialKernel):
    """
    Squared exponential kernel k(x, y) = exp(- ||x-y||_2 / l ) with l the lengthscale.
    Often loosely called the RBF kernel.
    """

    # Bounds for sigma2, lengthscale
    HYPERPARAMS_BOUNDS = (
        (1e-6, 1e2),
        (1e-2, 1e2)
    )

    def __init__(self, train_x: torch.Tensor):
        super(SquaredExponentialKernel, self).__init__(train_x)

        self.lengthscale = 1.0

    def get_hyperparams(self) -> torch.Tensor:
        """Return the tensor of the kernel hyperparameters."""
        return torch.tensor([self.sigma2, self.lengthscale])

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
    """
    Matern kernel, see e.g.
    https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

    The gradient of this kernel is not implemented yet, this would require differentiating the bessel function
    with respect to its order.
    """

    # Bounds for sigma2, lengthscale, nu
    HYPERPARAMS_BOUNDS = None # gradient for Matern is not implemented anyway

    def __init__(self, train_x: torch.Tensor):
        super(MaternKernel, self).__init__(train_x)
        self.nu = 1.5

    def set_hyperparams(self, hyperparams: torch.Tensor):
        """Update the kernel hyperparameters: (sigma2, lengthscale, nu).
        Nu is the smoothness parameter."""
        self.sigma2, self.lengthscale, self.nu = hyperparams
        self.nu = self.nu.item()
        self.lengthscale = self.lengthscale.item()

    def get_hyperparams(self) -> torch.Tensor:
        """Return the tensor of the kernel hyperparameters."""
        return torch.tensor([self.sigma2, self.lengthscale, self.nu])

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
        # TODO: implement this, currently return only NaNs
        grad = torch.empty_like(self.K) * torch.nan
        grad = grad.unsqueeze(dim=0)
        return grad


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
