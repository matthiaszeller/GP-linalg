from typing import Optional

import torch
from scipy.optimize import minimize

from src.cg import mbcg
from src.inference import inference
from src.kernel import Kernel, SquaredExponentialKernel
from math import log

from src.precond import PartialCholesky


class GPModel:

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, kernel: Kernel,
                 hyperparams: torch.Tensor, use_tensorflow_pivoted_cholesky: bool = False):
        """
        Gaussian process inference model.
        :param train_x: training features
        :param train_y: training targets
        :param kernel: kernel class
        :param hyperparams: initial hyperparameters of the kernel function
        :param use_tensorflow_pivoted_cholesky: for numerical experiments to speedup computations
        """
        super(GPModel, self).__init__()

        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel

        self.hyperparams = hyperparams.clone()

        self._buffer_l = None
        self._buffer_dl = None
        self._ysolve = None

    def _call_l(self, theta, *args):
        """Helper function called by blackbox optimizer for evaluating likelihood"""
        self.hyperparams = theta
        L, dL, ysolve = self.compute_likelihood(*args)
        self._buffer_l = L
        self._buffer_dl = dL
        self._ysolve = ysolve
        # The optimizer will minimize the function -> minus sign to maximize
        return - L.item()

    def _call_dl(self, theta, *args):
        """Helper function called by blackbox optimizer for evaluating likelihood gradient"""
        # Again, minus sign
        return - self._buffer_dl.numpy()

    def compute_pred_cov(self, k=10, m=10, mbcg_tol=1e-10):
        P = PartialCholesky(self.kernel.K, k, self.kernel.sigma2)
        res, _ = mbcg(self.kernel.Khat_fun, P.inv_fun, self.kernel.K,
                      torch.zeros_like(self.kernel.K), m, tol=mbcg_tol)
        cov = self.kernel.K - self.kernel.K.T @ res
        return cov

    def train(self, niter: int = 10, k=10, N=10, m=10, mbcg_tol=1e-10):
        """
        Train the Gaussian process with mBCG algorithm and an optimizer
        :param niter: iterations of the optimizer
        :param k: rank of low-rank approximation, i.e. number of pivoted Cholesky steps
        :param N: number of probe vectors
        :param m: max number of Lanczos steps
        :param mbcg_tol: tolerance for mBCG algorithm
        :return: predictive mean, predictive covariance, marginal log likelihood
        """
        # TODO: only works for kernel with 2 hyperparams
        res = minimize(
            fun=self._call_l,
            x0=self.hyperparams,
            args=(k, N, m, mbcg_tol),
            jac=self._call_dl,
            options={
                'maxiter': niter
            },
            bounds=[
                (1e-6, 1e10),
                (1e-2, 1e10),
            ]
        )

        pred_mean = self.kernel.K @ self._ysolve
        pred_cov = self.compute_pred_cov(k, m, mbcg_tol)
        pred_cov = pred_cov.diag()
        return pred_mean, pred_cov, self._buffer_l

    # def train(self, lr: float = 0.001, niter: int = 10, callback=None):
    #     ysolve = None
    #
    #     opt = torch.optim.Adam((self.hyperparams, ), lr=lr)
    #     for i in range(niter+1):
    #         L, dL, ysolve = self.compute_likelihood()
    #         print(f'iter {i:<5} loss {f"{L:.4}":<10} hyperparams {self.hyperparams}')
    #
    #         # Gradient step
    #         #self.hyperparams += lr * dL
    #         self.hyperparams.grad = -dL
    #         opt.step()
    #
    #         if callback is not None:
    #             pred_mean = self.kernel.K @ ysolve
    #             callback(pred_mean)
    #
    #     pred_mean = self.kernel.K @ ysolve
    #     return pred_mean

    def compute_likelihood(self, k=10, N=20, m=10, mbcg_tol=1e-10, callback_inference=None, info=None):
        """
        Approximate marginal log likelihood with mBCG algorithm
        :param k: rank of pivoted Cholesky approximation
        :param N: number of probe vectors
        :param m: max number of Lanczos steps
        :param mbcg_tol: tolerance for mBCG
        :param callback_inference: callback function with input (ysolve, logdet, traces)
        :param info: None or a dictionnary to bookkeep the number of mBCG iterations
        :return: likelihood, gradient of likelihood, ysolve
        """
        _, gradK = self.kernel.compute_kernel_and_grad(self.hyperparams)

        ysolve, logdet, traces = inference(self.train_y, self.kernel, k=k, N=N, m=m, mbcg_tol=mbcg_tol, info=info)
        if callback_inference is not None:
            callback_inference(ysolve, logdet, traces)

        # Khat = self.kernel.K + torch.eye(self.train_x.shape[0]) * self.kernel.sigma2
        # true = torch.linalg.solve(Khat, self.train_y).ravel()
        # err_ysolve = torch.norm(true - ysolve) / torch.norm(true)
        # err_logdet = abs( (torch.logdet(Khat) - logdet) / torch.logdet(Khat) )

        # Likelihood
        n = self.train_y.shape[0]
        L = - 0.5 * logdet - 0.5 * self.train_y.T @ ysolve - n/2 * log(2*torch.pi)

        dL = []
        for i in range(gradK.shape[0]):
            dl = ysolve.T @ (gradK[i] @ ysolve) - traces[i]
            dL.append(0.5 * dl)

        dL = torch.tensor(dL)

        return L, dL, ysolve


if __name__ == '__main__':
    torch.set_default_dtype(torch.double)
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import multivariate_normal


    def true_likelihood(y, cov):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError

        pdf = multivariate_normal.pdf(y, mean=np.zeros_like(y), cov=cov)
        likelihood = np.log(pdf)
        return likelihood

    n, d = 100, 1
    sigma2 = 0.1
    x = torch.rand(n, d)
    f = lambda x: (2*torch.pi*x).sin()
    y = f(x) + torch.randn_like(x) * sigma2**0.5

    lengthscale = 0.5
    hyperparams = torch.tensor([0.01, lengthscale])

    kernel = SquaredExponentialKernel(x)
    model = GPModel(x, y, kernel, hyperparams)
    pred_mean, pred_cov, l = model.train()

    x = x.squeeze(-1)
    y = y.squeeze(-1)
    sid = torch.argsort(x, dim=0)
    plt.scatter(x[sid], y[sid], s=5, label='data')
    plt.plot(x[sid], f(x[sid]), '--k', label='true')
    plt.plot(x[sid], pred_mean[sid], label='prediction', color='green')
    pred_std = pred_cov ** 0.5
    plt.fill_between(x[sid], pred_mean[sid] - pred_std[sid], pred_mean[sid] + pred_std[sid],
                     label='+- std', color='green', alpha=.2)
    plt.legend()
    plt.show()
    #
    # L, dL, ysolve = model.compute_likelihood(k=15, N=1, m=20)
    #
    # L_true = true_likelihood(y, kernel.Khat())
    # ysolve_true = torch.linalg.solve(kernel.Khat(), y).ravel()
    #
    # err_ysolve = (ysolve_true - ysolve).norm() / ysolve_true.norm()
    # err_L = abs((L - L_true) / L_true)
    # a=1

    # n = 100
    # sigma2 = 0.1
    # X = torch.linspace(0, 1, n)
    # y = torch.sin(2*torch.pi* 2 * X) + torch.randn(n) * sigma2**0.5
    # plt.plot(X, y)
    # plt.show()
    #
    # k = SquaredExponentialKernel(X)
    # preds = []
    # model = GPModel(X, y, k, torch.tensor([.5, 1.]))
    # mu = model.train(lr=1e-3, niter=400, callback=lambda p: preds.append(p))
    #
    # for i, p in enumerate(preds):
    #     if i % 10 == 0:
    #         plt.plot(X, p, label=f'i={i}')
    #
    # plt.legend()
    # plt.show()
