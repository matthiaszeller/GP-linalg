from typing import Optional

import torch
from scipy.optimize import minimize

from src.cg import mbcg
from src.inference import inference
from src.kernel import Kernel, SquaredExponentialKernel
from math import log
import numpy as np
from src.precond import PartialCholesky


class GPModel:

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, kernel: Kernel,
                 hyperparams: torch.Tensor, use_tensorflow_pivoted_cholesky: bool = False,
                 compute_true_quantities: bool = False):
        """
        Gaussian process inference model.
        :param train_x: training features
        :param train_y: training targets
        :param kernel: kernel class
        :param hyperparams: initial hyperparameters of the kernel function
        :param use_tensorflow_pivoted_cholesky: for numerical experiments to speedup computations
        :param compute_true_quantities: whether to compute the ground truth likelihood for numerical experiments
        """
        super(GPModel, self).__init__()

        if isinstance(train_x, np.ndarray):
            train_x = torch.from_numpy(train_x)
        if isinstance(train_y, np.ndarray):
            train_y = torch.from_numpy(train_y)

        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel
        self.verbose = False

        self.hyperparams = hyperparams.clone()

        # Helper variables for the blackbox optimizer
        self._buffer_l = None
        self._buffer_dl = None
        self._ysolve = None

        # Track likelihood
        self.training_likelihoods = []
        self.training_likelihoods_grad = []

        # For numerical experiments
        self._compute_true_quantities = compute_true_quantities
        self._true_training_likelihoods = []
        self._true_training_likelihoods_grad = []

    def train(self, niter: int = 10, k=10, N=10, m=10, mbcg_tol=1e-10, verbose: bool = True):
        """
        Train the Gaussian process using constrained optimization on the likelihood,
        where the likelihood estimation is based on mBCG output.

        :param niter: iterations of the optimizer
        :param k: rank of low-rank approximation, i.e. number of pivoted Cholesky steps
        :param N: number of probe vectors for stochastic trace estimation
        :param m: max number of CG/Lanczos steps
        :param mbcg_tol: tolerance for mBCG algorithm
        :return: predictive training mean, predictive training covariance, marginal log likelihood
        """
        self.verbose = verbose
        res = minimize(
            fun=self._call_l,
            x0=self.hyperparams,
            args=(k, N, m, mbcg_tol),
            jac=self._call_dl,
            options={
                'maxiter': niter
            },
            bounds=self.kernel.HYPERPARAMS_BOUNDS
        )

        mean, cov = self.compute_prediction(self.train_x)
        return mean, cov, self._buffer_l

    def compute_likelihood(self, k=10, N=10, m=20, mbcg_tol=1e-10, callback_inference=None, info=None):
        """
        Approximate marginal log likelihood based on mBCG output.

        :param k: rank of pivoted Cholesky approximation
        :param N: number of probe vectors
        :param m: max number of Lanczos steps
        :param mbcg_tol: tolerance for mBCG
        :param callback_inference: callback function with input (ysolve, logdet, traces)
        :param info: None or a dictionnary to bookkeep the number of mBCG iterations
        :return: likelihood, gradient of likelihood, ysolve
        """
        # Evaluate the kernel matrix and the gradient
        _, gradK = self.kernel.compute_kernel_and_grad(self.hyperparams)

        # Run mBCG algorithm
        ysolve, logdet, traces = inference(self.train_y, self.kernel, k=k, N=N, m=m, mbcg_tol=mbcg_tol, info=info)

        # Callback for results of mBCG (debugging, numerical experiments)
        if callback_inference is not None:
            callback_inference(ysolve, logdet, traces)

        # Khat = self.kernel.K + torch.eye(self.train_x.shape[0]) * self.kernel.sigma2
        # true = torch.linalg.solve(Khat, self.train_y).ravel()
        # err_ysolve = torch.norm(true - ysolve) / torch.norm(true)
        # err_logdet = abs( (torch.logdet(Khat) - logdet) / torch.logdet(Khat) )

        # Likelihood computation based on mBCG output
        n = self.train_y.shape[0]
        L = - 0.5 * logdet - 0.5 * self.train_y.T @ ysolve - n/2 * log(2*torch.pi)

        # Compute gradient of likelihood
        dL = []
        for i in range(gradK.shape[0]):
            dl = ysolve.T @ (gradK[i] @ ysolve) - traces[i]
            dL.append(0.5 * dl)

        dL = torch.tensor(dL)

        # Compute exact quantities, for debugging and numerical experiments
        if self._compute_true_quantities:
            true_logdet = torch.logdet(self.kernel.Khat())
            A = torch.linalg.solve(self.kernel.Khat(), self.kernel.grad)
            true_traces = torch.tensor([M.trace() for M in A])
            true_ysolve = torch.linalg.solve(self.kernel.Khat(), self.train_y)
            # Likelihood
            true_L = -0.5 * true_logdet - 0.5 * self.train_y.T @ true_ysolve - n/2 * log(2*torch.pi)
            true_dL = []
            for i in range(gradK.shape[0]):
                dl = true_ysolve.T @ gradK[i] @ true_ysolve - true_traces[i]
                true_dL.append(0.5 * dl)
            # Computations bookkeeping
            self._true_training_likelihoods.append(true_L)
            self._true_training_likelihoods_grad.append(torch.tensor(true_dL))

        return L, dL, ysolve

    def _call_l(self, theta, *args):
        """Helper function called by blackbox optimizer for evaluating likelihood"""
        # Update the hyperparams
        self.hyperparams = theta
        # Compute likelihood
        L, dL, ysolve = self.compute_likelihood(*args)
        if self.verbose:
            pred_mean = self.kernel.K @ ysolve
            mse = ((self.train_y - pred_mean) ** 2).mean()
            print(f'likelihood {f"{L.item():.5}":<15} MSE {f"{mse.item():.5}":<15} params {self.hyperparams.tolist()}')
        # Store for blackbox optimizer
        self._buffer_l = L
        self._buffer_dl = dL
        self._ysolve = ysolve
        # Keep track of likelihood
        self.training_likelihoods.append(L)
        self.training_likelihoods_grad.append(dL)
        # The optimizer will minimize the function -> minus sign to maximize
        return - L.item()

    def _call_dl(self, theta, *args):
        """Helper function called by blackbox optimizer for evaluating likelihood gradient"""
        # Again, minus sign
        return - self._buffer_dl.numpy()

    def compute_prediction(self, xtest):
        """This is not part of the training.
        A real implementation should handle this part more efficiently,
        but we focused on hyperparameter estimation"""
        K_test = self.kernel.compute_test_kernel(xtest)
        mean = K_test @ self._ysolve

        test_solve = torch.linalg.solve(self.kernel.Khat(), K_test.T)
        cov = self.kernel.new_data(xtest).K - K_test @ test_solve

        return mean, cov.diag()


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

    n, d = 50, 1
    sigma2 = 0.05
    x1, x2 = torch.rand(n//4, d), torch.rand(3*n//4, d)
    x = torch.concat((x1 * 0.2, x2 * 0.6 + 0.4))
    #x = torch.rand(n, d)
    f = lambda x: (2.5*torch.pi*x).sin()
    y = f(x) + torch.randn_like(x) * sigma2**0.5

    lengthscale = 0.5
    hyperparams = torch.tensor([0.01, lengthscale])

    kernel = SquaredExponentialKernel(x)
    model = GPModel(x, y, kernel, hyperparams, compute_true_quantities=True)
    pred_mean, pred_cov, l = model.train()

    # Plotting training quantities
    # x = x.squeeze(-1)
    # y = y.squeeze(-1)
    # sid = torch.argsort(x, dim=0)
    # plt.scatter(x[sid], y[sid], s=5, label='data')
    # plt.plot(x[sid], f(x[sid]), '--k', label='true')
    # plt.plot(x[sid], pred_mean[sid], label='prediction', color='green')
    # pred_std = pred_cov ** 0.5
    # plt.fill_between(x[sid], pred_mean[sid] - pred_std[sid], pred_mean[sid] + pred_std[sid],
    #                  label='+- std', color='green', alpha=.2)
    # plt.legend()
    # plt.show()

    grid = torch.linspace(x.min(), x.max(), 100)
    pred_mean, pred_cov = model.compute_prediction(grid)

    x = x.squeeze(-1)
    y = y.squeeze(-1)
    sid = torch.argsort(x, dim=0)
    plt.scatter(x[sid], y[sid], s=5, label='data')
    plt.plot(grid, f(grid), '--k', label='true')
    plt.plot(grid, pred_mean, label='prediction', color='green')
    pred_std = pred_cov ** 0.5
    plt.fill_between(grid, pred_mean - pred_std, pred_mean + pred_std,
                     label='+- std', color='green', alpha=.2)
    plt.legend()
    plt.show()

    # import pandas as pd
    # print('dataset \n\n')
    # #df = pd.read_csv('../tutorial/data/airfoil_self_noise.dat', sep='\t', header=None)
    # df = pd.read_csv('../tutorial/data/dataset_airfoil.csv')
    # X = df.values[:, :-1]
    # y = df.values[:, -1]
    # X = (X - X.mean(0)) / X.std(0)
    # y = (y - y.mean()) / y.std()
    # kernel = SquaredExponentialKernel(X)
    # kernel.compute_kernel(hyperparams)
    # P = PartialCholesky(kernel.K, 10, 0.01)
    # model = GPModel(X, y, kernel, hyperparams, compute_true_quantities=True)
    # pred_mean, pred_cov, l = model.train()

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
