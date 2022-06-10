

import torch
from matplotlib import pyplot as plt

from src.kernel import SquaredExponentialKernel
from src.model import GPModel
# Use 64 bits floating-point numbers
torch.set_default_dtype(torch.double)

# Parameters: dataset size n, data dimensionality d, noise variance sigma2
n, d = 50, 1
sigma2 = 0.1
x = torch.rand(n, d)
# Define targets: sinus curve
f = lambda x: (2*torch.pi*x).sin()
# Noisy observations
y = f(x) + torch.randn(n, 1) * sigma2**0.5

# Initial guesses for the hyperparameters
lengthscale_hat = 0.5
sigma2_hat = 0.01
hyperparams = torch.tensor([sigma2_hat, lengthscale_hat])

# Instantiate the kernel and the GP model
kernel = SquaredExponentialKernel(x)
model = GPModel(x, y, kernel, hyperparams)
# Train the model
pred_mean, pred_cov, l = model.train()

# Predict on test points (uniform grid)
grid = torch.linspace(0, 1, 100)
mu, cov = model.compute_prediction(grid)
std = cov**0.5

# Plot
plt.scatter(x, y, s=5, label='data')
plt.plot(grid, f(grid), '--k', label='true function')
plt.plot(grid, mu, label='predictive mean', color='g')
plt.fill_between(grid, mu - std, mu + std, label='predictive std', color='g', alpha=.3)
plt.legend()
plt.xlabel('Input $x$'); plt.ylabel('Output $f(x)$')
plt.show()

