

# GP-linalg

Randomized Algorithms for Gaussian Process Inference.
Semester project, [ANCHP](https://www.epfl.ch/labs/anchp/) Lab, EPFL.

## About

Gaussian process inference engine based on GPyTorch ([article](https://arxiv.org/pdf/1809.11165.pdf), 
[code](https://github.com/cornellius-gp/gpytorch)).
Theoretical analysis and numerical experiments can be found in the [report] of this repository. 
The engine relies on conjugate gradients and stochastic Lanczos quadrature,
where partial pivoted Cholesky is used as a low rank approximation for preconditioning.  

## How-to

### Getting started

One must first install the Python packages used in this project.
The environment file is provided as `environment.yaml`, which also contains the Jupyter Lab package to view the 
notebook. The easiest method for installing is 

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (a lightweight version of anaconda)
2. Create the conda `gpr` (Gaussian Process Regression) environment that is later used to run the code:
```shell
conda env create -f environment.yml
```

One installation is complete, one must activate the environment:
```shell
user@host:/path$ conda activate gpr
(gpr) user@host:/path$ # gpr now appears, i.e. the environment is active 
```


### Regression Example

Run the script `run_example.py` to see the inference engine in action on a toy dataset.
One should first activate the `gpr` environment and then run the script: 
```shell
$ conda activate gpr
(gpr) $ python run_example.py
```

This prints the likelihood at each optimization step and produces a figure.


## Project Structure

The user entry point is the `GPModel` class in `models.py` and the kernels in `kernel.py` 
. 

### File description

* `src` folder contains
  * `cg.py`: mBCG algorithm
  * `quadrature.py`: Lanczos quadrature
  * `chol.py`: pivoted Cholesky algorithm
  * `precond.py`: routines for preconditioners of kernel matrices
  * `kernel.py`: squared exponential and Matern kernels
  * `lanczos.py`: Lanczos method for linear systems, used for debugging
  * `inference.py`: put pieces together to compute linear solve, logdet and trace term
  * `model.py`: compute likelihood and its gradient, perform optimization to fit hyperparameters
* `tests`: unit tests for algorithms correctness

## Numerical Experiments

All numerical experiments are gathered in the Jupyter notebook: `notebook/Numerical Experiments.ipynb`.
In order to run it, `cd` into the root of this project and run:

```shell
user@host:/path$ conda activate gpr
(gpr) user@host:/path$ jupyter lab
```

This should open a browser tab with a jupyter server in which you can view and run the notebook. 
Make sure to run the cells in order.

Some cells perform heavy computations, those start with `%%time` and the running time 
appears under the cell. Those cells can be skipped as their results can be loaded in the following cells. 
