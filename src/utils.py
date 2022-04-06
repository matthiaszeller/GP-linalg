
import logging
from math import log, exp
from time import time
from typing import Callable, Union, Iterable

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]


def compute_logdet_bounds(eigs: torch.Tensor, relerr: float, proba_error: float, info = None):
    """
    Compute the bounds on the number of probe vectors and number of Lanczos steps
    :param eigs: eigenvalues of A (SPD)
    :param relerr: relative error on the logdet
    :param proba_error: upper bound on the probability of error
    :return: Nmin, m_min
    """
    assert (eigs > 0).all()
    # Condition number of A
    condA = eigs.max() / eigs.min()

    # Compute eigenvalues of log(A)
    eigs = eigs.log()
    # Compute the term with Frobenius norm, spectral norm and Trace of log(A)
    frob_square = (eigs ** 2).sum()
    spectral = eigs.abs().max()
    abstrace = abs(eigs.sum())

    N_min = (16 / relerr**2) * (frob_square + relerr * spectral) / (abstrace**2) * log(4 / proba_error)
    n = eigs.shape[0] # matrix size

    m_min = (condA + 1)**0.5 / 4 * log(4 / relerr / abstrace * n**2 * log(2*condA) * (1 + (condA + 1)**0.5))

    N_min, m_min = N_min.item(), m_min.item()

    if info is not None:
        info['eigs'] = eigs
        info['condA'] = condA.item()
        info['frob2'] = frob_square.item()
        info['norm2'] = spectral.item()
        info['trabs2'] = abstrace.item()**2

    return N_min, m_min


def build_array_like(input_data: Iterable, reference_array: Array) -> Array:
    if isinstance(reference_array, np.ndarray):
        return np.array(input_data)
    elif isinstance(reference_array, torch.Tensor):
        return torch.tensor(input_data).to(reference_array.device)

    raise ValueError


def build_sym_tridiag_matrix(diag: Array, offdiag: Array):
    # TODO: handle sparse? will break lanczos
    n = len(diag)
    if isinstance(diag, np.ndarray):
        # M = sparse.diags((diag, offdiag, offdiag), (0, -1, 1))
        indices = np.arange(n)
        M = np.zeros((n, n))
        M[(indices, indices)] = diag
        M[(indices[:-1], indices[1:])] = offdiag
        M[(indices[1:], indices[:-1])] = offdiag
    elif isinstance(diag, torch.Tensor):
        indices = torch.arange(n)
        M = torch.zeros((n, n), dtype=diag.dtype, device=diag.device)
        M[(indices, indices)] = diag
        M[(indices[:-1], indices[1:])] = offdiag
        M[(indices[1:], indices[:-1])] = offdiag
    else:
        raise ValueError('unrecognized type')

    return M


def timeit_decorate(reps=10):
    """
    Simple timer. Usage:
        >>> @timeit_decorate(100)
        >>> def fun_compute(A, b):
        >>>     return np.linalg.solve(A, b)
        >>> fun_compute(np.random.randn(10, 10), np.random.randn(10)) # returns mean, std

    :param reps: number of repetitions
    :return: mean, std computed over `reps` calls to the decorated function
    """
    def decorator(fun):
        def inner(*args, **kwargs):
            times = []
            for _ in range(reps):
                start = time()
                fun(*args, **kwargs)
                times.append(time() - start)

            return np.mean(times), np.std(times)

        return inner

    if not isinstance(reps, int):
        logging.warning('you probably called timeit decorator wrong (i.e., without arguments)')

    return decorator


def timer(fun_compute: Callable, reps: int, fun_post: Callable = None) -> Callable:
    """
    Time a function execution and post-process the results (e.g., perform sanity checks)
    Usage:
        >>> fun_compute = lambda A, x: np.linalg.solve(A, b)
        >>> def fun_post(xhat):
        >>>     if np.linalg.norm(np.abs(xhat - xtrue) < 1e-5):
        >>>         print('bad solution')
        >>> A, b = np.random.randn(10, 10), np.random.randn(10)
        >>> xtrue = ...
        >>> mu, std = timer(fun_compute, 10, fun_post)(A, b)
    :param fun_compute: the function to time
    :param reps: the number of repetitions
    :param fun_post: the post processing function, takes as input the output of fun_compute
    :return: a wrapper function that takes the same input as fun_compute
    """
    def wrapped(*args, **kwargs):
        times = []
        for _ in range(reps):
            s = time()
            res = fun_compute(*args, **kwargs)
            times.append(time() - s)

            if fun_post is not None:
                if isinstance(res, tuple):
                    fun_post(*res)
                else:
                    fun_post(res)

        return np.mean(times), np.std(times)

    return wrapped

# %%
