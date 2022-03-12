

import logging
from time import time
from typing import Callable, Union, Iterable

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]


def build_array_like(input_data: Iterable, reference_array: Array) -> Array:
    if isinstance(reference_array, np.ndarray):
        return np.array(input_data)
    elif isinstance(reference_array, torch.Tensor):
        return torch.tensor(input_data).to(reference_array.device)

    raise ValueError


def build_sym_tridiag_matrix(diag: Array, offdiag: Array):
    if isinstance(diag, np.ndarray):
        #M = sparse.diags((diag, offdiag, offdiag), (0, -1, 1))
        n = len(diag)
        indices = np.arange(n)
        M = np.zeros((n, n))
        M[(indices, indices)] = diag
        M[(indices[:-1], indices[1:])] = offdiag
        M[(indices[1:], indices[:-1])] = offdiag
    elif isinstance(diag, torch.Tensor):
        # TODO: don't forget to do to device?
        raise NotImplemented
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



#%%
