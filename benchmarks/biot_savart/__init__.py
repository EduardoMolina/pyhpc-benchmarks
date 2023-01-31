import math
import importlib
import functools


def generate_inputs(size):
    import numpy as np

    np.random.seed(17)

    shape = (
        math.ceil(20 * size ** (1 / 3)),
        3,
    )

    points     = np.random.uniform(-12, 20, size=shape)
    back_right = np.random.uniform(1e-2, 10, size=shape)
    front_right = np.random.uniform(1e-2, 10, size=shape)
    front_left = np.random.uniform(1e-2, 10, size=shape)
    back_left = np.random.uniform(1e-2, 10, size=shape)
    strengths = np.random.uniform(1e-2, 10, size=shape[0])
    ages = np.zeros(shape=shape[0])
    nu = 1.5e-6
    return points, back_right, front_right, front_left, back_left, strengths, ages, nu


def try_import(backend):
    try:
        return importlib.import_module(f".biot_savart_{backend}", __name__)
    except ImportError:
        return None


def get_callable(backend, size, device="cpu"):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, "prepare_inputs"):
        inputs = backend_module.prepare_inputs(*inputs, device=device)
    return functools.partial(backend_module.run, *inputs, device=device)


__implementations__ = (
    "jax",
    "numba",
    "numpy",
)
