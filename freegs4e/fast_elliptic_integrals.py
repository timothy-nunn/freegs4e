"""Implements complete first and second order elliptic integrals.
[1] Arndt, J. (2010). Matters Computational: ideas, algorithms, source code. Springer Science & Business Media.
"""

import numpy as np

try:
    from numba import njit, vectorize
except ImportError:

    def njit(*args, **kwargs):
        return lambda f: f

    vectorize = njit


@njit(cache=True, fastmath=True)
def _R_series_component(a, b, n):
    return 4**n * (a**4 - ((a**2 + b**2) / 2) ** 2)


@njit(cache=True, fastmath=True)
def _agm4(ak, bk):

    R = _R_series_component(ak, bk, 0)
    n = 0

    while np.abs(ak - bk) > 1e-10:
        n += 1
        ak1 = (ak + bk) / 2
        gk1 = (ak - bk) / 2
        bk1 = np.power((ak1**4) - (gk1**4), 1 / 4)

        ak = ak1
        bk = bk1
        R += _R_series_component(ak, bk, n)

    return (ak + bk) / 2, 1 - R


@njit(cache=True, fastmath=True)
def _agm(a, b):
    result, R = _agm4(np.sqrt(a), np.sqrt(b))
    return result**2, R


@njit(cache=True, fastmath=True)
def _ellipk_scalar(m):
    if m == 0.0:
        return np.pi / 2.0
    if m == 1.0:
        return np.inf
    elif m > 1.0:
        return np.nan

    agm, _ = _agm(1.0, np.sqrt(1.0 - m))
    return np.pi / (2.0 * agm)


@njit(cache=True, fastmath=True)
def _ellipe_scalar(m):
    if m == 0.0:
        return np.pi / 2.0
    elif m == 1.0:
        return 1.0
    elif m > 1.0:
        return np.nan
    elif m < 0:
        return _ellipe_scalar(m / (m - 1)) * np.sqrt(1 - m)

    agm_result, R = _agm(1.0, np.sqrt(1.0 - m))
    return (np.pi / (2.0 * agm_result)) * R


@vectorize(nopython=True)
def ellipe(m):
    return _ellipe_scalar(m)


@vectorize(nopython=True)
def ellipk(m):
    return _ellipk_scalar(m)
