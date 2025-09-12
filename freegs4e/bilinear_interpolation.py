import numpy as np

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

    prange = range

# The following two functions (_biliint_prepare_arrays and _biliint_final_calculation)
# contain code that it does not make to sense to parallelise.
# When writing biliint as fully parallel, it massively INCREASED the runtime.
# This is because it required using np.repeat to make certain arrays the same shape
# (which is required when doing +-*/ in parallel mode)
# and this required copying massive amounts of data
@njit(cache=True, fastmath=True, parallel=False)
def _biliint_prepare_arrays(R, Z, psi, points):
    nx, ny = np.shape(psi)

    R1d = R[:, :1]
    Z1d = Z[:1, :]

    points = points.reshape(2, -1)

    points_R = R1d - points[:1, :]
    points_Z = Z1d.T - points[1:2, :]

    idxs_R = np.sum(points_R < 0, axis=0)
    idxs_Z = np.sum(points_Z < 0, axis=0)

    idxs_R = np.where(idxs_R < nx, idxs_R, nx - 1)
    idxs_Z = np.where(idxs_Z < ny, idxs_Z, ny - 1)

    return points, points_R, idxs_R, points_Z, idxs_Z


@njit(cache=True, fastmath=True, parallel=True)
def biliint(R, Z, psi, points):
    """Simple bilinear interpolation of 2d map

    Parameters
    ----------
    R : np.array
        R coordinates on 2d grid
    Z : np.array
        Z coordinates on 2d grid
    psi : np.array
        function values on 2d grid
    points : np.array
        coordinates where the interpolation is sought
        shape (2, whatever)

    Returns
    -------
    np.array
        interpolated values, same shape as points: (1, whatever)
    """
    points_shape = np.shape(points)
    points, points_R, idxs_R, points_Z, idxs_Z = _biliint_prepare_arrays(R, Z, psi, points)

    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    dRdZ = dR * dZ
    len_points = np.shape(points)[1]

    qq = np.empty((len_points, 2, 2))

    for i in prange(len_points):
        qq[i, 0, 0] = psi[idxs_R[i] - 1, idxs_Z[i] - 1]
        qq[i, 0, 1] = psi[idxs_R[i] - 1, idxs_Z[i]]
        qq[i, 1, 0] = psi[idxs_R[i], idxs_Z[i] - 1]
        qq[i, 1, 1] = psi[idxs_R[i], idxs_Z[i]]


    xx = np.empty((len_points, 2))
    for i in prange(len_points):
        xx[i, 0] = points_R[idxs_R[i], i]
        xx[i, 1] = points_R[idxs_R[i] - 1, i]
    
    xx[:, 1] *= -1

    yy = np.empty((len_points, 2))
    for i in prange(len_points):
        yy[i, 0] = points_Z[idxs_Z[i], i]
        yy[i, 1] = points_Z[idxs_Z[i] - 1, i]

    yy[:, 1] *= -1

    yy_shape = np.stack((yy, yy), 1)
    vals = (
        np.sum(np.sum(qq * yy_shape, axis=-1) * xx, axis=-1) / dRdZ
    )
    return vals.reshape(points_shape[1:])
