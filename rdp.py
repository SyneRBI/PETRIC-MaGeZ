from __future__ import annotations

from typing import TYPE_CHECKING, Union
from types import ModuleType

import abc
import array_api_compat.numpy as np
from array_api_compat import device, to_device

if TYPE_CHECKING:
    import cupy as cp

    Array = Union[np.ndarray, cp.ndarray]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime

def neighbor_offsets(ndim, xp):
    # all offsets in {-1,0,1}^ndim except 0
    grid = xp.stack(xp.meshgrid(*([xp.arange(-1, 2)] * ndim), indexing="ij"), axis=-1)
    offsets = grid.reshape(-1, ndim)
    # remove zero offset
    mask = xp.any(offsets != 0, axis=1)
    return offsets[mask]  # shape: (num_neigh, ndim)


def neighbor_difference_and_sum(x, xp):
    """
    x: array_api-compatible array, shape (n1, n2, ..., nD)
    returns:
        d, s: shape (num_neigh, n1, n2, ..., nD)
              differences and sums with all nearest neighbors.
              If a neighbor is outside the image, the corresponding
              entry is 0 by definition.
    """
    ndim = x.ndim
    shape = x.shape
    offsets = neighbor_offsets(ndim, xp)          # (num_neigh, ndim)
    num_neigh = offsets.shape[0]

    d = xp.zeros((num_neigh,) + shape, dtype=x.dtype)
    s = xp.zeros((num_neigh,) + shape, dtype=x.dtype)

    for k in range(num_neigh):
        off = offsets[k]  # e.g. [-1, 0, 1, ...], shape (ndim,)

        # build slices for "center" and "neighbor" so that
        # center[...] and neighbor[...] have exactly the same shape
        center_sl = []
        neigh_sl = []
        for ax, (n, o) in enumerate(zip(shape, off)):
            if o == 0:
                # full axis in both
                center_sl.append(slice(0, n))
                neigh_sl.append(slice(0, n))
            elif o > 0:
                # neighbor is shifted +o → valid center is [0 : n-o]
                center_sl.append(slice(0, n - o))
                neigh_sl.append(slice(o, n))
            else:  # o < 0
                # neighbor is shifted -|o| → valid center is [|o| : n]
                o_abs = -o
                center_sl.append(slice(o_abs, n))
                neigh_sl.append(slice(0, n - o_abs))

        center_sl = tuple(center_sl)
        neigh_sl = tuple(neigh_sl)

        # view with aligned shapes
        xc = x[center_sl]
        xn = x[neigh_sl]

        # place results into the corresponding region of d[k], s[k]
        # outside that region, d and s remain zero
        d_view = d[(k,) + center_sl]
        s_view = s[(k,) + center_sl]

        d_view[...] = xc - xn
        s_view[...] = xc + xn

    return d, s


def neighbor_product(x, xp):
    """
    Same neighbor definition as neighbor_difference_and_sum, but returns
    products x * neighbor. Products with neighbors outside the image are 0.
    """
    ndim = x.ndim
    shape = x.shape
    offsets = neighbor_offsets(ndim, xp)
    num_neigh = offsets.shape[0]

    p = xp.zeros((num_neigh,) + shape, dtype=x.dtype)

    for k in range(num_neigh):
        off = offsets[k]

        center_sl = []
        neigh_sl = []
        for ax, (n, o) in enumerate(zip(shape, off)):
            if o == 0:
                center_sl.append(slice(0, n))
                neigh_sl.append(slice(0, n))
            elif o > 0:
                center_sl.append(slice(0, n - o))
                neigh_sl.append(slice(o, n))
            else:  # o < 0
                o_abs = -o
                center_sl.append(slice(o_abs, n))
                neigh_sl.append(slice(0, n - o_abs))

        center_sl = tuple(center_sl)
        neigh_sl = tuple(neigh_sl)

        xc = x[center_sl]
        xn = x[neigh_sl]

        p_view = p[(k,) + center_sl]
        p_view[...] = xc * xn

    return p


class SmoothFunction(abc.ABC):

    def __init__(self, in_shape, xp, dev, scale: float = 1.0) -> None:

        self._in_shape = in_shape
        self._scale = scale
        self._xp = xp
        self._dev = dev

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def xp(self):
        return self._xp

    @property
    def dev(self):
        return self._dev

    @abc.abstractmethod
    def _call(self, x: Array) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _gradient(self, x: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._call(x)
        else:
            res = self._scale * self._call(x)

        return res

    def gradient(self, x: Array) -> Array:
        dev_input = device(x)

        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._gradient(x)
        else:
            res = self._scale * self._gradient(x)

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        res = to_device(res, dev_input)

        return res

    def prox_function(self, z: Array, x: Array, T: Array) -> float:
        """returns the function h(z) = \sum_i f_i(z) + 0.5 * \sum_i (z_i - x_i)^2 / T_i
        which when minimized over z is the proximal operator of the function at x
        """
        return self.__call__(z) + 0.5 * float((((z - x) ** 2) / T).sum())

    def prox_gradient(self, z: Array, x: Array, T: Array) -> Array:
        """return the gradient of the prox function h(z), needed for numeric evaluation of the proximal operator"""
        return self.gradient(z) + (z - x) / T


class SmoothFunctionWithDiagonalHessian(SmoothFunction):
    @abc.abstractmethod
    def _diag_hessian(self, x: Array) -> Array:
        """(approximation) of the diagonal of the Hessian"""
        raise NotImplementedError

    def diag_hessian(self, x: Array) -> Array:
        dev_input = device(x)

        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._diag_hessian(x)
        else:
            res = self._scale * self._diag_hessian(x)

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        res = to_device(res, dev_input)

        return res


class RDP(SmoothFunctionWithDiagonalHessian):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        xp: ModuleType,
        dev: str,
        voxel_size: Array,
        eps: float | None = None,
        gamma: float = 2.0,
    ) -> None:
        self._gamma = gamma

        if eps is None:
            self._eps = xp.finfo(xp.float64).eps
        else:
            self._eps = eps

        self._ndim = len(in_shape)

        super().__init__(in_shape=in_shape, xp=xp, dev=dev)

        # number of nearest neighbors
        self._num_neigh = 3**self._ndim - 1

        self._voxel_size = voxel_size

        # array for differences and sums with nearest neighbors
        self._voxel_size_weights = xp.zeros(
            (self._num_neigh,) + in_shape, dtype=xp.float64
        )

        for i, ind in enumerate(xp.ndindex(self._ndim * (3,))):
            if i != (self._num_neigh // 2):
                offset = xp.asarray(ind, device=dev) - 1

                vw = voxel_size[2] / xp.linalg.norm(offset * voxel_size)

                if i < self._num_neigh // 2:
                    self._voxel_size_weights[i, ...] = vw
                else:
                    self._voxel_size_weights[i - 1, ...] = vw

        self._weights = self._voxel_size_weights
        self._kappa = None

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def weights(self) -> Array:
        return self._weights

    @property
    def kappa(self) -> Array | None:
        return self._kappa

    @kappa.setter
    def kappa(self, image: Array) -> None:
        self._kappa = image
        self._weights = (
            neighbor_product(self._kappa, self._xp) * self._voxel_size_weights
        )

    def _call(self, x: Array) -> float:

        if float(self.xp.min(x)) < 0:
            return self.xp.inf

        d, s = neighbor_difference_and_sum(x, self.xp)
        phi = s + self.gamma * self.xp.abs(d) + self.eps

        tmp = (d**2) / phi

        if self._weights is not None:
            tmp *= self._weights

        return 0.5 * float(self.xp.sum(tmp, dtype=self.xp.float64))

    def _gradient(self, x: Array) -> Array:
        d, s = neighbor_difference_and_sum(x, self.xp)
        phi = s + self.gamma * self.xp.abs(d) + self.eps

        tmp = d * (2 * phi - (d + self.gamma * self.xp.abs(d))) / (phi**2)

        if self._weights is not None:
            tmp *= self._weights

        return tmp.sum(axis=0)

    def _diag_hessian(self, x: Array) -> Array:
        d, s = neighbor_difference_and_sum(x, self.xp)
        phi = s + self.gamma * self.xp.abs(d) + self.eps

        tmp = ((s - d + self.eps) ** 2) / (phi**3)

        if self._weights is not None:
            tmp *= self._weights

        return 2 * tmp.sum(axis=0)
