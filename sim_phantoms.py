from __future__ import annotations

from typing import TYPE_CHECKING, Union
from types import ModuleType

import array_api_compat.numpy as np

if TYPE_CHECKING:
    import cupy as cp

    Array = Union[np.ndarray, cp.ndarray]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime


def pet_phantom(
    in_shape: tuple[int, int, int], xp: ModuleType, dev, mu_value: float = 0.01, type=-1
) -> tuple[Array, Array]:

    if type == -1:
        # box phantom
        x_em = xp.ones(in_shape, device=dev, dtype=xp.float32)
        c0 = in_shape[0] // 2
        c1 = in_shape[1] // 2
        x_em[(c0 - 4) : (c0 + 4), (c1 - 4) : (c1 + 4), :] = 3.0

        x_em[28:32, c1 : (c1 + 4), :] = 5.0
        x_em[c0 : (c0 + 4), 20:24, :] = 5.0

        x_em[-32:-28, c1 : (c1 + 4), :] = 0.1
        x_em[c0 : (c0 + 4), -24:-20, :] = 0.1

        x_em[:25, :, :] = 0
        x_em[-25:, :, :] = 0
        x_em[:, :10, :] = 0
        x_em[:, -10:, :] = 0

        x_em[:, :, :2] = 0
        x_em[:, :, -2:] = 0

        x_att = mu_value * xp.astype(x_em > 0, xp.float32)
    elif type == 1:
        # elliptical phantom
        x_em = xp.zeros(in_shape, device=dev, dtype=xp.float32)
        x_att = xp.zeros(in_shape, device=dev, dtype=xp.float32)
        c0 = in_shape[0] / 2
        c1 = in_shape[1] / 2
        a = in_shape[0] / 2.5  # semi-major axis
        b = in_shape[1] / 4  # semi-minor axis

        rix = in_shape[0] / 25
        riy = in_shape[1] / 25

        y, x = xp.ogrid[: in_shape[0], : in_shape[1]]

        outer_mask = ((x - c0) / a) ** 2 + ((y - c1) / b) ** 2 <= 1
        inner_mask = ((x - c0) / rix) ** 2 + ((y - c1) / riy) ** 2 <= 1

        for z in range(in_shape[2]):
            x_em[:, :, z][outer_mask] = 3.0
            x_em[:, :, z][inner_mask] = 0.0

            x_att[:, :, z][outer_mask] = mu_value
            x_att[:, :, z][inner_mask] = mu_value / 3

        x_em[:, :, :2] = 0
        x_em[:, :, -2:] = 0

        x_att[:, :, :2] = 0
        x_att[:, :, -2:] = 0

        x_att[:, :, 1][outer_mask] = mu_value
        x_att[:, :, -2][outer_mask] = mu_value
    else:
        raise ValueError("Invalid phantom type")

    return x_em, x_att


if __name__ == "__main__":
    import numpy as np
    import pymirc.viewer as pv

    a, b = pet_phantom((129, 129, 33), np, "cpu", type=1)

    vi = pv.ThreeAxisViewer([a, b])
