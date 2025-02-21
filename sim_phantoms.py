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
        oversampling_factor = 4

        # elliptical phantom on oversampled grid
        oversampled_shape = tuple(oversampling_factor * x for x in in_shape)
        x_em_oversampled = xp.zeros(oversampled_shape, device=dev, dtype=xp.float32)
        x_att_oversampled = xp.zeros(oversampled_shape, device=dev, dtype=xp.float32)
        c0 = oversampled_shape[0] / 2
        c1 = oversampled_shape[1] / 2
        c2 = oversampled_shape[2] / 2
        a = oversampled_shape[0] / 2.5  # semi-major axis
        b = oversampled_shape[1] / 4  # semi-minor axis

        rix = oversampled_shape[0] / 25
        riy = oversampled_shape[1] / 25

        y, x = xp.ogrid[: oversampled_shape[0], : oversampled_shape[1]]

        outer_mask = ((x - c0) / a) ** 2 + ((y - c1) / b) ** 2 <= 1
        inner_mask = ((x - c0) / rix) ** 2 + ((y - c1) / riy) ** 2 <= 1

        for z in range(oversampled_shape[2]):
            x_em_oversampled[:, :, z][outer_mask] = 3.0
            x_em_oversampled[:, :, z][inner_mask] = 0.0

            x_att_oversampled[:, :, z][outer_mask] = mu_value
            x_att_oversampled[:, :, z][inner_mask] = mu_value / 3

        # add a few spheres to the emission image

        x, y, z = xp.ogrid[
            : oversampled_shape[0], : oversampled_shape[1], : oversampled_shape[2]
        ]

        r_sp = 3 * [oversampled_shape[2] / 9]
        r_sp2 = 3 * [oversampled_shape[2] / 17]

        for z_offset in [c2, 0.35 * c2]:
            sp_mask = ((x - c0) / r_sp[0]) ** 2 + ((y - 1.4 * c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask] = 6.0

            sp_mask2 = ((x - 1.3 * c0) / r_sp[0]) ** 2 + ((y - c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask2] = 1.0

            sp_mask = ((x - c0) / r_sp2[0]) ** 2 + ((y - 0.6 * c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask] = 6.0

            sp_mask2 = ((x - 0.7 * c0) / r_sp2[0]) ** 2 + ((y - c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask2] = 1.0

        # downsample to original grid size by averaging
        x_em = xp.zeros(in_shape, device=dev, dtype=xp.float32)
        x_att = xp.zeros(in_shape, device=dev, dtype=xp.float32)

        for i in range(in_shape[0]):
            for j in range(in_shape[1]):
                for k in range(in_shape[2]):
                    x_em[i, j, k] = xp.mean(
                        x_em_oversampled[
                            oversampling_factor * i : oversampling_factor * (i + 1),
                            oversampling_factor * j : oversampling_factor * (j + 1),
                            oversampling_factor * k : oversampling_factor * (k + 1),
                        ]
                    )
                    x_att[i, j, k] = xp.mean(
                        x_att_oversampled[
                            oversampling_factor * i : oversampling_factor * (i + 1),
                            oversampling_factor * j : oversampling_factor * (j + 1),
                            oversampling_factor * k : oversampling_factor * (k + 1),
                        ]
                    )

        x_em[:, :, :2] = 0
        x_em[:, :, -2:] = 0

        x_att[:, :, 0] = 0
        x_att[:, :, -1] = 0

    else:
        raise ValueError("Invalid phantom type")

    return x_em, x_att


if __name__ == "__main__":
    import numpy as np
    import pymirc.viewer as pv

    a, b = pet_phantom((129, 129, 33), np, "cpu", type=1)

    vi = pv.ThreeAxisViewer([a, b])
