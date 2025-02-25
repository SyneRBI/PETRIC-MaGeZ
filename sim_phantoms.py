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
    in_shape: tuple[int, int, int],
    xp: ModuleType,
    dev,
    mu_value: float = 0.01,
    add_spheres: bool = True,
    add_inner_cylinder: bool = True,
    r0=0.45,
    r1=0.28,
) -> tuple[Array, Array]:
    """
    Generate a 3D PET phantom.

    Parameters
    ----------
    in_shape : tuple
        Shape of the phantom.
    xp : module
        Array API module to use.
    dev : str
        Device to use.
    mu_value : float
        Attenuation coefficient.
    Returns
    -------
    tuple
        Emission and attenuation images.

    Note
    ----

    The activity in the background should be 1.
    """
    if dev == "cpu":
        dev = None

    oversampling_factor = 4

    # elliptical phantom on oversampled grid
    oversampled_shape = tuple(oversampling_factor * x for x in in_shape)
    x_em_oversampled = xp.zeros(oversampled_shape, device=dev, dtype=xp.float32)
    x_att_oversampled = xp.zeros(oversampled_shape, device=dev, dtype=xp.float32)
    c0 = oversampled_shape[0] / 2
    c1 = oversampled_shape[1] / 2
    c2 = oversampled_shape[2] / 2
    a = r0 * oversampled_shape[0]  # semi-major axis
    b = r1 * oversampled_shape[1]  # semi-minor axis

    rix = oversampled_shape[0] / 25
    riy = oversampled_shape[1] / 25

    y, x = xp.ogrid[: oversampled_shape[0], : oversampled_shape[1]]

    outer_mask = ((x - c0) / a) ** 2 + ((y - c1) / b) ** 2 <= 1
    inner_mask = ((x - c0) / rix) ** 2 + ((y - c1) / riy) ** 2 <= 1

    for z in range(oversampled_shape[2]):
        x_em_oversampled[:, :, z][outer_mask] = 1.0
        x_att_oversampled[:, :, z][outer_mask] = mu_value

        if add_inner_cylinder:
            x_em_oversampled[:, :, z][inner_mask] = 0.25
            x_att_oversampled[:, :, z][inner_mask] = mu_value / 3

    # add a few spheres to the emission image

    if add_spheres:
        x, y, z = xp.ogrid[
            : oversampled_shape[0], : oversampled_shape[1], : oversampled_shape[2]
        ]

        r_sp = 3 * [oversampled_shape[2] / 9]
        r_sp2 = 3 * [oversampled_shape[2] / 17]

        for z_offset in [c2, 0.45 * c2]:
            sp_mask = ((x - c0) / r_sp[0]) ** 2 + ((y - 1.4 * c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask] = 2.5

            sp_mask2 = ((x - 1.3 * c0) / r_sp[0]) ** 2 + ((y - c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask2] = 0.25

            sp_mask = ((x - c0) / r_sp2[0]) ** 2 + ((y - 0.6 * c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask] = 2.5

            sp_mask2 = ((x - 0.7 * c0) / r_sp2[0]) ** 2 + ((y - c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask2] = 0.25

    # downsample to original grid size by averaging
    x_em_oversampled = (
        x_em_oversampled[::4, :, :]
        + x_em_oversampled[1::4, :, :]
        + x_em_oversampled[2::4, :, :]
        + x_em_oversampled[3::4, :, :]
    )
    x_em_oversampled = (
        x_em_oversampled[:, ::4, :]
        + x_em_oversampled[:, 1::4, :]
        + x_em_oversampled[:, 2::4, :]
        + x_em_oversampled[:, 3::4, :]
    )
    x_em_oversampled = (
        x_em_oversampled[:, :, ::4]
        + x_em_oversampled[:, :, 1::4]
        + x_em_oversampled[:, :, 2::4]
        + x_em_oversampled[:, :, 3::4]
    )

    x_att_oversampled = (
        x_att_oversampled[::4, :, :]
        + x_att_oversampled[1::4, :, :]
        + x_att_oversampled[2::4, :, :]
        + x_att_oversampled[3::4, :, :]
    )
    x_att_oversampled = (
        x_att_oversampled[:, ::4, :]
        + x_att_oversampled[:, 1::4, :]
        + x_att_oversampled[:, 2::4, :]
        + x_att_oversampled[:, 3::4, :]
    )
    x_att_oversampled = (
        x_att_oversampled[:, :, ::4]
        + x_att_oversampled[:, :, 1::4]
        + x_att_oversampled[:, :, 2::4]
        + x_att_oversampled[:, :, 3::4]
    )

    x_em = x_em_oversampled.copy() / (oversampling_factor**3)
    x_att = x_att_oversampled.copy() / (oversampling_factor**3)

    x_em[:, :, :3] = 0
    x_em[:, :, -3:] = 0

    # make the attenuation a bit wider in z (plastic wall)
    x_att[:, :, :2] = 0
    x_att[:, :, -2:] = 0

    return x_em, x_att


if __name__ == "__main__":
    import numpy as np

    sh = (129, 129, 33)

    e1, a1 = pet_phantom(sh, np, "cpu")
    e2, a2 = pet_phantom(
        sh, np, "cpu", add_spheres=True, add_inner_cylinder=False, r0=0.25, r1=0.25
    )

    # import pymirc.viewer as pv
    # vi = pv.ThreeAxisViewer([e1, a1, e2, a2])
