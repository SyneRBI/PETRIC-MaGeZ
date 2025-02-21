"""script to compare SPDHG, SGD and SVRG for PET reconstruction with quad. and RDP prior"""

# %%
from __future__ import annotations

try:
    import array_api_compat.cupy as xp
except ImportError:
    import array_api_compat.numpy as xp

import parallelproj
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

from array_api_compat import to_device
from scipy.optimize import fmin_l_bfgs_b
from pathlib import Path

import argparse

from sim_utils import (
    SubsetNegPoissonLogLWithPrior,
    split_fwd_model,
    OSEM,
    SVRG,
    validate_stepsize_lambda_str,
    MLEMPreconditioner,
    HarmonicPreconditioner,
)

from rdp import RDP
from sim_phantoms import pet_phantom

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--true_counts", type=int, default=int(1e7))
parser.add_argument("--beta_rel", type=float, default=1.0)
parser.add_argument(
    "--step_size_func",
    type=validate_stepsize_lambda_str,
    help="Lambda function mapping update [int] to stepsize [float] (e.g., 'lambda x: float(x**2)')",
    default="lambda x: 1.0",
)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--num_subsets", type=int, default=27)
parser.add_argument("--precond_type", type=int, default=2, choices=[1, 2])
parser.add_argument("--phantom_type", type=int, default=1, choices=[-1, 1])

args = parser.parse_args()

# %%
# input parameters

seed = int(args.seed)

# true counts, reasonable range
true_counts = args.true_counts
# regularization weight, reasonable range: 5e-5 * (true_counts / 1e6) is medium regularization
beta = args.beta_rel * (5e-4) * (true_counts / 1e7)
# RDP gamma parameter
gamma_rdp = 2.0
# type of preconditioner: 1 MLEM, 2: Harmonic
precond_type = args.precond_type

# number of epochs / subsets for stochastic gradient algorithms
num_epochs = args.num_epochs
num_subsets = args.num_subsets

# max number of updates for reference L-BFGS-B solution
num_iter_bfgs_ref = 500

# %%
# number of rings of simulated PET scanner, should be odd in this example
num_rings = 17
# resolution of the simulated PET scanner in mm
fwhm_data_mm = 4.5
# simulated TOF or non-TOF system
tof = False
# mean of contamination sinogram, relative to mean of trues sinogram, reasonable range: 0.5 - 1.0
contam_fraction = 0.5
# show the geometry of the scanner / image volume
show_geometry = False
# verbose output
verbose = True
# track cost function values after every update (slow)
track_cost = True

# number of epochs / subsets for intial OSEM
num_epochs_osem = 1
num_subsets_osem = 27

# step size update function
step_size_func = args.step_size_func

# phantom type (int)
phantom_type = args.phantom_type

# %%
# random seed
np.random.seed(seed)

# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=300.0,
    num_sides=36,
    num_lor_endpoints_per_side=12,
    lor_spacing=4.0,
    ring_positions=xp.linspace(
        -5 * (num_rings - 1) / 2, 5 * (num_rings - 1) / 2, num_rings
    ),
    symmetry_axis=2,
)

# %%
# setup the LOR descriptor that defines the sinogram

img_shape = (161, 161, 2 * num_rings - 1)
voxel_size = (2.5, 2.5, 2.5)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=40,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

if lor_desc.num_views % num_subsets != 0:
    raise ValueError(
        f"num_subsets_sgd ({num_subsets}) must be a divisor of num_views ({lor_desc.num_views})"
    )
if lor_desc.num_views % num_subsets_osem != 0:
    raise ValueError(
        f"num_subsets_osem  ({num_subsets_osem}) must be a divisor of num_views ({lor_desc.num_views})"
    )

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# %%
# setup of true emission and attenuation image
print("setting up phantom")
x_true, x_att = pet_phantom(img_shape, xp, dev, mu_value=0.01, type=phantom_type)

# %%
# calculate the attenuation sinogram
print("calculating attenuation sinogram")
att_sino = xp.exp(-proj(x_att))

# %%
# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
if tof is True:
    proj.tof_parameters = parallelproj.TOFParameters(
        num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
    )

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
if proj.tof:
    att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino
    )
else:
    att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=fwhm_data_mm / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

# %%
if show_geometry:
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    proj.show_geometry(ax)
    lor_desc.show_views(ax, xp.asarray([0, lor_desc.num_views // 4]), xp.asarray([1]))
    fig.show()

# %%
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

if true_counts > 0:
    scale_fac = true_counts / float(xp.sum(noise_free_data))
    noise_free_data *= scale_fac
    x_true *= scale_fac

# generate a contant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    contam_fraction * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
data = xp.asarray(
    np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

# %%
# run quick OSEM with one iteration

pet_subset_lin_op_seq_osem, subset_slices_osem = split_fwd_model(
    pet_lin_op, num_subsets_osem
)

data_fidelity = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq_osem, contamination, subset_slices_osem
)

x0 = xp.ones(pet_lin_op.in_shape, device=dev, dtype=xp.float32)
osem_alg = OSEM(data_fidelity)
x_osem = osem_alg.run(x0, num_epochs_osem)

x_init = x_osem.copy()

# %%
# setup of the cost function

fwd_ones = pet_lin_op(xp.ones(pet_lin_op.in_shape, device=dev, dtype=xp.float32))
fwd_osem = pet_lin_op(x_osem) + contamination
kappa_img = xp.sqrt(pet_lin_op.adjoint((data * fwd_ones) / (fwd_osem**2)))

prior = RDP(
    img_shape,
    xp=xp,
    dev=dev,
    voxel_size=xp.asarray(voxel_size, device=dev),
    eps=float(xp.max(x_osem)) / 100,
    gamma=gamma_rdp,
)

prior.kappa = kappa_img
prior.scale = beta

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, device=dev, dtype=xp.float32)
)

## %%
# x = x_init.copy()
# h = prior.diag_hessian(x)
# d_data = to_device(adjoint_ones, "cpu")
# d_prior = to_device(h * x, "cpu")
#
## %%
# import pymirc.viewer as pv
# vi = pv.ThreeAxisViewer([d_data, d_prior, d_data > d_prior])

# %%


pet_subset_lin_op_seq, subset_slices = split_fwd_model(pet_lin_op, num_subsets)

cost_function = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq, contamination, subset_slices, prior=prior
)


# %%
# run (pre-conditioned) L-BFGS-B without subsets as reference

# the preconditioner is taken from
# Tsai et al: "Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors in PET Reconstruction"
# doi: 10.1109/TMI.2019.2913889

# calculate data term preconditioner
pp = 1 / xp.sqrt(kappa_img**2 + 1e-4)
pp_np = xp.asnumpy(pp).ravel()

precond_cf = lambda z: cost_function(z * pp_np)
precond_grad = lambda z: cost_function.gradient(z * pp_np) * pp_np

z0_bfgs = to_device((x_init / pp).ravel(), "cpu")

bounds = x0.size * [(0, None)]

sim_path = Path("sim_results")
sim_path.mkdir(exist_ok=True)

ref_file = (
    sim_path
    / f"rdp_t_{true_counts:.2E}_b_{beta:.2E}_g_{gamma_rdp:.2E}_n_{num_iter_bfgs_ref}_nr_{num_rings}_tof_{tof}_cf_{contam_fraction}_s_{seed}_ph_{phantom_type}.npy"
)

if ref_file.exists():
    print("loading L-BFGS-B reference")
    x_ref = xp.asarray(np.load(ref_file), device=dev)
else:
    print("running L-BFGS-B reference")
    res = fmin_l_bfgs_b(
        precond_cf,
        z0_bfgs,
        precond_grad,
        disp=1,
        maxiter=num_iter_bfgs_ref,
        bounds=bounds,
        m=10,
        factr=1.0,
        pgtol=1e-16,
    )

    x_ref = pp * xp.asarray(res[0].reshape(img_shape), device=dev)
    xp.save(ref_file, x_ref)

# %%
cost_osem = cost_function(x_osem)
cost_ref = cost_function(x_ref)
x_osem_scale = float(xp.mean(x_init))

print(cost_ref)


# %%
# define NRMSE callback
# NRMSE = global MSE divided by the background signal (scale_fac)


def nrmse(vec_x, vec_y, norm):
    return float(xp.sqrt(xp.mean((vec_x - vec_y) ** 2))) / norm


nmrse_callback = lambda x: nrmse(x, x_ref, norm=scale_fac)

# %%
nrmse_osem = nmrse_callback(x_osem)
nrmse_init = nmrse_callback(x_init)


# %%
# run SVRG
# setup subset negative Poisson log-likelihood without prior
subset_neglogL = SubsetNegPoissonLogLWithPrior(
    data, pet_subset_lin_op_seq, contamination, subset_slices, prior=None
)

# setup the preconditioner
if precond_type == 1:
    diag_precond = MLEMPreconditioner(adjoint_ones)
elif precond_type == 2:
    diag_precond = HarmonicPreconditioner(adjoint_ones, prior=prior, factor=2.0)
else:
    raise ValueError("invalid preconditioner version")

# add a gentle Gaussian pre-filter to the preconditioner
# otherwise we get very high values in slices that are close to the
# edge of the FOV and contain a strong gradient in z (e.g. edge of the phantom)
diag_precond.filter_function = parallelproj.GaussianFilterOperator(
    img_shape, sigma=3.0 / (2.35 * xp.asarray(voxel_size))
)

print("running SVRG")
svrg_alg = SVRG(
    subset_neglogL,
    prior,
    x_init,
    diag_precond_func=diag_precond.__call__,
    verbose=False,
    step_size_func=step_size_func,
)
nrmse_svrg = svrg_alg.run(num_epochs * num_subsets, callback=nmrse_callback)
cost_svrg = cost_function(svrg_alg.x)

print(f"cost ref   : {cost_ref:.8E}")
print(f"cost svrg .: {cost_svrg:.8E}")

if cost_svrg < cost_ref:
    print("WARNING: SVRG cost is lower than reference cost. Find better reference.")

# %%
fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)

sl0 = x_true.shape[1] // 2
sl1 = x_true.shape[2] // 2

ims = dict(cmap="Greys", vmin=0, vmax=1.1 * float(xp.max(x_true)))
ims_diff = dict(cmap="seismic", vmin=-0.05, vmax=0.05)

ax[0, 0].imshow(to_device(x_ref[..., sl1], "cpu"), **ims)
im0 = ax[1, 0].imshow(to_device(x_ref[:, sl0, :].T, "cpu"), **ims)
ax[0, 0].set_title(f"reference (L-BFGS-B)", fontsize="medium")

ax[0, 1].imshow(to_device(svrg_alg.x[..., sl1], "cpu"), **ims)
im1 = ax[1, 1].imshow(to_device(svrg_alg.x[:, sl0, :].T, "cpu"), **ims)
ax[0, 1].set_title(
    f"SVRG {num_epochs} epochs, {num_subsets} subsets", fontsize="medium"
)

ax[0, 2].imshow(
    to_device((svrg_alg.x[..., sl1] - x_ref[..., sl1]) / scale_fac, "cpu"), **ims_diff
)
im2 = ax[1, 2].imshow(
    to_device((svrg_alg.x[:, sl0, :].T - x_ref[:, sl0, :].T) / scale_fac, "cpu"),
    **ims_diff,
)
ax[0, 2].set_title(f"(SVRG - ref.) / bg. activity", fontsize="medium")

fig.colorbar(im0, ax=ax[1, 0], location="bottom", fraction=0.05)
fig.colorbar(im1, ax=ax[1, 1], location="bottom", fraction=0.05)
fig.colorbar(im2, ax=ax[1, 2], location="bottom", fraction=0.05)


update_arr = np.arange(num_subsets * num_epochs)

ax[0, -1].semilogy(update_arr / num_subsets, nrmse_svrg, label="SVRG")
ax[0, -1].axhline(0.01, color="black", ls="--")
ax[0, -1].set_title("whole image NRMSE", fontsize="medium")
ax[0, -1].set_xlabel("epoch")
ax[0, -1].grid(ls=":")
ax[0, -1].legend()

ax[1, -1].plot(
    update_arr / num_subsets, [svrg_alg.step_size_func(x) for x in update_arr]
)
ax[1, -1].set_title("step size", fontsize="medium")
ax[1, -1].set_xlabel("epoch")
ax[1, -1].grid(ls=":")

fig.show()
